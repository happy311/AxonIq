"""
AxonIQ — Deterministic MS Symptom Pattern Database (ported from NeuroCheck v9.1)

This module is a NON-LLM, regex + NegEx-based classifier. It exists to act as
a grounding / floor layer alongside the LangGraph LLM classification in
node_llm.py — not to replace it.

Why this exists (2026-09):
  AxonIQ's tier/feature state came entirely from the LLM's own per-turn JSON
  classification, with only 3 hardcoded deterministic floors (optic neuritis,
  Lhermitte's, Uhthoff's) in node_llm._apply_clinical_floor(). Two failure
  modes were observed:
    1. The LLM under-tiers a real, matching symptom (e.g. calls it WATCH
       instead of the MODERATE the system prompt itself defines for "1
       confirmed MS symptom"), delaying or entirely skipping the MRI request
       — node_goal_setter's `turns >= 6 -> conclude (no MRI)` catch-all then
       fires on a genuine case.
    2. Free-text LLM classification is an open hallucination surface with no
       independent check.

  NeuroCheck (the predecessor build) had already solved symptom detection
  deterministically: 64 SYMPTOM_PATTERNS categories + 19 ANCHOR_FEATURES +
  15 LOW_PRIORITY_PATTERNS, all NegEx negation-checked, validated on 150
  cases. This module ports that database and its scoring logic verbatim,
  and exposes scan_symptom_db() for node_llm.py to call every turn on the
  cumulative conversation text — its tier/features are unioned with the
  LLM's own output (max tier, union features), never replacing them.

Source: neurocheck/api/main.py (SYMPTOM_PATTERNS, ANCHOR_FEATURES,
LOW_PRIORITY_PATTERNS, EMERGENCY_FLAGS, SEVERITY_MODIFIERS, classify()).
Ported unchanged except: (a) wrapped in scan_symptom_db() instead of a
FastAPI route, (b) added a DIS_REGION_MAP so anchor/feature hits can also
populate AgentState.dis_regions for the McDonald 2017 fast-track in
node_goal_setter.py, (c) EMERGENCY_FLAGS kept here for reference/coverage
comparison only — AxonIQ's live emergency path is still
api/agent/tools/emergency.py; the two pattern sets have NOT been merged in
this pass (worth a follow-up diff).
"""
from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple, TypedDict

from spacy.lang.en import English
from negspacy.termsets import termset
from negspacy.negation import Negex  # noqa: F401 — registers the 'negex' factory


EMERGENCY_FLAGS: Dict[str, dict] = {
    "stroke_suspected": {
        "patterns": [r"(sudden|woke up).{0,20}(face droop|arm weak|slurred speech)"],
        "label": "Possible Acute Stroke",
        "action": "Call emergency services immediately (999 / 911). Do not wait — time is critical.",
    },
    "meningitis_suspected": {
        "patterns": [
            r"(fever|high temp).{0,30}(neck).{0,20}(stiff|rigid|pain)",
            r"photophobia.{0,20}fever",
        ],
        "label": "Possible Meningitis / Encephalitis",
        "action": "Seek emergency care immediately. Meningitis is life-threatening.",
    },
    "hemorrhage_suspected": {
        "patterns": [r"(sudden|worst).{0,20}(headache).{0,20}(life|ever|thunderclap)"],
        "label": "Possible Subarachnoid Haemorrhage",
        "action": "Call emergency services immediately. Thunderclap headache requires emergency evaluation.",
    },
    "gbs_suspected": {
        "patterns": [
            r"(both legs|feet).{0,30}(weak|paralyzed).{0,30}(moving up|spreading up|hands)",
            r"ascending paralysis",
            r"weakness.{0,30}(spreading|moving up).{0,20}(arm|hand)",
        ],
        "label": "Possible Guillain-Barré Syndrome",
        "action": "Seek emergency care immediately. Ascending weakness requires urgent neurological assessment.",
    },
}


ANCHOR_FEATURES = {
    # ── Visual / optic ────────────────────────────────────────────────────────
    "lhermittes_sign",
    "lhermittes_variant",
    "uhthoff_phenomenon",
    "pain_on_eye_movement",
    "monocular_vision_loss",
    "ino_features",
    "ms_hug",
    "oscillopsia",
    "bilateral_vision_loss",
    "diplopia",
    # ── Brainstem / TN ────────────────────────────────────────────────────────
    "paroxysmal_tn_young",
    "area_postrema",
    # ── Spinal / sensory ──────────────────────────────────────────────────────
    "sensory_idioms",
    "limb_numbness",
    # ── Ocular / IIH overlap ──────────────────────────────────────────────────
    "optic_neuritis_idioms",
    "obesity_disc_flag",
    # ── History / DIT ─────────────────────────────────────────────────────────
    "prior_cis_episode",
    # ── Vestibular ────────────────────────────────────────────────────────────
    "vertigo_ms",
    # ── Paroxysmal / rare (v8.1) ──────────────────────────────────────────────
    "paroxysmal_ms",
    "tonic_spasms_ms",
    "tumefactive_ms",
}


SYMPTOM_PATTERNS: Dict[str, List[tuple]] = {

    # ── ANCHOR: Lhermitte's sign ──────────────────────────────────────────────
    "lhermittes_sign": [
        (r"electric.{0,20}(shock|jolt|current|zap).{0,30}spine", "Lhermitte's sign — electric shock down spine"),
        (r"(shock|jolt|current|zap).{0,30}(when|if|on).{0,20}(bend|bow|look|flex|tilt).{0,20}(neck|head|chin|down)", "Shock on neck bending"),
        (r"(sharp|sudden).{0,20}(electric|zap|jolt|current).{0,20}(run|shoot|travel).{0,30}(spine|leg|back)", "Lhermitte-type electric sensation"),
        (r"\blhermitte.{0,20}(sign|phenomenon|symptom)\b", "Lhermitte's sign (clinical)"),
    ],

    # ── ANCHOR: Lhermitte variant (cough/sneeze trigger) — v8.1 ─────────────
    "lhermittes_variant": [
        (r"(shock|jolt|electric).{0,20}(cough|coughing|sneeze|sneezing|limb movement)", "Lhermitte variant — cough/sneeze trigger"),
        (r"\blhermitte.{0,20}(cough|movement|limb|trigger|variant)\b", "Lhermitte variant trigger"),
    ],

    # ── ANCHOR: Uhthoff's phenomenon ─────────────────────────────────────────
    # v8.1: must have visual OR gait/balance component to distinguish from
    #       generic heat sensitivity (LP category)
    "uhthoff_phenomenon": [
        (r"(vision|sight|see|eyesight).{0,30}(blur|worse|bad|dim|grey|gray|dark).{0,30}(hot|heat|shower|bath|exercise|warm|fever)", "Uhthoff's — vision worse in heat"),
        (r"(hot|heat|shower|bath|exercise|warm|fever).{0,30}(vision|sight|see|eyesight).{0,30}(blur|worse|bad|dim)", "Uhthoff's — heat worsens vision"),
        (r"(shimmer|shimmers|obscur).{0,40}(hot shower|warm shower|shower|bath|exercise|heat|fever)", "Uhthoff's — shimmer in heat"),
        (r"worsening.{0,20}(vision|balance|weakness|gait).{0,30}(exercise|warm|heat|hot|fever)", "Uhthoff's — visual/gait worsening in heat"),
        (r"\buhthoff.{0,20}(phenomenon|sign|syndrome)\b", "Uhthoff's phenomenon (clinical)"),
        (r"heat intolerance.{0,40}(gait|walk|stumbl|fall|vision)", "Uhthoff's — heat intolerance with visual/gait"),
    ],

    # ── ANCHOR: Pain on eye movement ─────────────────────────────────────────
    "pain_on_eye_movement": [
        (r"pain.{0,30}(mov|look).{0,15}eye", "Pain on eye movement"),
        (r"(move|moving|turning|side.to.side).{0,20}(eye|eyes).{0,30}(pain|hurt|ache|sore|throb)", "Moving eyes causes pain"),
        (r"(exacerbated|worse|painful).{0,30}(looking up|side.to.side|look.{0,5}side)", "Pain exacerbated by eye movement"),
        (r"pain.{0,20}(eye.{0,10}movement|moving the eye)", "Pain on eye movement"),
        (r"retrobulbar.{0,15}pain", "Retrobulbar pain (ON)"),
        (r"(discomfort|pain).{0,20}(left eye|right eye|eye).{0,20}(worse|made worse|exacerbated)", "Discomfort worse with eye movements"),
    ],

    # ── ANCHOR: Monocular vision loss ─────────────────────────────────────────
    "monocular_vision_loss": [
        (r"(one|left|right|l-eye|r-eye) eye.{0,30}(blur|vision|dim|loss|blind)", "Monocular vision loss"),
        (r"blurring in one eye", "Blurring in one eye"),
        (r"(left|right).{0,5}eye.{0,20}(dimm|blur|dull|fad|fog|dim|vision|loss)", "One eye dimming / blurring"),
        (r"vision.{0,20}(painful|pain).{0,20}(monocular|one eye)", "Painful monocular vision loss"),
        (r"(monocular).{0,20}(vision loss|blind|dim|blur|visual|impairment)", "Monocular vision loss (clinical)"),
        (r"unilateral.{0,20}(visual|vision).{0,20}(impairment|loss|deficit|blurr|acuity)", "Unilateral visual impairment"),
        (r"(optic neuritis|optic nerve).{0,30}(vision|visual|acuity|pain)", "Optic neuritis (clinical)"),
    ],

    # ── Dyschromatopsia ───────────────────────────────────────────────────────
    "dyschromatopsia": [
        (r"(red|colour|color).{0,20}(fad|wash|dim|dull|less|pale|grey|gray|wrong|off|not bright|desatur|less vivid)", "Dyschromatopsia — colour fading"),
        (r"(grey|gray|greyed|greyed out|less vibrant).{0,20}(eye|vision|right|left)", "Vision greyed out / less vibrant"),
        (r"red.green color loss", "Red-green colour loss"),
        (r"(fading|faded).{0,15}(color|colour)", "Faded colours"),
        (r"colors (look|appear|seem).{0,20}(dull|washed|faded|wrong|off|different)", "Colours appear wrong"),
    ],

    # ── ANCHOR: INO / eye movement disorder ──────────────────────────────────
    # v8.1: word-boundary fix prevents 'ino' matching in 'coordination'
    # v8.1: added lateral gaze palsy / sixth nerve palsy patterns
    "ino_features": [
        (r"(eye|eyes).{0,20}(jiggle|wobble|bounce|shake|jump).{0,20}(look|gaze|side)", "INO — eyes jiggle on gaze"),
        (r"\b(adduction lag|internuclear ophthalmoplegia|ophthalmoplegia)\b", "Internuclear ophthalmoplegia"),
        (r"cannot.{0,20}move.{0,20}(eye|eyes).{0,20}(nose|toward|nasal)", "INO — eye cannot adduct"),
        (r"(eye|eyes).{0,10}(crooked|jiggly|nystagmus|wiggling|oscillating)", "Jiggly / crooked eyes (INO/nystagmus)"),
        (r"(lateral gaze palsy|sixth nerve palsy|6th nerve palsy)", "Sixth nerve / lateral gaze palsy"),
        (r"(eye|eyes).{0,20}(cannot|can.{0,5}not).{0,20}(move outward|abduct|lateral)", "Eye cannot abduct"),
        (r"abduction (deficit|palsy|weakness)", "Abduction deficit (6th nerve)"),
    ],

    # ── ANCHOR: Oscillopsia ───────────────────────────────────────────────────
    "oscillopsia": [
        (r"(room|world|everything|objects|things).{0,30}(wiggl|wobbl|bounc|shak|oscillat|vibrat|jump)", "Oscillopsia — room appears to wiggle"),
        (r"world.{0,20}(split|splits)", "World splits on gaze"),
        (r"object.{0,15}jump.{0,15}(focus|see|gaze)", "Objects jump when focusing"),
        (r"vision.{0,20}(jump|jumpy|jiggly|shimmer)", "Jumpy / shimmery vision"),
        (r"\b(opsoclonus|oscillopsia)\b", "Oscillopsia / opsoclonus (clinical)"),
    ],

    # ── ANCHOR: MS Hug ────────────────────────────────────────────────────────
    "ms_hug": [
        (r"(tight|squeezing|pressure|band|belt|corset|grip).{0,30}(chest|torso|stomach|abdomen|waist|rib|midsection|trunk|middle)", "MS hug — tight band around trunk"),
        (r"(banding|girdle).{0,20}(sensation|feeling|pain).{0,20}(chest|abdomen|trunk|waist)", "Girdle / banding sensation"),
    ],

    # ── ANCHOR: Bilateral vision loss (NMOSD) ────────────────────────────────
    "bilateral_vision_loss": [
        (r"(bilateral|both eyes).{0,30}(vision loss|blind|blur|dim|black|loss of sight)", "Bilateral vision loss"),
        (r"(vision loss|blind|blur|dim|black|loss of sight).{0,30}(bilateral|both eyes)", "Bilateral vision loss"),
        (r"(nlp|no light perception)", "No light perception (NLP)"),
        (r"(bottom half|top half|altitudinal).{0,20}(gone|lost|blind|vision)", "Altitudinal vision loss"),
        (r"both eyes.{0,20}(painful|pain)", "Both eyes painful (ADEM / NMO flag)"),
        (r"(simultaneous).{0,30}vision loss", "Simultaneous bilateral vision loss"),
    ],

    # Post-viral vision (ADEM flag)
    "post_viral_vision": [
        (r"(flu|illness|viral|flu-like).{0,30}(both eyes|bilateral|eye|vision)", "Post-viral vision loss (ADEM flag)"),
        (r"(flu.like illness|viral illness).{0,40}(eye|vision|painful)", "Post-viral eye involvement"),
    ],

    # ── ANCHOR: Diplopia ──────────────────────────────────────────────────────
    "diplopia": [
        (r"double vision", "Double vision (diplopia)"),
        (r"seeing double", "Seeing double"),
        (r"diplopia", "Diplopia (clinical)"),
        (r"(seeing|see).{0,10}two", "Seeing two images"),
        (r"world.{0,20}split.{0,20}(left|right|look)", "World splits on gaze"),
        (r"(eyes.{0,20}crooked|jiggly eyes)", "Crooked / jiggly eyes"),
        (r"(eyes? pointing|eye pointing).{0,20}(wrong direction|wrong way|deviation)", "Eye pointing wrong direction"),
        (r"binocular.{0,15}(diplopia|vision)", "Binocular diplopia"),
    ],

    # ── ANCHOR: Paroxysmal TN / facial pain ──────────────────────────────────
    "paroxysmal_tn_young": [
        (r"electric shocks? in (jaw|face)", "Electric shock in jaw / face (TN)"),
        (r"severe face pain", "Severe face pain (TN)"),
        (r"trigeminal neuralgia", "Trigeminal neuralgia"),
        (r"facial shock", "Facial shock"),
        (r"(facial|face).{0,20}(numb|numbness|pain.{0,15}young|electric|stab|shock)", "Facial numbness / pain"),
        (r"(face|jaw).{0,15}pain.{0,20}(cold wind|trigger|touch|brush)", "Trigger-evoked face pain (TN)"),
        (r"(sharp facial|facial).{0,10}(pain|shock)", "Sharp facial pain"),
        (r"(v1|v2|trigeminal).{0,20}(pain|numb|symptom)", "V1/V2 trigeminal pain"),
        (r"on and off.{0,20}facial pain", "Paroxysmal facial pain"),
        (r"(severe facial|severe face).{0,10}(pain|ache)", "Severe facial pain"),
        (r"(face|facial).{0,15}(weakness|numb|numbness)", "Facial weakness / numbness"),
        (r"(face|jaw|cheek).{0,20}pain.{0,30}(light touch|touch|trigger|cold|wind|brush)", "Light-touch triggered facial pain (TN)"),
        (r"stabbing.{0,20}(face|jaw|cheek|facial|character)", "Stabbing facial pain"),
    ],

    # ── ANCHOR: Area postrema (NMO) ───────────────────────────────────────────
    "area_postrema": [
        (r"intractable hiccups", "Intractable hiccups (area postrema)"),
        (r"uncontrollable vomiting", "Uncontrollable vomiting"),
        (r"hiccups and vomiting", "Hiccups and vomiting (NMO sign)"),
        (r"nausea and vomiting", "Nausea and vomiting"),
        (r"\barea postrema\b", "Area postrema syndrome (NMO)"),
    ],

    # ── ANCHOR: Vertigo / vestibular MS ──────────────────────────────────────
    "vertigo_ms": [
        (r"(vertigo|dizziness|dizzy).{0,30}(fall|balance|walk|gait)", "Vertigo with falls / gait problems"),
        (r"(constant|persistent|abrupt|sudden).{0,15}(vertigo|dizziness)", "Constant or sudden-onset vertigo"),
        (r"(increasingly frequent falls|progressively.{0,15}fall|frequent falls)", "Increasingly frequent falls"),
        (r"(lost all equilibrium|loss of equilibrium|lost equilibrium)", "Loss of equilibrium"),
        (r"(lightheadedness|lightheaded|light.headed).{0,30}(dizziness|balance|standing)", "Lightheadedness with balance difficulty"),
    ],

    # ── ANCHOR: Optic neuritis idioms ─────────────────────────────────────────
    "optic_neuritis_idioms": [
        (r"(dirty water|smudge|foggy|foggy vision|fog).{0,20}(eye|vision|see)", "ON — foggy / smudged vision"),
        (r"(looking through|vision like).{0,20}(dirty|foggy|cloudy|smeared|smudge|mist|haze)", "ON — dirty / foggy vision"),
        (r"(scotoma|central blur|central blurr|central loss|central vision)", "Scotoma / central vision loss"),
        (r"(eye soreness|eye sore|sore eye|soreness behind the eye|eye ache|ache behind the eye)", "Eye soreness (ON)"),
        (r"pulling sensation.{0,20}(behind|eye)", "Pulling sensation behind the eye"),
        (r"(vision|visual).{0,20}(deficit|dimming)", "Visual deficit / dimming"),
        (r"(blurring|blur).{0,20}one eye", "Blurring in one eye (ON)"),
        (r"transient obscur", "Transient obscurations"),
        (r"(washed out|less vibrant|world.{0,10}dim)", "Washed-out / dim vision"),
        (r"(optic neuritis|optic nerve)", "Optic neuritis (clinical)"),
        (r"(visual acuity|visual impairment|visual field).{0,30}(decreas|loss|reduced|impair|defect)", "Visual acuity / field loss"),
        (r"(blurry vision|blurred vision|blurring of vision)", "Blurred vision"),
        (r"\b(afferent pupillary defect|relative afferent)\b", "Afferent pupillary defect (RAPD)"),
    ],

    # ── ANCHOR: IIH / obesity / disc swelling ────────────────────────────────
    "obesity_disc_flag": [
        (r"(disc swelling|papilledema|disc edema|swelling.{0,20}eye|swelling.{0,20}optic)", "Disc swelling (papilledema)"),
        (r"(bilateral disc|disc.{0,10}swelling)", "Bilateral disc swelling"),
        (r"(eyes.{0,20}abnormal|abnormal.{0,20}eye)", "Abnormal eye exam finding"),
        (r"(heavy legs|heavy leg|legs feel heavy|leg heaviness).{0,30}(walk|walking|clinic|stand)", "Heavy legs (spinal cord flag)"),
        (r"(\d{3}\s*kg|obese|high bmi|overweight|obesity)", "Obesity marker (IIH / MS overlap)"),
    ],

    # ── ANCHOR: Prior CIS / DIT ───────────────────────────────────────────────
    "prior_cis_episode": [
        (r"(years?|months?) ago.{0,40}(numb|weak|vision|tingling|balance).{0,60}(better|resolved|went away|recovered)", "Prior resolved neurological episode"),
        (r"(episode|attack|event).{0,20}(numb|weakness|vision|tingling).{0,20}(resolved|recovered|cleared)", "Prior episode resolved"),
        (r"(transient|tingling|numbness|numb).{0,40}(lasted|for one week|for a week|weeks?).{0,30}(ago|years? ago|last)", "Transient episode — DIT marker"),
        (r"history of optic neuritis", "History of optic neuritis (DIT)"),
        (r"(previous|prior|history of).{0,30}(episode|attack|relapse|event|numbness|tingling|weakness|vision)", "Prior neurological episode"),
        (r"(spontaneously resolved|resolved spontaneously|went away on its own|resolved on its own)", "Spontaneous resolution (CIS pattern)"),
        (r"clinically isolated syndrome|first (episode|attack|event|relapse)", "Clinically isolated syndrome"),
        (r"(tingling|numbness|numb).{0,40}(last monsoon|last year|years? ago|months? ago)", "Prior monsoon / seasonal episode (DIT)"),
        (r"monsoon.{0,40}(resolved|better|went away|numb|tingle)", "Monsoon-triggered episode resolved"),
        (r"(episode|attack).{0,50}(numb|weak|tingling|vision).{0,50}(resolved|went away|got better|cleared)", "Prior episode resolved (flexible)"),
    ],

    # ── Standard: Cerebellar ataxia ───────────────────────────────────────────
    "cerebellar_ataxia": [
        (r"(clumsiness|clumsy|dropping keys|drops things|cerebellar)", "Cerebellar clumsiness"),
        (r"(steady|steady myself|off balance|balance problems|poor balance)", "Balance problems (cerebellar)"),
        (r"(seasick|feels seasick|nausea.{0,20}walk|dizzy.{0,20}walk)", "Seasick feeling (cerebellar)"),
        (r"swallowing difficulty", "Swallowing difficulty (brainstem)"),
        (r"\b(dysdiadochokinesia|ataxia|coordination deficit)\b", "Ataxia / coordination deficit"),
        (r"(impaired balance|loss of balance|balance impair|vestibular)", "Impaired balance"),
        (r"(equilibrium|lost.{0,10}equilibrium|lost.{0,10}balance)", "Loss of equilibrium"),
        (r"(heel.to.shin|finger.to.nose|tandem gait)", "Cerebellar examination findings"),
    ],

    # ── Standard: Motor weakness / grip ──────────────────────────────────────
    "motor_weakness_grip": [
        (r"(weak|lose|loss of|give out|clumsy).{0,20}(grip|grasp|hand|power|oomph|button|zip)", "Weak grip / clumsy hands"),
        (r"(drop|dropping).{0,20}(cup|keys|things|mug|bottle)", "Dropping objects"),
        (r"(typing errors|typos|handwriting|script)|(buttoning trouble|loss of oomph)", "Fine motor deficit"),
        (r"(hand|arm|upper extremity|upper limb).{0,20}(weakness|weak|paresis|paresthesia)", "Upper extremity weakness"),
        (r"(decreased|reduced).{0,20}(dexterity|grip|coordination|finger)", "Decreased dexterity"),
        (r"\b(tremor|tremors|intention tremor|action tremor)\b", "Tremor"),
        (r"\b(hemiparesis|paraparesis|monoparesis|quadriparesis)\b", "Paresis / paralysis"),
        (r"(hyperreflexia|babinski.{0,10}(sign|positive)|extensor plantar)", "Upper motor neuron signs"),
        (r"left lower limb weakness|right lower limb weakness|lower limb weakness", "Lower limb weakness"),
    ],

    # ── Standard: Gait ataxia ─────────────────────────────────────────────────
    "gait_ataxia": [
        (r"(drunk|drunken|boat|seasick|stagger|lurch|veer|unsteady|balance|fall|leaner).{0,30}(walk|walking|gait|stand|pivot|staircase|hallway)", "Gait ataxia / unsteady walk"),
        (r"(occasional falls|fall|tripping|trips).{0,20}(walk|balance|gait)", "Falls (gait / balance)"),
        (r"(stumbling gait|stumbl.{0,10}gait|stumbl.{0,10}walk)", "Stumbling gait"),
        (r"(gait disturbance|gait disorder|gait impairment|difficulty.{0,10}walk|difficulty.{0,10}ambul)", "Gait disturbance (clinical)"),
        (r"(tendency to fall|multiple falls|several falls|bad falls|frequent falls)", "Tendency to fall"),
        (r"(ambulation|ambulating|ambulatory).{0,20}(difficult|impaired|problem)", "Ambulation difficulty"),
        (r"(increasingly frequent falls|progressively.{0,15}fall)", "Increasingly frequent falls"),
        (r"(ataxic gait|ataxic walking)", "Ataxic gait"),
    ],

    # ── Standard: Spasticity ─────────────────────────────────────────────────
    "spasticity": [
        (r"(spring.loaded|springy|twitch|jump|stiff|rigid|locked|cramp|spasm|board|glued).{0,20}(leg|feet|foot|muscle|calf|knee|finger|hand)", "Spasticity / stiffness"),
        (r"(locked muscles|stiff fingers|stiff knee)", "Locked / stiff muscles"),
        (r"\b(spasticity|hyperreflexia|hyperreflexic|extensor plantar|babinski|clonus)\b", "Spasticity (clinical)"),
        (r"(increased tone|increased reflexes|brisk reflexes)", "Increased tone / reflexes"),
        (r"spastic (quadriparesis|paraparesis|hemiparesis)", "Spastic paresis"),
        (r"(muscle spasm|muscle cramp).{0,20}(ms|neurological|multiple sclerosis)", "MS-related muscle spasm"),
    ],

    # ── Standard: Tremor (v8.1 — separated from motor_weakness_grip) ────────
    "tremor_standard": [
        (r"(uncontrollable shaking|uncontrollable trembling|tremor).{0,20}(arm|leg|head|hand|body)", "Uncontrollable tremor"),
        (r"\b(intention tremor|action tremor|cerebellar tremor)\b", "Intention / cerebellar tremor"),
    ],

    # ── Standard: Foot drop ───────────────────────────────────────────────────
    "foot_drop": [
        (r"(foot|toe|toes|feet).{0,20}(drag|dragging|drop|dropping|scuff|scrape)", "Foot drop"),
        (r"toe dragging", "Toe dragging"),
        (r"(lift|lifting).{0,15}(knee|foot).{0,20}(clear|foot|toe|floor|step)", "Lifting knee to clear foot (foot drop)"),
        (r"foot drop|drop foot", "Foot drop (clinical)"),
    ],

    # ── ANCHOR: Limb numbness / paresthesia ───────────────────────────────────
    "limb_numbness": [
        (r"(arm|leg|hand|foot|feet|finger|fingertip|thigh|shin|back|scalp|neck|cheek|face|ankle).{0,30}(numb|numbness|tingling|pins and needles)", "Limb / body numbness or tingling"),
        (r"(numbness|numb|tingling).{0,30}(fingertip|finger|hand|arm|leg|ankle)", "Fingertip / limb numbness"),
        (r"(tingling in|numbness in|numb in).{0,20}(extremities|arm|leg|hand|foot|body|side|face|ankle)", "Tingling / numbness in extremities"),
        (r"\b(hemibody|hemi.body|hemisensory)\b.{0,30}(paresthesia|numb|weak|sensory)", "Hemibody paresthesia (MS relapse)"),
        (r"\b(paresthesia|paresthesias|dysesthesia)\b.{0,20}(sensory|numb|tingle|sensation|deficit)", "Clinical paresthesia"),
        (r"(altered sensation|sensory deficit|sensory loss|hemisensory)", "Sensory deficit (clinical)"),
        (r"(paresthesia|tingling|numbness).{0,30}(ascend|ascending|spread|spreading).{0,30}(leg|limb|trunk|umbilicus|body)", "Ascending paresthesia (myelitis)"),
        (r"(right.sided|left.sided).{0,20}(weakness|numbness|numb|paresthesia)", "Hemisensory / hemibody symptoms"),
        (r"glove.{0,10}stocking.{0,10}(pattern|numbness|paresthesia)", "Glove-and-stocking pattern"),
        (r"(radiation|radiating).{0,20}(numb|numbness|tingling|pain).{0,20}(ankle|leg|arm|foot|thigh|shin)", "Radiating numbness / tingling"),
    ],

    # ── ANCHOR: Sensory idioms / dysesthesia ─────────────────────────────────
    "sensory_idioms": [
        (r"(cold|water|ice|frozen).{0,20}(trickle|trickling|running|poured|pack)", "Cold water / ice sensation"),
        (r"(twice its size|swollen|puffy).{0,30}(looks normal|looks fine|perfectly normal)", "Phantom swelling (dysesthesia)"),
        (r"(hand|arm|leg|foot).{0,20}feels.{0,20}(twice|swollen|puffy|bigger).{0,30}(normal|fine)", "Phantom swelling"),
        (r"(dead|wooden|block|dead feeling|completely dead).{0,20}(arm|leg|patch|skin|left arm|right arm)", "Dead / wooden sensation"),
        (r"(completely dead|dead feeling|deadness).{0,20}(arm|leg|hand|foot)", "Dead limb sensation"),
        (r"(left arm|right arm|arm|leg).{0,30}(completely dead|dead|wooden|numb)", "Arm / leg feels dead"),
        (r"(arm|leg).{0,20}(feels|feel|is|are).{0,10}(completely dead|dead|wooden)", "Limb feels dead"),
        (r"(hot coals|stinging needles|bee sting|stinging).{0,20}(feet|foot|sole|face|skin|soles)", "Hot coals / stinging sensation"),
        (r"(walking on hot coals|feel like hot coals|feels like hot coals)", "Walking on hot coals"),
        (r"soles.{0,30}(hot coals|needles|stinging|burning|fire)", "Burning soles"),
        (r"(sandpaper).{0,20}(skin|rubbing|touch|wound|on a wound)", "Allodynia — sandpaper sensation"),
        (r"(light touch|cotton|shirt).{0,30}(sandpaper|stinging|hurts|pain|painful|wound)", "Allodynia — light touch painful"),
        (r"(cell phone|vibration).{0,20}(hip|bone|deep)", "Internal vibration (pallanaesthesia)"),
        (r"(gloves|socks).{0,30}(muffled|numb|thick)", "Glove-and-stocking numbness"),
        (r"(sloshy|wet).{0,20}(shoes|feet)", "Sloshy / wet sensation"),
        (r"(ants|ant|insect|bug).{0,20}(crawl|crawling|creeping).{0,20}(scalp|neck|skin|face|arm|leg)", "Formication (ants crawling)"),
        (r"(palm.sized|patch).{0,40}(wooden|numb)", "Specific numb patch"),
        (r"felt.{0,10}(wooden|completely numb)", "Felt wooden / numb"),
    ],

    # ── Standard: Proprioception loss ─────────────────────────────────────────
    "proprioception_loss": [
        (r"(lose|loses|lost|where).{0,20}(feet|track of|dark|look down)", "Proprioception loss"),
        (r"(walk.{0,15}dark|dark.{0,15}walk).{0,30}(look down|feet|where)", "Proprioception loss in dark"),
        (r"vibratory sense.{0,20}(loss|impaired|decreased|reduced)", "Vibratory sense loss"),
        (r"(position sense|proprioception).{0,20}(impaired|loss|reduced)", "Proprioception impaired (clinical)"),
    ],

    # ── Standard: Autonomic / bladder / bowel ────────────────────────────────
    "autonomic_ms": [
        (r"(urinary urgency|bladder urgency|urge to urinate|urgency urination|frequency urination)", "Urinary urgency"),
        (r"(bowel|bladder).{0,20}(problem|control|urgency|incontinence|dysfunction)", "Bladder / bowel dysfunction"),
        (r"(urinary|urinating).{0,20}(problem|urgency|frequent|difficult)", "Urinary dysfunction"),
        (r"(incontinence|incontinent|nocturia|multiple voids|voiding)", "Incontinence / nocturia"),
        (r"(incomplete bladder|bladder emptying|urinary bladder)", "Incomplete bladder emptying"),
        (r"\b(neurogenic bladder|neurogenic bowel)\b", "Neurogenic bladder / bowel"),
        (r"(constipation|bowel.{0,20}(control|incontinence|problem)).{0,20}(ms|neurological|multiple sclerosis)", "MS-related bowel dysfunction"),
        (r"(fecal incontinence|bowel incontinence|bowel leakage)", "Fecal / bowel incontinence"),
    ],

    # ── Standard: Visual mimic flags ─────────────────────────────────────────
    "visual_mimic_flags": [
        (r"(floater|floaters|dark spot|black spot|spot.{0,15}moves|moving spot)", "Floaters / dark spot (retinal flag)"),
        (r"(red.{0,10}eye|red.{0,10}painful eye|painful.{0,10}red eye)", "Red painful eye (mimic flag)"),
        (r"(flickering lights|flicker.{0,15}light|scintillating).{0,30}(20 min|min|lasts|aura)", "Flickering lights — migraine aura flag"),
        (r"(chronic|progressive).{0,20}(vision loss|visual loss).{0,30}(1 year|year|months?)", "Chronic progressive vision loss — tumour flag"),
    ],

    # ── Standard: Cognitive / brain fog ──────────────────────────────────────
    "cognitive_ms": [
        (r"(brain|mind).{0,15}(fog|cloudy|slow|blank|mush)", "Brain fog"),
        (r"(memory|remember).{0,20}(loss|lapse|bad|poor|failing|forget)", "Memory lapses"),
        (r"(cognitive|concentration|attention).{0,20}(impair|problem|difficult|decline|deficit)", "Cognitive impairment (clinical)"),
        (r"impaired.{0,10}(memory|short.term memory|recall)", "Impaired memory"),
        (r"(word.{0,10}finding|anomia|forget.{0,10}words|lose.{0,10}words)", "Word-finding difficulty"),
        (r"(slow.{0,10}processing|processing speed|slow.{0,10}thinking|thinking.{0,10}slow)", "Slowed cognitive processing"),
    ],

    # ── Standard: Fatigue ─────────────────────────────────────────────────────
    "fatigue_ms": [
        (r"(severe|extreme|debilitating|significant|profound|marked).{0,20}(fatigue|exhaustion|tiredness)", "Severe MS fatigue"),
        (r"(fatigue|exhaustion).{0,20}(ms|multiple sclerosis|relapse|symptom)", "MS-associated fatigue"),
        (r"(chronic fatigue|persistent fatigue|overwhelming fatigue|constant fatigue)", "Chronic / persistent fatigue"),
    ],

    # ── Standard: Dysarthria / dysphagia ─────────────────────────────────────
    "dysarthria_dysphagia": [
        (r"(slurred speech|dysarthria|speech.{0,15}(slur|difficult|impaired))", "Dysarthria (slurred speech)"),
        (r"(swallowing.{0,20}(difficult|problem|impaired)|dysphagia|aspirat)", "Dysphagia / aspiration"),
        (r"(tongue.{0,15}(slow|weak|impaired)|difficulty speaking)", "Tongue / speech difficulty"),
    ],

    # ── Standard: Dysphagia isolated (v8.1) ──────────────────────────────────
    "dysphagia_ms": [
        (r"(swallowing.{0,20}(difficult|problem|impaired|trouble)|dysphagia)", "Swallowing difficulty"),
        (r"(food.{0,20}stuck.{0,20}throat|choking.{0,20}(eat|drink|swallow))", "Food stuck / choking"),
        (r"(coughing.{0,20}(eat|drink|swallow)|aspiration)", "Aspiration / coughing when eating"),
    ],

    # ── Standard: Pseudobulbar affect ────────────────────────────────────────
    "pseudobulbar_affect": [
        (r"(laugh|laughing|cry|crying|weep).{0,30}(uncontrollable|random|no reason|inappropriate)", "Pseudobulbar affect (PBA)"),
        (r"\b(pseudobulbar|emotional lability)\b.{0,20}", "Pseudobulbar affect / emotional lability"),
    ],

    # ── Standard: Nocturnal spasms ────────────────────────────────────────────
    # Note: RLS intentionally omitted here — stays in sleep_disturbance_ms (LP)
    "nocturnal_spasms": [
        (r"(spasm|cramp|stiff|jerk|kick).{0,30}(night|sleep|bed|nocturnal|waking)", "Nocturnal spasms / kicks"),
        (r"(night kick|night kicks)", "Night kicks"),
    ],

    # ── Standard: Thermal dysesthesia ────────────────────────────────────────
    "thermal_dysesthesia": [
        (r"(feet|legs|foot|soles).{0,20}(burn|fire|hot|hot coals|stinging|needles)", "Burning / hot feet / legs"),
    ],

    # ═══════════════════════════════════════════════════════════════════════════
    # NEW IN v8.1 — Paroxysmal / Rare / Internet-sourced categories
    # ═══════════════════════════════════════════════════════════════════════════

    # ── ANCHOR: Paroxysmal MS symptoms ────────────────────────────────────────
    # Source: PMC 2024, Frontiers in Neurology, StatPearls
    "paroxysmal_ms": [
        (r"(paroxysmal).{0,30}(dysarthria|ataxia|diplopia|hemiparesis|sensory|symptom)", "Paroxysmal MS symptom"),
        (r"(episodic|stereotyped|brief.{0,10}recurring).{0,20}(motor|sensory|speech|movement|neurological)", "Stereotyped recurring neurological episodes"),
        (r"(dystonic|dystonia).{0,20}(episode|attack|spasm).{0,20}(ms|sclerosis|neurological)", "Dystonic episode (paroxysmal MS)"),
        (r"(paroxysmal dysarthria|paroxysmal ataxia|paroxysmal hemiparesis|paroxysmal sensory)", "Paroxysmal dysarthria / ataxia / hemiparesis"),
        (r"(paroxysmal.{0,20}(symptom|episode|attack)).{0,20}(ms|sclerosis|demyelinating)", "Paroxysmal MS episode"),
        (r"(paroxysmal dystonia|paroxysmal posturing).{0,20}(ms|first|initial|manifestation)", "Paroxysmal dystonia — MS first sign"),
    ],

    # ── ANCHOR: Tonic spasms ──────────────────────────────────────────────────
    # Source: Frontiers 2024, PubMed, Cureus
    "tonic_spasms_ms": [
        (r"(tonic spasm|painful tonic|tonic seizure).{0,20}(ms|multiple sclerosis|episodic|brief|posture|limb)", "Tonic spasms in MS"),
        (r"(involuntary.{0,20}(flexion|extension|posturing|tonic).{0,20}(ms|episodic|brief))", "Involuntary tonic posturing"),
        (r"(painful.{0,15}stereotyped.{0,15}(episode|spasm|attack)|stereotyped.{0,15}painful.{0,15}episode)", "Painful stereotyped episodes"),
    ],

    # ── ANCHOR: Tumefactive MS ────────────────────────────────────────────────
    # Source: Frontiers 2024, ScienceDirect
    "tumefactive_ms": [
        (r"(tumefactive|large lesion|mass effect).{0,20}(ms|multiple sclerosis|demyelinating)", "Tumefactive MS"),
        (r"(rapid.{0,15}deterioration|progressive.{0,15}deterioration).{0,20}(ms|lesion|neurological)", "Rapid neurological deterioration"),
        (r"ring.{0,10}enhancement.{0,20}(ms|lesion|gadolinium|mri)", "Ring-enhancing lesion (tumefactive)"),
        (r"tumefactive multiple sclerosis", "Tumefactive multiple sclerosis"),
    ],

    # ── Standard: Paroxysmal pruritis ─────────────────────────────────────────
    # Source: PMC 2020 review, Cureus case report
    "paroxysmal_pruritis": [
        (r"(paroxysmal|intense|episodic|sudden).{0,15}(itch|itching|pruritis|pruritus)", "Paroxysmal itching"),
        (r"(intense itching|severe itching).{0,20}(dermatome|ms|neurological|skin|no.{0,10}rash)", "Intense dermatome itching"),
        (r"itch.{0,30}(ms|multiple sclerosis|dermatome|neurological|spinal cord)", "Neurological pruritis in MS"),
    ],

    # ── Standard: Transverse myelitis ────────────────────────────────────────
    # Source: NINDS, StatPearls
    "transverse_myelitis": [
        (r"(transverse myelitis|myelitis|spinal cord.{0,20}(inflam|lesion|demyelinating))", "Transverse / spinal cord myelitis"),
        (r"(acute myelitis|subacute myelitis|partial myelitis)", "Acute / partial myelitis"),
    ],

    # ── Standard: Cerebellar variant ─────────────────────────────────────────
    # Source: StatPearls, WebMD
    "cerebellar_ms": [
        (r"\b(cerebellar.{0,20}(ataxia|tremor|variant|ms|lesion))\b", "Cerebellar MS variant"),
        (r"(charcot triad|intention tremor.{0,20}nystagmus|nystagmus.{0,20}intention tremor)", "Charcot triad (cerebellar MS)"),
    ],

    # ── Standard: Gustatory / taste disturbance ───────────────────────────────
    # Source: StatPearls, PMC
    "gustatory_ms": [
        (r"(taste.{0,20}(hallucination|altered|distorted|loss|abnormal|changed))", "Taste hallucination / disturbance"),
        (r"(dysgeusia|ageusia|gustatory.{0,15}(symptom|disturbance|hallucination))", "Dysgeusia / ageusia (clinical)"),
        (r"(food.{0,15}(taste|tastes).{0,15}(different|wrong|strange|metallic|bland|off)).{0,20}(neurological|ms)", "Food tastes wrong / metallic"),
        (r"(altered taste|taste disturbance).{0,20}(neurological|ms|sclerosis)", "Altered taste (MS context)"),
    ],

    # ── Standard: Euphoria / inappropriate affect ─────────────────────────────
    # Source: Merck Manual 2024
    "euphoria_ms": [
        (r"(euphoria|euphoric|inappropriate happiness|manic episode).{0,20}(ms|neurological|cerebral|brain)", "Euphoria in MS"),
        (r"(apathy|poor judgment|inattention).{0,20}(ms|neurological|cerebral|cognitive)", "Apathy / poor judgment (MS)"),
    ],

    # ── Standard: Useless hand (Oppenheim) ───────────────────────────────────
    # Source: StatPearls 2024
    "useless_hand_ms": [
        (r"(useless hand|oppenheim|loss of.{0,10}function.{0,10}hand|hand.{0,10}useless).{0,20}(ms|neurological|demyelinating)", "Useless hand (Oppenheim sign)"),
        (r"(difficulty.{0,20}(crossing road|pour|pouring|playing ball).{0,20}(ms|neurological|dexterity))", "Difficulty pouring / crossing road"),
        (r"(sense of.{0,10}uselessness.{0,10}hand|hand.{0,10}function.{0,10}lost)", "Sense of hand uselessness"),
    ],

    # ── Standard: Seizures in MS ─────────────────────────────────────────────
    # Source: StatPearls, Frontiers, PubMed
    "seizures_ms": [
        (r"(seizure|epileptic).{0,30}(ms|multiple sclerosis|demyelinating)", "Seizures in MS"),
        (r"(focal seizure|tonic.clonic|absence seizure).{0,20}(ms|neurological|brain lesion)", "Focal / generalised seizure in MS"),
        (r"(epilepsy|epileptic).{0,20}(ms|multiple sclerosis)", "Epilepsy / MS comorbidity"),
    ],

    # ── Standard: Prodromal / RIS (2024 McDonald) ────────────────────────────
    # Source: PMC 2024, McDonald criteria 2024
    "prodromal_ms": [
        (r"(radiologically isolated syndrome|\bris\b).{0,20}(ms|white matter|lesion)", "Radiologically isolated syndrome (RIS)"),
        (r"(subclinical|asymptomatic).{0,20}(ms|lesion|demyelinating|white matter)", "Subclinical / asymptomatic MS lesions"),
        (r"(prodromal.{0,20}ms|ms.{0,20}prodromal)", "Prodromal MS"),
    ],

    # ── Standard: Psychiatric first presentation ──────────────────────────────
    # Source: Merck, Cureus
    "psychiatric_ms": [
        (r"(psychiatric.{0,20}(symptom|presentation|ms|first))", "Psychiatric first presentation of MS"),
        (r"(mood disorder.{0,20}(ms|neurological|brain|lesion))", "Mood disorder with MS lesion"),
        (r"(psychosis|mania|bipolar).{0,20}(ms|multiple sclerosis|demyelinating)", "Psychosis / mania in MS"),
    ],

    # ── Standard: Functional overlay on MS ───────────────────────────────────
    # Source: NHS, Frontiers
    "functional_ms": [
        (r"(functional neurological|functional overlay).{0,20}(ms|multiple sclerosis|confirmed)", "Functional overlay on MS"),
        (r"(functional.{0,15}disorder|non.organic).{0,20}(ms|confirmed|comorbid)", "Functional neurological disorder comorbid MS"),
        (r"functional overlay on.{0,15}(ms|multiple sclerosis|confirmed ms)", "Functional overlay on confirmed MS"),
    ],
}


LOW_PRIORITY_PATTERNS: Dict[str, List[tuple]] = {

    # LP-1: Heat sensitivity (non-visual)
    # Distinct from Uhthoff's (which requires visual/gait component).
    # Generalised heat intolerance, summer worsening — common MS soft sign.
    "heat_sensitivity_general": [
        (r"(heat|warm|hot).{0,30}(worsen|worse|trigger|tired|exhaust|weak|fatigue)", "Heat worsening general symptoms"),
        (r"(feel worse|feel bad|symptoms worse).{0,20}(heat|warm|hot|summer|sauna)", "Symptoms worse in heat"),
        (r"(cannot tolerate|intolerant to|sensitive to).{0,20}(heat|warm|hot|sun)", "Heat intolerance (general)"),
        (r"(summer|hot weather|warm weather).{0,20}(worse|bad|difficult|relapse|trigger)", "Summer symptom worsening"),
    ],

    # LP-2: Chronic pain (non-TN)
    "chronic_pain_ms": [
        (r"(chronic|persistent|constant|ongoing).{0,15}(pain|aching|burning|discomfort)", "Chronic pain"),
        (r"(back pain|leg pain|arm pain|body pain|joint pain).{0,30}(ms|neurological|nerve|numb)", "MS-related pain"),
        (r"(burning|aching).{0,20}(legs|arms|back|body|limbs).{0,30}(constant|chronic|always|persistent)", "Burning / aching extremities"),
        (r"(central sensitiz|central sensitisation|neuropathic pain|allodynia).{0,20}", "Central sensitisation / neuropathic pain"),
        (r"(pain.{0,15}unresponsive|pain.{0,15}not respond).{0,20}(paracetamol|ibuprofen|nsaid|analgesic)", "Pain unresponsive to analgesia"),
    ],

    # LP-3: Sleep disturbance
    "sleep_disturbance_ms": [
        (r"(insomnia|cannot sleep|trouble sleeping|poor sleep|sleep disturbance)", "Insomnia / sleep disturbance"),
        (r"(non.restorative sleep|unrefreshing sleep|wake.{0,10}unrefreshed|tired after sleep)", "Non-restorative sleep"),
        (r"(periodic limb|limb movement).{0,20}(sleep|night|bed)", "Periodic limb movement in sleep"),
        (r"(sleep.{0,15}(problem|issue|disorder)).{0,20}(ms|fatigue|spasm|pain)", "Sleep problem (MS context)"),
    ],

    # LP-4: Sexual dysfunction
    "sexual_dysfunction_ms": [
        (r"(sexual dysfunction|sexual problem|sexual difficulty|sex.{0,10}problem)", "Sexual dysfunction"),
        (r"(reduced libido|low libido|decreased libido|loss of libido|no desire)", "Reduced libido"),
        (r"(erectile dysfunction|impotence|erection problem)", "Erectile dysfunction"),
        (r"(genital.{0,15}(numb|numbness|tingle|sensation|sensory))", "Genital numbness / tingling"),
        (r"(arousal.{0,15}(problem|difficult|reduced|impaired)|lubrication.{0,15}(problem|reduced|impaired))", "Arousal / lubrication difficulty"),
        (r"(orgasm.{0,15}(difficult|problem|impaired|absent|reduced))", "Orgasmic dysfunction"),
        (r"(vaginal dryness|decreased lubrication)", "Vaginal dryness / decreased lubrication"),
    ],

    # LP-5: Mood and affect (sustained — distinct from pseudobulbar)
    "mood_affect_ms": [
        (r"(depression|depressed|low mood|feeling down|hopeless).{0,30}(ms|neurological|since|for months|for years)", "MS-related depression"),
        (r"(anxiety|anxious|panic attack|worry).{0,30}(ms|neurological|symptoms|worsening)", "MS-related anxiety"),
        (r"(mood swings|emotional lability|irritability|mood changes).{0,30}(ms|neurological|recent|new)", "Mood swings / lability"),
        (r"(euphoria|euphoric).{0,20}(ms|inappropriate|unusual)", "Euphoria (MS sign)"),
        (r"(emotional.{0,10}(flatness|blunting|numbing)|apathy|motivation.{0,10}(loss|reduced|low))", "Emotional flattening / apathy"),
    ],

    # LP-6: Tinnitus / hearing changes
    "tinnitus_hearing_ms": [
        (r"(tinnitus|ringing in.{0,10}ear|buzzing in.{0,10}ear|ringing ear)", "Tinnitus (ringing in ears)"),
        (r"(chronic.{0,10}buzzing|persistent.{0,10}buzzing|constant.{0,10}buzzing).{0,20}(ear|head)", "Chronic buzzing in ears"),
        (r"(hearing loss|hearing impairment|hard of hearing|decreased hearing)", "Hearing loss / impairment"),
        (r"(sudden hearing loss|sudden deafness)", "Sudden hearing loss"),
        (r"(auditory.{0,15}(hallucination|distortion|problem)|sound.{0,10}(distorted|strange|weird))", "Auditory disturbance"),
    ],

    # LP-7: Cognitive fatigue (specific — distinct from generalised cognitive_ms)
    "cognitive_fatigue_ms": [
        (r"(mental fatigue|cognitive fatigue|brain.{0,10}exhaust|mental exhaustion)", "Cognitive / mental fatigue"),
        (r"(brain.{0,10}(shut|shut down|give up|stop working)|mind.{0,10}blank.{0,10}(exhaust|tired|fatigue))", "Brain shutting down feeling"),
        (r"(cannot.{0,10}concentrate|hard to concentrate|concentration.{0,10}(problem|difficult|poor))", "Concentration difficulty"),
        (r"(mental.{0,10}(fog|foggy|cloudy|slow)).{0,20}(after|following|when).{0,20}(effort|activity|task)", "Mental fog after effort"),
    ],

    # LP-8: Bowel dysfunction (isolated — standard if combined with other autonomic)
    "bowel_dysfunction_ms": [
        (r"(constipation|bowel.{0,20}(problem|control|issue|incontinence))", "Constipation / bowel control problems"),
        (r"(fecal incontinence|bowel incontinence|bowel leakage)", "Bowel incontinence"),
        (r"(neurogenic bowel|bowel dysfunction)", "Neurogenic bowel dysfunction"),
    ],

    # LP-9: Respiratory (MS-related)
    "respiratory_ms": [
        (r"(shortness of breath|breathing.{0,20}(difficulty|problem|trouble)).{0,20}(ms|neurological|weakness|fatigue)", "Shortness of breath (MS respiratory weakness)"),
        (r"(respiratory.{0,20}(weakness|dysfunction|problem)).{0,20}(ms|neurological)", "Respiratory weakness in MS"),
        (r"diaphragm.{0,20}(weakness|paralysis|dysfunction)", "Diaphragm weakness / dysfunction"),
    ],

    # LP-10: Headache (MS context)
    "headache_ms": [
        (r"(headache|head pain).{0,20}(ms|multiple sclerosis|relapse|lesion|worsening)", "Headache in MS context"),
        (r"(ms.{0,20}migraine|migraine.{0,20}ms)", "Migraine — MS comorbidity"),
        (r"(persistent headache|chronic headache|new.{0,10}headache).{0,20}(neurological|ms|spine)", "Persistent headache (MS context)"),
        (r"headache.{0,60}(started|began).{0,30}(neurological|ms|symptom)", "Headache onset with neurological symptoms"),
    ],

    # LP-11: Skin sensation / hypersensitivity
    "skin_sensation_ms": [
        (r"(skin.{0,20}sensitive|sensitive skin|hypersensitive).{0,20}(ms|neurological|touch)", "Skin hypersensitivity"),
        (r"\b(allodynia|hyperaesthesia|cutaneous.{0,15}(pain|sensitivity))\b", "Allodynia / hyperaesthesia"),
        (r"(prickling|pins and needles).{0,20}(skin|face|arm|leg|body)", "Prickling / pins and needles on skin"),
    ],

    # LP-12: Pain in MS (central / neuropathic — non-TN)
    "pain_ms": [
        (r"(neuropathic pain|central pain).{0,20}(ms|multiple sclerosis|neurological)", "Neuropathic / central pain in MS"),
        (r"(trigeminal neuralgia|painful.{0,15}spasm)", "Painful spasm / TN-like pain"),
        (r"(burning pain|stabbing pain|shooting pain).{0,20}(leg|arm|back|face|body|extremities)", "Burning / stabbing / shooting pain"),
    ],

    # LP-13: Menstrual / perimenstrual worsening (v8.1 new)
    # Source: MSAA, PMC 2024
    "menstrual_ms": [
        (r"(perimenstrual|menstrual|period).{0,20}(worse|worsen|relapse|trigger|symptom|ms)", "Perimenstrual symptom worsening"),
        (r"(symptoms worse.{0,20}(period|menstruation|cycle)|period.{0,30}ms symptom)", "Symptoms worse around period"),
    ],

    # LP-14: Stress-triggered worsening (v8.1 new)
    # Source: NHS, WebMD
    "stress_triggered_ms": [
        (r"(psychological stress|emotional stress|stress).{0,30}(trigger|worsen|worse|uhthoff|ms|symptom)", "Stress triggering MS symptoms"),
        (r"(stress.{0,20}trigger.{0,20}ms|ms.{0,20}stress.{0,20}trigger)", "Stress — MS pseudo-relapse trigger"),
    ],

    # LP-15: Pregnancy / postpartum (v8.1 new)
    # Source: NINDS, MS Society
    "pregnancy_ms": [
        (r"(pregnancy|pregnant|postpartum).{0,30}(ms|relapse|symptom|worse|improve)", "Pregnancy / postpartum MS risk"),
        (r"(ms.{0,20}(pregnancy|pregnant|postpartum)|(pregnancy|pregnant|postpartum).{0,20}ms)", "MS and pregnancy / postpartum"),
    ],
}


SEVERITY_MODIFIERS = {
    "severe": {
        "patterns": [r"\b(severe|complete|total|worst|extreme|unbearable|debilitating|paralyzed|blind)\b"],
        "weight": 1.5,
        "label": "Severe",
    },
    "mild": {
        "patterns": [r"\b(mild|slight|faint|occasional|sometimes|a little|minor|brief)\b"],
        "weight": 0.5,
        "label": "Mild / Occasional",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# DIS REGION MAPPING  (new — bridges symptom keys to McDonald 2017 DIS regions
# so a regex hit can also populate AgentState.dis_regions, not just tier)
# Conservative: only unambiguous keys are mapped. Anything not listed here
# still counts toward tier/features, just not toward a specific DIS region.
# ═══════════════════════════════════════════════════════════════════════════════
DIS_REGION_MAP: Dict[str, str] = {
    # optic_nerve
    "pain_on_eye_movement":      "optic_nerve",
    "monocular_vision_loss":     "optic_nerve",
    "dyschromatopsia":           "optic_nerve",
    "optic_neuritis_idioms":     "optic_nerve",
    "bilateral_vision_loss":     "optic_nerve",
    # brainstem
    "ino_features":              "brainstem",
    "oscillopsia":                "brainstem",
    "paroxysmal_tn_young":       "brainstem",
    "area_postrema":             "brainstem",
    # spinal_cord
    "lhermittes_sign":           "spinal_cord",
    "lhermittes_variant":        "spinal_cord",
    "sensory_idioms":            "spinal_cord",
    "limb_numbness":             "spinal_cord",
    "ms_hug":                    "spinal_cord",
    "bowel_dysfunction_ms":      "spinal_cord",
    # cerebellar
    "vertigo_ms":                "cerebellar",
}


# ═══════════════════════════════════════════════════════════════════════════════
# NEGATION-AWARE NLP PIPELINE (lazy singleton — built once, in-memory)
# Same termset as NeuroCheck's lifespan() setup, minus the on-disk model:
# a blank English + sentencizer is enough for NegEx and needs no download.
# ═══════════════════════════════════════════════════════════════════════════════
_nlp = None
_doc_cache: dict = {}  # single-call cache: parse text ONCE per scan_symptom_db() call


def _get_nlp():
    global _nlp
    if _nlp is None:
        nlp = English()
        nlp.add_pipe("sentencizer")
        ts = termset("en_clinical")
        ts.add_patterns({
            "preceding_negations": [
                "negative for", "cleared of", "denies any", "no history of",
                "do not have", "does not have", "no evidence of", "without",
                "denies", "rules out", "not had", "not having", "never had",
                "excluded", "has been excluded", "has not had",
            ],
            "following_negations": [
                "ruled out by", "ruled out", "is absent",
                "is negative", "has been excluded",
            ],
            "pseudo_negations": ["might not be", "not entirely sure", "not certain"],
        })
        nlp.add_pipe("negex", config={"ent_types": ["SYMPTOM"], "neg_termset": ts.get_patterns()})
        _nlp = nlp
    return _nlp


def _get_doc(text_lower: str):
    if text_lower not in _doc_cache:
        _doc_cache.clear()  # keep cache size = 1
        _doc_cache[text_lower] = _get_nlp()(text_lower)
    return _doc_cache[text_lower]


def _negex_check(text_lower: str, start_char: int, end_char: int) -> bool:
    """True if the matched span is negated via NegEx."""
    doc = _get_doc(text_lower)
    span = doc.char_span(start_char, end_char, label="SYMPTOM", alignment_mode="expand")
    return bool(span and span._.negex)


def _severity_multiplier(text_lower: str) -> float:
    for mod in SEVERITY_MODIFIERS.values():
        for p in mod["patterns"]:
            if re.search(p, text_lower):
                return mod["weight"]
    return 1.0


class SymptomDBResult(TypedDict):
    tier: str                 # LOW | WATCH | MODERATE | HIGH | CRITICAL_EMERGENCY
    features: List[str]       # human-readable labels, e.g. "Lhermitte's sign — electric shock down spine"
    matched_keys: List[str]   # internal keys, e.g. "lhermittes_sign" — for DIS_REGION_MAP lookup
    dis_regions: List[str]    # derived via DIS_REGION_MAP
    has_anchor: bool


def scan_symptom_db(text: str) -> SymptomDBResult:
    """
    Deterministic, negation-aware MS symptom scan — mirrors NeuroCheck's
    classify() scoring exactly:

      HIGH     = t12_score >= 6, OR (t12_score >= 3 AND an anchor feature hit)
      MODERATE = t12_score > 0
      WATCH    = LP-only (no Tier 1/2 hits, LP score >= 1)
      LOW      = nothing detected

    Call this on the FULL cumulative conversation text (all human turns so
    far), not just the latest message — symptoms mentioned two turns ago
    must still count. Intended caller: node_llm.py, merged into the existing
    tier/features union (never overwrites the LLM's own output, only floors it).
    """
    if not text or not text.strip():
        return SymptomDBResult(tier="LOW", features=[], matched_keys=[], dis_regions=[], has_anchor=False)

    t = text.lower()

    # Emergency check — no NegEx, fires immediately (mirrors NeuroCheck; kept
    # for reference/comparison, NOT wired into the live emergency short-circuit
    # in this pass — see module docstring).
    for key, data in EMERGENCY_FLAGS.items():
        if any(re.search(p, t) for p in data["patterns"]):
            return SymptomDBResult(
                tier="CRITICAL_EMERGENCY", features=[data["label"]],
                matched_keys=[key], dis_regions=[], has_anchor=False,
            )

    mult = _severity_multiplier(t)

    features: List[str] = []
    matched_keys: List[str] = []
    dis_regions: List[str] = []
    has_anchor = False
    t12_score = 0.0

    for feat_key, patterns in SYMPTOM_PATTERNS.items():
        for pattern, label in patterns:
            for m in re.finditer(pattern, t):
                if not _negex_check(t, m.start(), m.end()):
                    t12_score += 3.0 * mult
                    if feat_key in ANCHOR_FEATURES:
                        has_anchor = True
                    features.append(label)
                    matched_keys.append(feat_key)
                    if feat_key in DIS_REGION_MAP:
                        dis_regions.append(DIS_REGION_MAP[feat_key])
                    break
            else:
                continue
            break

    lp_score = 0.0
    for feat_key, patterns in LOW_PRIORITY_PATTERNS.items():
        for pattern, label in patterns:
            for m in re.finditer(pattern, t):
                if not _negex_check(t, m.start(), m.end()):
                    lp_score += 1.0
                    features.append(label)
                    matched_keys.append(feat_key)
                    break
            else:
                continue
            break

    if t12_score >= 6 or (t12_score >= 3 and has_anchor):
        tier = "HIGH"
    elif t12_score > 0:
        tier = "MODERATE"
    elif lp_score >= 1:
        tier = "WATCH"
    else:
        tier = "LOW"

    return SymptomDBResult(
        tier=tier,
        features=list(dict.fromkeys(features)),
        matched_keys=list(dict.fromkeys(matched_keys)),
        dis_regions=list(dict.fromkeys(dis_regions)),
        has_anchor=has_anchor,
    )
