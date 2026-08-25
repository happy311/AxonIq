"""
AxonIQ — Greeting Detection & Fast-Path Responses (v11.2)

Handles non-clinical messages without invoking the LLM.
Pure functions — no I/O.

v11.2 fix: has_clinical_context parameter
  When a session has active symptoms, simple mid-session greetings ("hey", "hi")
  no longer fall through to the LLM. greeting_response() returns a contextual
  "I'm still here" message instead. This prevents the LLM from generating
  "Hello! It seems like you might have more questions..." when the patient
  just says "hey" mid-conversation.

IMPORTANT: CONTEXT_DEPENDENT words (yes, no, ok, etc.) must NEVER be treated
as greetings mid-conversation. They are direct answers to clinical questions.
"""
from __future__ import annotations
import re
from typing import Optional

_GREETINGS = {
    "hi", "hey", "hello", "hiya", "howdy", "greetings", "sup", "yo", "hai",
    "good morning", "good afternoon", "good evening", "good night", "good day",
    "morning", "afternoon", "evening",
    "thanks", "thank you", "thank you so much", "thanks a lot", "thank u",
    "ty", "thnx", "thx", "many thanks", "much appreciated", "appreciate it",
    "appreciate that", "that's helpful", "thats helpful", "very helpful",
    "that helped", "that was helpful", "that was very helpful",
    "great", "awesome", "amazing", "wonderful", "excellent", "fantastic", "nice",
    "good", "very good", "cool", "perfect", "brilliant", "superb", "impressive",
    "wow", "oh wow", "that's great", "thats great", "that's amazing",
    "that's good", "thats good", "good to know", "interesting",
    "please continue", "go on", "continue", "go ahead", "please go on",
    "tell me more", "and then", "and", "what else", "ok go on", "okay go on",
    "bye", "goodbye", "good bye", "see you", "see ya", "take care", "later",
    "talk later", "ttyl", "cya", "have a good day", "have a nice day",
    "farewell", "until next time",
    "who are you", "what are you", "what is neurocheck", "what's neurocheck",
    "whats neurocheck", "what can you do", "how does this work",
    "how do you work", "what do you do", "tell me about yourself",
    "are you a doctor", "are you an ai", "are you real", "are you human",
    "what", "huh", "sorry", "pardon", "can you repeat", "come again",
    "i didn't understand", "i don't understand", "please repeat",
    "say that again", "what do you mean", "what does that mean",
    "hello ji", "hi ji", "namaste", "namaskar", "jai hind",
    "haan ji", "nahi", "theek hai", "theek", "bilkul", "accha",
    "thoda", "shukriya", "dhanyawad",
}

_GREETING_PATTERNS = [
    r"^(hi|hey|hello|good[\s]\w+)[!. ]*$",
    r"^(thanks?( you)?[\w ]*)$",
    r"^(that.s (helpful|great|good|amazing|clear))[!. ]*$",
    r"^(how are you|how.s it going)[?!. ]*$",
    r"^(can you help me|i need help)[?!. ]*$",
]

# Valid answers to clinical questions — NEVER classify as greetings mid-conversation
CONTEXT_DEPENDENT = {
    "yes", "yeah", "yep", "yup", "no", "nope", "nah", "ok", "okay", "k", "kk",
    "sure", "alright", "got it", "understood", "correct", "exactly", "right",
    "true", "not really", "maybe", "possibly", "i think so", "i guess", "hmm",
    "huh", "what", "go on", "continue", "please continue", "go ahead", "and",
    "good", "great", "nice", "cool", "interesting", "i see", "oh i see",
    "makes sense", "noted", "that's right", "thats right", "i don't know",
    "idk", "not sure", "haan", "nahi", "accha", "theek",
}

# Simple one-word/short greetings that should NOT fall through to LLM mid-session
_SIMPLE_GREETINGS = {"hi", "hey", "hello", "hiya", "howdy", "hai", "yo", "sup"}


def is_greeting(
    text: str,
    has_history: bool = False,
    has_clinical_context: bool = False,
) -> bool:
    """
    Return True only if the message is a pure greeting/meta message.

    has_history:          True if at least one assistant message exists in session.
    has_clinical_context: True if the session has confirmed MS features/symptoms.

    CRITICAL: when has_history=True, context-dependent words (yes, no, ok, etc.)
    are NEVER greetings — they are clinical answers that must reach the LLM.
    """
    cleaned = text.strip().lower().rstrip("!.,? ")

    # Mid-conversation: context-dependent words are clinical answers, not greetings
    if has_history and cleaned in CONTEXT_DEPENDENT:
        return False

    if cleaned in _GREETINGS:
        return True

    return any(re.match(pat, cleaned) for pat in _GREETING_PATTERNS)


def greeting_response(
    msg: str,
    is_first_turn: bool,
    has_clinical_context: bool = False,
    phase: str = "gathering",
) -> Optional[str]:
    """
    Return a fast-path response, or None to let the LLM handle it.

    v11.2: When has_clinical_context=True and user sends a simple mid-session
    greeting, return a contextual reminder instead of None (which would let the
    LLM generate an unhelpful generic greeting).
    """
    msg_lower = msg.lower().strip()

    # ── Meta questions (always handle regardless of session state) ────────────
    if any(x in msg_lower for x in [
        "who are you", "what are you", "what is neurocheck", "whats neurocheck",
        "what can you do", "tell me about", "how does this work",
        "how do you work", "what do you do",
    ]):
        return (
            "I'm AxonIQ, an AI-powered clinical decision support assistant "
            "specialising in Multiple Sclerosis (MS) and related neurological conditions.\n\n"
            "I was designed by Dr. Avasarala (MD PhD, University of Kentucky) and "
            "Dr. Kadambari (PhD, NIT Warangal).\n\n"
            "Please describe your symptoms and I'll begin a structured clinical assessment."
        )

    if any(x in msg_lower for x in [
        "are you a doctor", "are you an ai", "are you real",
        "are you human", "are you a robot",
    ]):
        return (
            "I'm an AI assistant — not a doctor. I provide clinical decision support "
            "to help identify patterns consistent with MS, but I cannot diagnose.\n\n"
            "Could you tell me what symptoms you've been experiencing?"
        )

    if any(x in msg_lower for x in ["can you help", "i need help", "help me", "need some help"]):
        return (
            "Absolutely, I'm here to help. Please describe your neurological symptoms "
            "in as much detail as you can — what you're feeling, where it is, when it "
            "started, and whether it has happened before."
        )

    if any(x in msg_lower for x in [
        "sorry", "didn't understand", "don't understand", "what do you mean",
        "can you repeat", "say that again", "pardon",
    ]):
        return (
            "No problem! Let me know which part was unclear and I'll explain differently. "
            "Or just describe your symptoms and I'll start a fresh assessment."
        )

    if any(x in msg_lower for x in [
        "bye", "goodbye", "good bye", "see you", "take care", "farewell", "ttyl",
    ]):
        return "Take care! If your symptoms change or new ones develop, don't hesitate to return. Wishing you good health."

    if any(x in msg_lower for x in [
        "thank", "thanks", "ty", "thnx", "appreciate", "helpful", "that helped",
    ]):
        return "You're welcome! Is there anything else you'd like me to assess?"

    # ── Mid-session simple greeting with active clinical context ───────────────
    # v11.2 fix: prevents LLM from outputting "Hello! How can I assist you?"
    # when user just says "hey" mid-conversation with active symptoms.
    cleaned = msg_lower.rstrip("!.,? ")
    if not is_first_turn and has_clinical_context and cleaned in _SIMPLE_GREETINGS:
        if phase == "mri_requested":
            return (
                "Hello! I'm still here. Based on our conversation so far, "
                "the next step is for you to share your MRI results.\n\n"
                "If you have a written report from your radiologist, please paste it here. "
                "If you have a scan file (.nii.gz), you can upload it. "
                "If you haven't had a scan yet, please ask your doctor for a "
                "Brain and Spinal Cord MRI with and without contrast."
            )
        return (
            "Hello! I'm still here. Please feel free to continue describing "
            "your symptoms, or ask me anything about what we've discussed so far."
        )

    # ── First-turn welcome ────────────────────────────────────────────────────
    if is_first_turn:
        return (
            "Hello! I'm AxonIQ, your MS clinical decision support assistant.\n\n"
            "I'm here to help evaluate neurological symptoms that may be related to "
            "Multiple Sclerosis. Could you describe what you've been experiencing? "
            "Please include when it started, which part of the body is affected, "
            "and whether it has happened before."
        )

    return None
