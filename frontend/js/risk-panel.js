/**
 * AxonIQ — Risk Panel Module
 *
 * Design decisions:
 *   1. Panel is updated DIRECTLY from /chat response.risk_panel — no extra API call per message.
 *   2. /analyze is only called when loading a session from history (no risk_panel available).
 *   3. MRI section is shown/hidden by the agent response — not by frontend guessing.
 *   4. Workup/action are removed — LLM delivers these naturally in prose.
 */

const TIER_COLORS = {
  LOW:                '#065F46',
  WATCH:              '#5B21B6',
  MODERATE:           '#1E40AF',
  HIGH:               '#92400E',
  CRITICAL_EMERGENCY: '#B91C1C',
};

// ── Panel open/close/reset ────────────────────────────────────────────────────

function closeRiskPanel() {
  document.getElementById('riskPanel')?.classList.add('hidden');
}

function resetRiskPanel() {
  mriRequested = false;
  mriSubmitted = false;
  mriAnalysed  = false;
  const panel   = document.getElementById('riskPanel');
  const idle    = document.getElementById('riskIdle');
  const content = document.getElementById('riskContent');
  panel?.classList.add('hidden');
  if (idle)    idle.style.display    = 'block';
  if (content) content.style.display = 'none';
}

// ── Main render — called from chat.js with data.risk_panel and concluded flag ─

/**
 * Only renders the risk panel when concluded=true (final assessment turn).
 * On all other turns the panel stays hidden — no mid-conversation updates.
 */
function renderRiskPanel(data, concluded) {
  if (!data || !concluded) return;   // ← hide panel on every non-conclusion turn
  const tier  = (data.tier || 'LOW').toUpperCase();
  const color = TIER_COLORS[tier] || TIER_COLORS.LOW;

  const panel   = document.getElementById('riskPanel');
  const idle    = document.getElementById('riskIdle');
  const content = document.getElementById('riskContent');
  if (!panel || !idle || !content) return;

  panel.classList.remove('hidden');
  idle.style.display    = 'none';
  content.style.display = 'block';

  // Tier label
  const lbl = document.getElementById('rpTierLabel');
  const sub  = document.getElementById('rpTierSub');
  if (lbl) {
    lbl.textContent = data.tier_label || tier;
    lbl.className   = 'rp-tier-label tier-color-' + tier;
  }
  if (sub) sub.textContent = data.tier_subtitle || '';

  // Detected features
  const features = Array.isArray(data.features)
    ? data.features.map(f => typeof f === 'string' ? f : (f.label || f)).filter(Boolean)
    : [];
  const fSec  = document.getElementById('rpFeaturesSection');
  const fList = document.getElementById('rpFeatureList');
  if (fSec && fList) {
    if (features.length) {
      fSec.style.display = 'block';
      fList.innerHTML = features
        .map(f => `<div class="feature-chip">${escapeHtml(String(f))}</div>`)
        .join('');
    } else {
      fSec.style.display = 'none';
    }
  }

  // MRI section — show when agent has requested MRI and user hasn't submitted yet
  _updateMriSection(data);
}

// ── MRI Section ───────────────────────────────────────────────────────────────

/**
 * Show MRI upload section when agent goal was request_mri.
 * Detect this from the agent's prose containing MRI request keywords.
 * Once MRI submitted → show done badge.
 */
function _updateMriSection(data) {
  const mriBox = document.getElementById('rpMriSection');
  if (!mriBox) return;

  if (mriAnalysed) {
    // MRI already processed — show done badge
    mriBox.innerHTML = `
      <div class="mri-done-box">
        <span class="mri-done-icon">🔬</span>
        <div>
          <strong>MRI Analysis Complete</strong><br>
          Findings have been incorporated into your risk assessment.
        </div>
      </div>`;
    mriBox.style.display = 'block';
    return;
  }

  if (mriRequested && !mriSubmitted) {
    // Show MRI input widget
    mriBox.innerHTML = `
      <div class="mri-request-box">
        <div class="mri-title">🔬 MRI Report Available?</div>
        <p>Paste your radiologist report or scan notes below. The AI will analyse your lesion findings and incorporate them into the risk assessment.</p>
        <textarea
          class="mri-textarea"
          id="mriTextInput"
          placeholder="e.g. MRI brain shows multiple T2/FLAIR hyperintensities in periventricular and juxtacortical regions…"
        ></textarea>
        <button class="mri-submit-btn" onclick="submitMriText()">Analyse MRI Report</button>
      </div>`;
    mriBox.style.display = 'block';
    return;
  }

  mriBox.style.display = 'none';
}

/**
 * Called by the MRI submit button.
 * Sends MRI text as a chat message so it goes through the full agentic pipeline.
 */
function submitMriText() {
  const textarea = document.getElementById('mriTextInput');
  if (!textarea) return;
  const text = textarea.value.trim();
  if (!text || text.length < 20) {
    textarea.style.borderColor = 'var(--red)';
    textarea.placeholder = 'Please paste your MRI report text (at least a few sentences)…';
    return;
  }
  mriSubmitted = true;

  // Send as normal chat message — agent will detect MRI keywords and route to analyse_mri node
  const input = document.getElementById('msgInput');
  if (input) {
    input.value = text;
    sendMessage();
  }
}

/**
 * Called from chat.js after receiving /chat response.
 * Detects if agent requested MRI or analysed MRI based on response content.
 */
function updateMriState(responseText, riskPanel) {
  // Detect MRI request in agent response
  const mriRequestPhrases = [
    'mri scan', 'mri of your brain', 'radiologist report',
    'scan notes', 'mri findings', 'have you had an mri',
    'paste your', 'share your mri',
  ];
  const lower = (responseText || '').toLowerCase();
  if (!mriRequested && mriRequestPhrases.some(p => lower.includes(p))) {
    mriRequested = true;
  }

  // Detect MRI analysis completed — features will include "MRI:" prefix
  const features = riskPanel?.features || [];
  if (features.some(f => String(f).startsWith('MRI:'))) {
    mriAnalysed  = true;
    mriSubmitted = true;
  }
}

// ── History rebuild — called when loading a session ──────────────────────────

/**
 * Re-runs /analyze only for session history loading
 * where we have no risk_panel from a fresh /chat call.
 */
async function rebuildRiskFromHistory(messages) {
  // Only show panel if conclusion was delivered in this session
  const conclusionMarker = "clinical decision support only and is not a diagnosis";
  const wasConcluded = (messages || []).some(
    m => m.role === 'assistant' && m.content?.toLowerCase().includes(conclusionMarker)
  );
  if (!wasConcluded) return;   // session not concluded yet — keep panel hidden

  const humanMessages = (messages || [])
    .filter(m => m.role === 'human')
    .map(m => m.content)
    .join(' | ');

  if (humanMessages.trim().length < 5) return;

  try {
    const res = await fetch(`${API}/analyze`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ text: humanMessages }),
    });
    if (!res.ok) return;
    const data = await res.json();
    if (data?.tier) renderRiskPanel(data, true);   // pass concluded=true for history rebuild
  } catch {
    // Non-critical — silent fail
  }
}
