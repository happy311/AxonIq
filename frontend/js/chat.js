/**
 * AxonIQ — Chat Module
 *
 * System design:
 *  - sendMessage() handles normal text chat
 *  - uploadMri() handles .nii.gz file upload → then triggers a chat message
 *  - No risk panel, no tier display — backend classification is internal
 */

// ── Input initialisation ──────────────────────────────────────────────────────

function initChatInput() {
  const input = document.getElementById('msgInput');
  if (!input) return;
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 160) + 'px';
  });
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
}

function sendChip(text) {
  const input = document.getElementById('msgInput');
  if (input) { input.value = text; sendMessage(); }
}

// ── Send text message ─────────────────────────────────────────────────────────

// ── MRI lock state ────────────────────────────────────────────────────────────
// While an MRI analysis is running the entire chat is frozen.
// Nothing can be typed or sent until the result (or error) arrives.
let _mriLockActive    = false;
let _mriTimerInterval = null;
let _mriTimerBubble   = null;
let _mriPollTimeoutId = null;

// ── Send text message ─────────────────────────────────────────────────────────

async function sendMessage() {
  const input   = document.getElementById('msgInput');
  const sendBtn = document.getElementById('sendBtn');
  if (!input || !sendBtn) return;

  const text = input.value.trim();
  // Bail out if MRI analysis is in progress — chat is locked until result arrives.
  if (!text || isWaiting || _mriLockActive || !authToken) return;

  isWaiting          = true;
  input.value        = '';
  input.style.height = 'auto';
  sendBtn.disabled   = true;

  const welcome = document.getElementById('welcome');
  if (welcome) welcome.style.display = 'none';

  addBubble('user', text, new Date().toISOString());
  const typing = addTyping();

  // AbortController: 30-min hard ceiling on every chat request.
  const controller   = new AbortController();
  const abortTimerId = setTimeout(() => controller.abort(), 30 * 60 * 1000);

  try {
    const res = await fetch(`${API}/chat`, {
      method:  'POST',
      headers: {
        'Content-Type':  'application/json',
        'Authorization': `Bearer ${authToken}`,
      },
      body:   JSON.stringify({ message: text, session_id: sessionId }),
      signal: controller.signal,
    });

    clearTimeout(abortTimerId);
    if (typing && typing.isConnected) typing.remove();

    if (res.status === 401) { signOut(); return; }
    if (res.status === 429) {
      addBubble('ai', '⚠️ You are sending messages too quickly. Please wait a moment.', new Date().toISOString());
      return;
    }
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server error (${res.status})`);
    }

    const data = await res.json();
    sessionId  = data.session_id;

    const lbl = document.getElementById('sessionLabel');
    if (lbl) lbl.textContent = `Session: ${sessionId.slice(0, 8)}…`;

    addBubble('ai', data.response, new Date().toISOString());
    loadSessionList();

  } catch (err) {
    clearTimeout(abortTimerId);
    if (typing && typing.isConnected) typing.remove();

    const isAbort = err.name === 'AbortError';
    addBubble('ai',
      isAbort
        ? '⏱️ Request timed out after 30 minutes. Please try again.'
        : `⚠️ ${err.message}`,
      new Date().toISOString()
    );
  } finally {
    isWaiting        = false;
    sendBtn.disabled = false;
    input.focus();
  }
}

// ── MRI chat lock / unlock ────────────────────────────────────────────────────

function lockChatForMri() {
  _mriLockActive = true;
  const input   = document.getElementById('msgInput');
  const sendBtn = document.getElementById('sendBtn');
  const mriBtn  = document.getElementById('mriToggleBtn');

  if (input) {
    input.disabled    = true;
    input.placeholder = '⏳ MRI analysis running — chat locked until result arrives…';
  }
  if (sendBtn) sendBtn.disabled = true;
  if (mriBtn)  mriBtn.disabled  = true;
}

function unlockChatAfterMri() {
  _mriLockActive = false;

  // Stop and clear the timer
  if (_mriTimerInterval) { clearInterval(_mriTimerInterval); _mriTimerInterval = null; }

  // Stop any pending status poll (e.g. unlock triggered by sign-out mid-poll)
  if (_mriPollTimeoutId) { clearTimeout(_mriPollTimeoutId); _mriPollTimeoutId = null; }

  // Remove the timer bubble
  if (_mriTimerBubble && _mriTimerBubble.isConnected) {
    _mriTimerBubble.remove();
    _mriTimerBubble = null;
  }

  const input   = document.getElementById('msgInput');
  const sendBtn = document.getElementById('sendBtn');
  const mriBtn  = document.getElementById('mriToggleBtn');

  if (input) {
    input.disabled    = false;
    input.placeholder = 'Type your message…';
  }
  if (sendBtn) sendBtn.disabled = false;
  if (mriBtn)  mriBtn.disabled  = false;

  input?.focus();
}

// ── MRI timer bubble ──────────────────────────────────────────────────────────
// Shown immediately after upload completes; counts up every second.
// Removed automatically by unlockChatAfterMri().

function addMriTimerBubble() {
  const msgs = document.getElementById('messages');
  if (!msgs) return null;

  const row     = document.createElement('div');
  row.className = 'msg-row ai';

  const av       = document.createElement('div');
  av.className   = 'avatar ai';
  av.textContent = 'AQ';

  const b       = document.createElement('div');
  b.className   = 'bubble ai mri-wait-bubble';
  b.innerHTML   =
    '🔬 <strong>MRI analysis running…</strong><br>' +
    'Your MRI scans are being analysed. ' +
    'This typically takes <strong>10–20 minutes</strong>. ' +
    '<strong>Please keep this tab open.</strong><br>' +
    '<em>Chat is locked — it will resume automatically when the result arrives.</em><br>' +
    '<span class="mri-timer" id="mriTimerDisplay">⏱ Elapsed: 0 min 0 sec</span>';

  row.appendChild(av);
  row.appendChild(b);
  msgs.appendChild(row);
  msgs.scrollTop = msgs.scrollHeight;

  // Start live 1-second counter
  const startedAt = Date.now();
  if (_mriTimerInterval) clearInterval(_mriTimerInterval);
  _mriTimerInterval = setInterval(() => {
    const secs = Math.floor((Date.now() - startedAt) / 1000);
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    const el = document.getElementById('mriTimerDisplay');
    if (el) el.textContent = `⏱ Elapsed: ${m} min ${s} sec`;
    msgs.scrollTop = msgs.scrollHeight;
  }, 1000);

  return row;
}

// ── MRI analysis: status polling + result fetch ───────────────────────────────
// v12 design: the upload endpoint kicks off analysis in the background and
// returns instantly — no request is ever held open for the 10–20 min the
// analysis actually takes. Instead we poll a cheap /mri/status endpoint every
// few seconds, and once it reports "done" we fetch the result exactly once.
// Each individual request here completes in well under a second; only the
// polling *sequence* spans the full analysis window, which is exactly what
// avoids the long-held-connection problem.
//
// Never use sendMessage() for any of this — it would double-lock and show a
// typing bubble on top of the timer. These functions own the lock completely.

const MRI_POLL_INTERVAL_MS = 5000;       // how often to check status
const MRI_POLL_MAX_WAIT_MS = 30 * 60 * 1000; // 30-min safety ceiling, same as before

function _pollMriStatus() {
  const startedAt = Date.now();

  const poll = async () => {
    if (Date.now() - startedAt > MRI_POLL_MAX_WAIT_MS) {
      unlockChatAfterMri();
      addBubble('ai',
        '⏱️ MRI analysis timed out after 30 minutes. Please try uploading again.',
        new Date().toISOString()
      );
      return;
    }

    let res;
    try {
      res = await fetch(`${API}/mri/status/${encodeURIComponent(sessionId)}`, {
        headers: { 'Authorization': `Bearer ${authToken}` },
      });
    } catch (err) {
      // Network hiccup — a quick status ping is safe to just retry, no need
      // to give up the whole analysis over one dropped request.
      _mriPollTimeoutId = setTimeout(poll, MRI_POLL_INTERVAL_MS);
      return;
    }

    if (res.status === 401) { unlockChatAfterMri(); signOut(); return; }

    if (!res.ok) {
      // Transient server/proxy hiccup on a lightweight poll — retry rather
      // than aborting a 10–20 min analysis over one bad response.
      _mriPollTimeoutId = setTimeout(poll, MRI_POLL_INTERVAL_MS);
      return;
    }

    const data = await res.json();

    if (data.status === 'processing' || data.status === 'queued') {
      _mriPollTimeoutId = setTimeout(poll, MRI_POLL_INTERVAL_MS);
      return;
    }

    if (data.status === 'error') {
      unlockChatAfterMri();
      addBubble('ai',
        `⚠️ MRI analysis error: ${escapeHtml(data.error || 'Unknown error')}. You can try uploading again.`,
        new Date().toISOString()
      );
      return;
    }

    if (data.status === 'done') {
      await _fetchMriResult();
      return;
    }

    // status === "none", or anything unrecognised — the job vanished
    // (e.g. server restarted mid-analysis). Don't loop forever.
    unlockChatAfterMri();
    addBubble('ai',
      '⚠️ Could not confirm MRI analysis status. Please try uploading again.',
      new Date().toISOString()
    );
  };

  poll();
}

async function _fetchMriResult() {
  try {
    const res = await fetch(`${API}/mri/result/${encodeURIComponent(sessionId)}`, {
      headers: { 'Authorization': `Bearer ${authToken}` },
    });

    if (res.status === 401) { unlockChatAfterMri(); signOut(); return; }
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server error (${res.status})`);
    }

    const data = await res.json();

    // Unlock first, then show result so input is available immediately
    unlockChatAfterMri();
    addBubble('ai', data.response, new Date().toISOString());
    loadSessionList();

  } catch (err) {
    unlockChatAfterMri();
    addBubble('ai',
      `⚠️ Could not fetch MRI result: ${escapeHtml(err.message)}. You can try uploading again.`,
      new Date().toISOString()
    );
  }
}

// ── MRI Upload (FLAIR-only) ───────────────────────────────────────────────────

// State for the single slot
const _mriFiles = { flair: null };

function toggleMriPanel() {
  const panel = document.getElementById('mriPanel');
  if (!panel) { console.error('[MRI] #mriPanel not found in DOM'); return; }
  const isOpen = panel.style.display !== 'none';
  panel.style.display = isOpen ? 'none' : 'block';
  const btn = document.getElementById('mriToggleBtn');
  if (btn) btn.classList.toggle('active', !isOpen);
}

function closeMriPanel() {
  const panel = document.getElementById('mriPanel');
  if (panel) panel.style.display = 'none';
  const btn = document.getElementById('mriToggleBtn');
  if (btn) btn.classList.remove('active');
}

function handleMriSlot(input, slot) {
  const file = input.files?.[0];
  if (!file) return;
  input.value = '';   // allow re-selecting same file

  if (!file.name.endsWith('.nii.gz') && !file.name.endsWith('.nii')) {
    addBubble('ai',
      `⚠️ Only NIfTI files (.nii.gz or .nii) are accepted for ${slot.toUpperCase()}.`,
      new Date().toISOString()
    );
    return;
  }

  _mriFiles[slot] = file;

  // Update slot UI
  const nameEl = document.getElementById('flairName');
  const slotEl = document.getElementById('flairSlot');
  if (nameEl) nameEl.textContent = `✅ ${file.name} (${(file.size/1024/1024).toFixed(1)} MB)`;
  if (slotEl) slotEl.classList.add('ready');

  _updateMriAnalyzeBtn();
}

function _updateMriAnalyzeBtn() {
  const btn  = document.getElementById('mriAnalyzeBtn');
  const hint = document.getElementById('mriHint');
  const ready = !!_mriFiles.flair;
  if (btn) btn.disabled = !ready;
  if (hint) {
    hint.textContent = ready
      ? '✅ FLAIR file ready — click Analyze MRI'
      : 'Select a FLAIR file to enable analysis';
  }
}

async function submitMriPair() {
  if (!_mriFiles.flair) return;
  if (!sessionId) {
    addBubble('ai',
      'Please start a conversation first before uploading MRI scans.',
      new Date().toISOString()
    );
    return;
  }

  // BUG FIX: Lock chat BEFORE the upload starts, not after it finishes.
  // NIfTI files can be 50-500 MB — the upload alone can take several minutes.
  // During that window, a user message would reach the server, pop the queued
  // nifti_paths (consuming them), and the analysis request would find nothing.
  // Locking early closes that race entirely.
  lockChatForMri();

  // Close panel and show progress
  closeMriPanel();

  const flairSize = (_mriFiles.flair.size / 1024 / 1024).toFixed(1);

  const progressBubble = addBubble('ai',
    `📡 Uploading MRI file…<br>` +
    `&nbsp;&nbsp;FLAIR: <strong>${escapeHtml(_mriFiles.flair.name)}</strong> (${flairSize} MB)`,
    new Date().toISOString()
  );

  try {
    const form = new FormData();
    form.append('flair', _mriFiles.flair);

    const res = await fetch(
      `${API}/mri/upload?session_id=${encodeURIComponent(sessionId)}`,
      {
        method:  'POST',
        headers: { 'Authorization': `Bearer ${authToken}` },
        body:    form,
      }
    );

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Upload failed (${res.status})`);
    }

    const data = await res.json();

    progressBubble.querySelector('.bubble').innerHTML =
      `✅ MRI file uploaded (FLAIR ${data.flair_mb} MB).<br>` +
      `🔬 MRI is being analysed — this may take <strong>10–20 minutes</strong>.<br>` +
      `<em>Chat is locked — it will resume automatically when the result arrives.</em>`;

    // Reset slot state so the same file can be re-uploaded if needed
    _mriFiles.flair = null;
    document.getElementById('flairName').textContent = 'Click to select .nii / .nii.gz';
    document.getElementById('flairSlot')?.classList.remove('ready');
    _updateMriAnalyzeBtn();

    // Show the user's message in chat so the conversation is complete
    addBubble('user', 'I have uploaded my FLAIR MRI scan for analysis.', new Date().toISOString());

    // Start the timer bubble — chat is already locked from the top of this function.
    _mriTimerBubble = addMriTimerBubble();

    // The upload call above already returned with analysis running in the
    // background on the server. Start polling for status instead of holding
    // a request open — no await needed, _pollMriStatus() unlocks when done.
    _pollMriStatus();

  } catch (err) {
    // Upload failed — unlock so the user can retry.
    unlockChatAfterMri();
    progressBubble.querySelector('.bubble').innerHTML =
      `⚠️ Upload failed: ${escapeHtml(err.message)}<br><em>You can try uploading again.</em>`;
  }
}

// ── New chat ──────────────────────────────────────────────────────────────────

function newChat() {
  sessionId = null;
  clearMessages();
  const lbl = document.getElementById('sessionLabel');
  if (lbl) lbl.textContent = '';
  document.querySelectorAll('.session-item').forEach(el => el.classList.remove('active'));
  loadSessionList();
  document.getElementById('msgInput')?.focus();
}

function clearMessages() {
  document.getElementById('messages')
    ?.querySelectorAll('.msg-row')
    .forEach(el => el.remove());
  const welcome = document.getElementById('welcome');
  if (welcome) welcome.style.display = 'flex';
}

// ── Bubble rendering ──────────────────────────────────────────────────────────

function addBubble(role, text, timestamp) {
  const msgs = document.getElementById('messages');
  if (!msgs) return document.createElement('div');

  const row      = document.createElement('div');
  row.className  = `msg-row ${role}`;

  const avatar       = document.createElement('div');
  avatar.className   = `avatar ${role}`;
  avatar.textContent = role === 'ai'
    ? 'AQ'
    : (currentUser ? currentUser.username[0].toUpperCase() : 'U');

  const bubble     = document.createElement('div');
  bubble.className = `bubble ${role}`;
  bubble.innerHTML = formatMessage(text);

  const ts       = document.createElement('div');
  ts.className   = 'msg-ts';
  ts.textContent = formatTimestamp(timestamp);

  const inner = document.createElement('div');
  inner.appendChild(bubble);
  inner.appendChild(ts);

  row.appendChild(avatar);
  row.appendChild(inner);
  msgs.appendChild(row);
  msgs.scrollTop = msgs.scrollHeight;
  return row;
}

function addTyping() {
  const msgs = document.getElementById('messages');
  if (!msgs) return document.createElement('div');

  const row    = document.createElement('div');
  row.className = 'msg-row ai';

  const av       = document.createElement('div');
  av.className   = 'avatar ai';
  av.textContent = 'AQ';

  const b     = document.createElement('div');
  b.className = 'typing-bubble';
  b.innerHTML = '<div class="dot"></div><div class="dot"></div><div class="dot"></div>';

  row.appendChild(av);
  row.appendChild(b);
  msgs.appendChild(row);
  msgs.scrollTop = msgs.scrollHeight;
  return row;
}
