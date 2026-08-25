/**
 * AxonIQ — Shell Module
 * Handles: app shell reveal, sidebar, session list, session loading/deletion.
 */

function showApp() {
  const authOverlay = document.getElementById('authOverlay');
  const appShell    = document.getElementById('appShell');
  if (authOverlay) authOverlay.style.display = 'none';
  if (appShell)    appShell.classList.add('visible');
  if (currentUser) {
    const avatar   = document.getElementById('headerAvatar');
    const username = document.getElementById('headerUsername');
    if (avatar)   avatar.textContent   = currentUser.username[0].toUpperCase();
    if (username) username.textContent = currentUser.username;
  }
  loadSessionList();
}

function toggleUserMenu() {
  document.getElementById('userDropdown')?.classList.toggle('open');
}

function toggleSidebar() {
  document.getElementById('sidebar')?.classList.toggle('collapsed');
}

// ── Session list ──────────────────────────────────────────────────────────────

async function loadSessionList() {
  try {
    const res = await authFetch('/user/sessions');
    if (res.status === 401) { signOut(); return; }
    const data = await res.json();
    renderSessionList(data.sessions || []);
  } catch (e) {
    console.error('[shell] loadSessionList error:', e);
  }
}

function renderSessionList(sessions) {
  const list  = document.getElementById('sessionList');
  const empty = document.getElementById('sidebarEmpty');
  if (!list) return;

  list.querySelectorAll('.session-item').forEach(el => el.remove());

  if (!sessions.length) {
    if (empty) empty.style.display = 'block';
    return;
  }
  if (empty) empty.style.display = 'none';

  sessions.forEach(s => {
    const item        = document.createElement('div');
    item.className    = 'session-item' + (s.session_uuid === sessionId ? ' active' : '');
    item.dataset.uuid = s.session_uuid;

    const ts = new Date(s.updated_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });

    // Show MRI badge if session has MRI analysis
    const mriBadge = s.ms_symptoms && JSON.stringify(s.ms_symptoms).includes('MRI:')
      ? ' <span style="font-size:10px;color:var(--mri-text)">🔬</span>' : '';

    item.innerHTML = `
      <div class="session-item-icon">💬</div>
      <div class="session-item-body">
        <div class="session-title">${escapeHtml(s.title || 'Chat')}${mriBadge}</div>
        <div class="session-meta">${ts}</div>
      </div>
      <button class="session-delete" onclick="deleteSession(event,'${s.session_uuid}')">✕</button>`;
    item.addEventListener('click', () => loadSession(s.session_uuid));
    list.appendChild(item);
  });
}

async function loadSession(uuid) {
  if (uuid === sessionId) return;
  sessionId = uuid;

  // Reset MRI state for new session load
  mriRequested = false;
  mriSubmitted = false;
  mriAnalysed  = false;

  document.querySelectorAll('.session-item').forEach(el =>
    el.classList.toggle('active', el.dataset.uuid === uuid)
  );

  clearMessages();

  const loading = document.getElementById('sessionLoading');
  if (loading) loading.classList.add('visible');

  try {
    const res = await authFetch(`/session/${uuid}/history`);
    if (loading) loading.classList.remove('visible');

    if (res.status === 401) { signOut(); return; }
    if (!res.ok) return;

    const data = await res.json();
    const messages = data.messages || [];

    if (messages.length) {
      const welcome = document.getElementById('welcome');
      if (welcome) welcome.style.display = 'none';

      messages.forEach(m =>
        addBubble(m.role === 'human' ? 'user' : 'ai', m.content, m.created_at)
      );

      // Rebuild MRI state from loaded messages
      messages.forEach(m => {
      });

      // Rebuild risk panel from history via /analyze
      // Pass full messages so rebuildRiskFromHistory can separate turns properly
      await rebuildRiskFromHistory(messages);
    }
  } catch (e) {
    if (loading) loading.classList.remove('visible');
    console.error('[shell] loadSession error:', e);
  }

  const sessionLabel = document.getElementById('sessionLabel');
  if (sessionLabel) sessionLabel.textContent = `Session: ${uuid.slice(0,8)}…`;
}

async function deleteSession(e, uuid) {
  e.stopPropagation();
  if (!confirm('Delete this conversation?')) return;
  try {
    const res = await authFetch(`/session/${uuid}`, { method: 'DELETE' });
    if (res.status === 401) { signOut(); return; }
  } catch { /* ignore network errors — proceed with UI cleanup */ }
  if (sessionId === uuid) {
    sessionId = null;
    clearMessages();
    const lbl = document.getElementById('sessionLabel');
    if (lbl) lbl.textContent = '';
    document.querySelectorAll('.session-item').forEach(el => el.classList.remove('active'));
  }
  loadSessionList();
}
