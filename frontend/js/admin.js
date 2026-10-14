/**
 * AxonIQ — Admin panel (external script: the page's CSP forbids inline scripts).
 *
 * XSS rules followed here:
 *  - no inline event handlers; markup carries data-* attributes and ONE delegated listener
 *    dispatches through an allow-list (values are data, never code);
 *  - every interpolated value goes through esc() (escapes & < > " ') — including actions,
 *    error messages, timestamps and CSS class suffixes (cls() additionally whitelists chars);
 *  - the session lives in an HttpOnly cookie; nothing secret is kept in JS-visible storage.
 * Utilities (escapeHtml, authFetch, getCookie) come from utils.js.
 */
const API = window.location.origin;
let allUsers = [];
let selectedUserId = null;
const esc = escapeHtml;
const cls = v => String(v ?? '').replace(/[^A-Za-z0-9_-]/g, '');

// ── Safe formatters ───────────────────────────────────────────────────────────
function fmtDate(iso) {
  try { return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' }); } catch { return String(iso); }
}
function fmtDateFull(iso) {
  try { return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit' }); } catch { return String(iso); }
}
function fmtTime(iso) {
  try {
    const d = new Date(iso), now = new Date();
    return d.toDateString() === now.toDateString()
      ? d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      : d.toLocaleDateString([], { month: 'short', day: 'numeric' }) + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } catch { return String(iso); }
}
const num = v => (Number.isFinite(Number(v)) ? Number(v) : 0);

// ── Delegated events (allow-list) ─────────────────────────────────────────────
const ACTIONS = {
  adminLogin:    () => adminLogin(),
  adminSignOut:  () => adminSignOut(),
  showAllLogs:   () => showAllLogs(),
  showAudit:     () => showAudit(),
  showDoctors:   () => showDoctors(),
  approveDoctor: el => reviewDoctor(num(el.dataset.id), 'approve'),
  removeDoctor:  el => reviewDoctor(num(el.dataset.id), 'remove'),
  selectUser:    el => selectUser(num(el.dataset.id)),
  showUserLogs:  el => showUserLogs(num(el.dataset.id)),
  resetPassword: el => resetPassword(num(el.dataset.id)),
  exportCSV:     el => exportUserCSV(num(el.dataset.id)),
  exportJSON:    el => exportUserJSON(num(el.dataset.id)),
  toggleSession: el => toggleSession(el.dataset.target),
};
document.addEventListener('click', ev => {
  const el = ev.target.closest('[data-action]');
  if (el && Object.prototype.hasOwnProperty.call(ACTIONS, el.dataset.action)) ACTIONS[el.dataset.action](el, ev);
});
document.addEventListener('input', ev => {
  if (ev.target.dataset && ev.target.dataset.input === 'filterUsers') filterUsers();
});

window.addEventListener('DOMContentLoaded', async () => {
  document.getElementById('adminPass').addEventListener('keydown', e => { if (e.key === 'Enter') adminLogin(); });
  try {                                   // already signed in (cookie) and an admin?
    const res = await authFetch('/auth/me');
    if (res.ok && (await res.json()).is_admin) showAdmin();
  } catch { /* stay on login */ }
});

// ── Auth ──────────────────────────────────────────────────────────────────────
async function adminLogin() {
  const user = document.getElementById('adminUser').value.trim();
  const pass = document.getElementById('adminPass').value;
  const err  = document.getElementById('authError');
  err.style.display = 'none';
  if (!user || !pass) { err.textContent = 'Fill in all fields'; err.style.display = 'block'; return; }

  const form = new URLSearchParams();
  form.append('username', user);
  form.append('password', pass);
  try {
    const res = await fetch(`${API}/auth/login`, {
      method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      credentials: 'same-origin', body: form,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(errorText(data, 'Login failed'));
    const me = await (await authFetch('/auth/me')).json();
    if (!me.is_admin) {
      await fetch(`${API}/auth/logout`, { method: 'POST', credentials: 'same-origin' });
      throw new Error('Access denied — not an admin account');
    }
    document.getElementById('adminPass').value = '';
    showAdmin();
  } catch (e) {
    err.textContent = e.message;            // textContent: never parsed as HTML
    err.style.display = 'block';
  }
}

async function adminSignOut() {
  try { await fetch(`${API}/auth/logout`, { method: 'POST', credentials: 'same-origin' }); } catch {}
  document.getElementById('adminShell').classList.remove('visible');
  document.getElementById('authOverlay').style.display = 'flex';
}

function showAdmin() {
  document.getElementById('authOverlay').style.display = 'none';
  document.getElementById('adminShell').classList.add('visible');
  loadDashboard();
}

// ── Dashboard ─────────────────────────────────────────────────────────────────
async function loadDashboard() {
  try {
    const res = await authFetch('/admin/users');
    if (res.status === 401 || res.status === 403) { adminSignOut(); return; }
    const data = await res.json();
    allUsers = data.users || [];
    document.getElementById('statUsers').textContent = allUsers.length;
    document.getElementById('statSessions').textContent = num(data.total_sessions);
    document.getElementById('statMessages').textContent = num(data.total_messages);
    renderUserList(allUsers);
    refreshPendingBadge();
  } catch (e) { console.error(e); }
}

// ── Doctor approvals ──────────────────────────────────────────────────────────
async function refreshPendingBadge() {
  try {
    const res = await authFetch('/admin/doctors?status=pending');
    if (!res.ok) return;
    const n = ((await res.json()).doctors || []).length;
    const b = document.getElementById('pendingBadge');
    b.textContent = n; b.style.display = n ? 'inline-block' : 'none';
  } catch { /* badge is cosmetic */ }
}

async function showDoctors() {
  const main = document.getElementById('mainPanel');
  main.innerHTML = '<div class="loading"><div class="spin"></div> Loading doctors…</div>';
  try {
    const res = await authFetch('/admin/doctors');
    const docs = (await res.json()).doctors || [];
    const row = d => {
      const st = cls(d.doctor_status);
      const btns = (d.doctor_status !== 'approved'
          ? `<button class="export-btn primary" data-action="approveDoctor" data-id="${num(d.id)}">${d.doctor_status === 'removed' ? 'Re-permit' : 'Permit'}</button>` : '')
        + (d.doctor_status !== 'removed'
          ? `<button class="export-btn" style="margin-left:6px;border-color:#FECACA;color:var(--red);" data-action="removeDoctor" data-id="${num(d.id)}">${d.doctor_status === 'pending' ? 'Reject' : 'Remove'}</button>` : '');
      return `<tr>
        <td><strong>${esc(d.username)}</strong><br><span style="color:var(--muted);font-size:12px">${esc(d.email)}</span></td>
        <td>${esc(d.doctor_license || '—')}<br><span style="color:var(--muted);font-size:12px">${esc(d.doctor_institution || '')}</span></td>
        <td><span class="conclusion-tier doc-status-${st}">${esc(d.doctor_status || '')}</span></td>
        <td style="font-size:12px;color:var(--muted)">${esc(fmtDate(d.created_at))}${d.doctor_reviewed_by ? `<br>by ${esc(d.doctor_reviewed_by)}` : ''}</td>
        <td>${d.doctor_status === 'approved' ? num(d.assigned_count) : '—'}</td>
        <td style="white-space:nowrap">${btns}</td></tr>`;
    };
    main.innerHTML = `<h2 style="margin-bottom:6px">Doctor approvals</h2>
      <p style="color:var(--muted);font-size:13px;margin-bottom:16px">Verify the licence number with your medical council before permitting. Removing a doctor ends their sessions immediately.</p>
      ${docs.length ? `<table class="doc-table"><thead><tr><th>Doctor</th><th>Licence / institution</th><th>Status</th><th>Registered</th><th>Patients</th><th></th></tr></thead><tbody>${docs.map(row).join('')}</tbody></table>`
                    : '<div style="color:var(--muted);font-size:14px;padding:20px 0">No doctors have registered yet.</div>'}`;
  } catch (e) {
    main.innerHTML = `<div style="color:var(--red);padding:20px">Error: ${esc(e.message)}</div>`;
  }
}

async function reviewDoctor(id, verb) {
  if (verb === 'remove' && !window.confirm('Remove this doctor\'s access?')) return;
  const res = await authFetch(`/admin/doctors/${id}/${verb}`, { method: 'POST' });
  if (!res.ok) { window.alert('Action failed (' + res.status + ')'); return; }
  await showDoctors();
  refreshPendingBadge();
}

function renderUserList(users) {
  const list = document.getElementById('userList');
  if (!users.length) { list.innerHTML = '<div style="padding:16px;font-size:13px;color:var(--muted)">No users found</div>'; return; }
  list.innerHTML = users.map(u => `
    <div class="user-item ${u.id === selectedUserId ? 'active' : ''}" data-action="selectUser" data-id="${num(u.id)}">
      <div class="user-item-name">${esc(u.username)}<span class="user-badge">${num(u.session_count)}</span>
        ${u.latest_tier ? `<span class="conclusion-tier tier-${cls(u.latest_tier)}" style="font-size:9px;padding:1px 6px;margin-left:4px;">${esc(String(u.latest_tier).replace('_', ' '))}</span>` : ''}
      </div>
      <div class="user-item-meta">${esc(u.email)} · ${esc(fmtDate(u.created_at))}</div>
    </div>`).join('');
}

function filterUsers() {
  const q = document.getElementById('userSearch').value.toLowerCase();
  renderUserList(allUsers.filter(u => u.username.toLowerCase().includes(q) || u.email.toLowerCase().includes(q)));
}

// ── User detail ───────────────────────────────────────────────────────────────
async function selectUser(id) {
  const u = allUsers.find(x => x.id === id);
  if (!u) return;
  selectedUserId = id;
  document.querySelectorAll('.user-item').forEach(el => el.classList.toggle('active', num(el.dataset.id) === id));

  const main = document.getElementById('mainPanel');
  main.innerHTML = `<div class="loading"><div class="spin"></div> Loading conversations…</div>`;
  try {
    const res = await authFetch(`/admin/user/${id}/sessions`);
    const data = await res.json();
    const sessions = data.sessions || [];
    main.innerHTML = `
      <div class="user-header">
        <div class="user-avatar-lg">${esc((u.username[0] || '?').toUpperCase())}</div>
        <div class="user-header-info">
          <h2>${esc(u.username)}</h2>
          <p>${esc(u.email)} &nbsp;·&nbsp; Joined ${esc(fmtDate(u.created_at))}</p>
        </div>
        <div class="user-stats">
          <div class="user-stat"><div class="n">${num(u.session_count)}</div><div class="l">Sessions</div></div>
          <div class="user-stat"><div class="n">${num(u.message_count)}</div><div class="l">Messages</div></div>
          <button class="export-btn" style="margin-left:8px;" data-action="showUserLogs" data-id="${num(id)}">📋 Logs</button>
          <button class="export-btn" style="margin-left:4px;border-color:#FECACA;color:var(--red);" data-action="resetPassword" data-id="${num(id)}">🔑 Send reset link</button>
        </div>
      </div>
      <div class="export-bar">
        <button class="export-btn primary" data-action="exportCSV" data-id="${num(id)}">⬇ Export CSV</button>
        <button class="export-btn" data-action="exportJSON" data-id="${num(id)}">⬇ Export JSON</button>
        <span class="section-label">${sessions.length} conversation${sessions.length !== 1 ? 's' : ''}</span>
      </div>
      <div class="sessions-grid" id="sessionsGrid">
        ${sessions.length === 0
          ? '<div style="color:var(--muted);font-size:14px;padding:20px 0">No conversations yet.</div>'
          : sessions.map((s, i) => renderSessionCard(s, i)).join('')}
      </div>`;
  } catch (e) {
    main.innerHTML = `<div style="color:var(--red);padding:20px">Error loading sessions: ${esc(e.message)}</div>`;
  }
}

function renderSessionCard(s, idx) {
  const msgs = s.messages || [];
  const humanCount = msgs.filter(m => m.role === 'human').length;
  const c = s.conclusion;
  const tier = c ? cls(c.tier) : '';

  const conclusionHtml = c ? `
    <div class="conclusion-bar">
      <span class="conclusion-tier tier-${tier}">${esc(c.tier_label || c.tier)}</span>
      ${c.symptoms && c.symptoms.length ? `
        <div class="conclusion-block">
          <div class="conclusion-label">Detected Symptoms</div>
          <div class="conclusion-items">${c.symptoms.map(sym => `<span class="conclusion-chip">${esc(sym)}</span>`).join('')}</div>
        </div>` : ''}
      ${c.workup && c.workup.length ? `
        <div class="conclusion-block">
          <div class="conclusion-label">Recommended Workup</div>
          <div class="conclusion-items">${c.workup.slice(0, 3).map(w => `<span class="conclusion-chip">${esc(String(w).substring(0, 40) + (String(w).length > 40 ? '…' : ''))}</span>`).join('')}</div>
        </div>` : ''}
      ${c.action ? `
        <div class="conclusion-block" style="flex:2">
          <div class="conclusion-label">Clinical Action</div>
          <div class="conclusion-action">${esc(c.action)}</div>
        </div>` : ''}
    </div>` : `<div class="no-conclusion">No MS classification recorded for this session</div>`;

  return `
    <div class="session-card">
      <div class="session-card-header" data-action="toggleSession" data-target="sess_${num(idx)}">
        <div class="session-icon">💬</div>
        <div class="session-title-wrap">
          <div class="session-title">${esc(s.title || 'Untitled Chat')}
            ${c ? `<span class="conclusion-tier tier-${tier}" style="margin-left:8px;font-size:11px;">${esc(c.tier_label || c.tier)}</span>` : ''}
          </div>
          <div class="session-date">${esc(fmtDateFull(s.created_at))} &nbsp;·&nbsp; ${humanCount} message${humanCount !== 1 ? 's' : ''} from user</div>
        </div>
        <div class="session-toggle" id="toggle_sess_${num(idx)}">▼</div>
      </div>
      <div class="session-body" id="sess_${num(idx)}">
        ${conclusionHtml}
        ${msgs.map(m => `
          <div class="msg-row ${cls(m.role)}">
            <div class="msg-role"><span class="msg-role-badge ${cls(m.role)}">${m.role === 'human' ? 'Patient' : 'AxonIQ'}</span></div>
            <div class="msg-content">${esc(m.content)}</div>
            <div class="msg-time">${esc(fmtTime(m.created_at))}</div>
          </div>`).join('')}
      </div>
    </div>`;
}

function toggleSession(id) {
  if (!/^sess_\d+$/.test(id || '')) return;
  const body   = document.getElementById(id);
  const toggle = document.getElementById('toggle_' + id);
  const header = body.previousElementSibling;
  const isOpen = body.classList.contains('open');
  body.classList.toggle('open', !isOpen);
  toggle.classList.toggle('open', !isOpen);
  header.classList.toggle('open', !isOpen);
}

// ── Export (CSV is formula-injection-safe and correctly quoted) ───────────────
function csvCell(v) {
  let s = String(v ?? '');
  if (/^[=+\-@\t\r]/.test(s)) s = "'" + s;          // neutralise Excel/Sheets formulas
  return '"' + s.replace(/"/g, '""') + '"';          // RFC 4180 quoting
}

async function exportUserCSV(userId) {
  const u = allUsers.find(x => x.id === userId); if (!u) return;
  const res  = await authFetch(`/admin/user/${userId}/sessions`);
  const data = await res.json();
  const rows = [['username', 'session_title', 'role', 'content', 'timestamp']];
  for (const s of (data.sessions || []))
    for (const m of (s.messages || []))
      rows.push([u.username, s.title || '', m.role, m.content, m.created_at]);
  download(`axoniq_user_${num(userId)}.csv`, rows.map(r => r.map(csvCell).join(',')).join('\r\n'), 'text/csv');
}

async function exportUserJSON(userId) {
  const res  = await authFetch(`/admin/user/${userId}/sessions`);
  const data = await res.json();
  download(`axoniq_user_${num(userId)}.json`, JSON.stringify(data, null, 2), 'application/json');
}

function download(filename, content, type) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([content], { type }));
  a.download = filename;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// ── Logs / audit ──────────────────────────────────────────────────────────────
async function showAllLogs() {
  selectedUserId = null;
  document.querySelectorAll('.user-item').forEach(el => el.classList.remove('active'));
  const main = document.getElementById('mainPanel');
  main.innerHTML = `<div class="loading"><div class="spin"></div> Loading logs…</div>`;
  try {
    const data = await (await authFetch('/admin/logs')).json();
    renderLogsTable(main, data.logs || [], 'All User Activity');
  } catch (e) { main.innerHTML = `<div style="color:var(--red);padding:20px">Error: ${esc(e.message)}</div>`; }
}

async function showUserLogs(userId) {
  const u = allUsers.find(x => x.id === userId); if (!u) return;
  const main = document.getElementById('mainPanel');
  const existing = document.getElementById(`user_logs_${userId}`);
  if (existing) { existing.remove(); return; }
  try {
    const data = await (await authFetch(`/admin/user/${userId}/logs`)).json();
    const container = document.createElement('div');
    container.id = `user_logs_${num(userId)}`;
    container.style.marginTop = '20px';
    renderLogsTable(container, data.logs || [], `Activity Log — ${u.username}`);
    main.appendChild(container);
    container.scrollIntoView({ behavior: 'smooth' });
  } catch (e) { console.error(e); }
}

function renderLogsTable(container, logs, title) {
  const h = `<h3 style="font-family:'Libre Baskerville',serif;font-size:17px;color:var(--navy);margin-bottom:16px;">${esc(title)}</h3>`;
  if (!logs.length) { container.innerHTML = `${h}<p style="color:var(--muted);font-size:13px;">No activity logged yet.</p>`; return; }
  container.innerHTML = `
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;">${h}
      <span style="font-size:12px;color:var(--muted);">${logs.length} events</span></div>
    <table class="log-table">
      <thead><tr><th>Time</th><th>User</th><th>Action</th><th>Detail</th><th>IP</th></tr></thead>
      <tbody>${logs.map(l => `
        <tr>
          <td style="font-family:'DM Mono',monospace;white-space:nowrap;color:var(--muted);">${esc(fmtTime(l.created_at))}</td>
          <td style="font-weight:500;">${esc(l.username || '—')}</td>
          <td><span class="log-action log-${cls(l.action)}">${esc(l.action)}</span></td>
          <td style="max-width:320px;word-break:break-word;">${esc(l.detail || '—')}</td>
          <td style="font-family:'DM Mono',monospace;font-size:11px;color:var(--muted);">${esc(l.ip_address || '—')}</td>
        </tr>`).join('')}
      </tbody>
    </table>`;
}

async function showAudit() {
  selectedUserId = null;
  document.querySelectorAll('.user-item').forEach(el => el.classList.remove('active'));
  const main = document.getElementById('mainPanel');
  main.innerHTML = `<div class="loading"><div class="spin"></div> Loading audit trail…</div>`;
  try {
    const data = await (await authFetch('/admin/audit')).json();
    const rows = data.entries || [];
    main.innerHTML = `<h3 style="font-family:'Libre Baskerville',serif;font-size:17px;color:var(--navy);margin-bottom:6px;">Admin audit trail</h3>
      <p style="color:var(--muted);font-size:12px;margin-bottom:14px;">Append-only. Every admin action — including reading patient chats — is recorded.</p>
      <table class="log-table"><thead><tr><th>Time</th><th>Admin</th><th>Action</th><th>Target</th><th>IP</th></tr></thead><tbody>
      ${rows.map(r => `<tr>
        <td style="white-space:nowrap;color:var(--muted);">${esc(fmtDateFull(r.created_at))}</td>
        <td>${esc(r.admin_username || ('#' + (r.admin_id ?? '—')))}</td>
        <td><span class="log-action">${esc(r.action)}</span></td>
        <td>${esc(r.target || '—')}</td><td>${esc(r.ip || '—')}</td></tr>`).join('')}
      </tbody></table>`;
  } catch (e) { main.innerHTML = `<div style="color:var(--red);padding:20px">Error: ${esc(e.message)}</div>`; }
}

// ── Password reset: a one-time link is e-mailed to the user; the admin never sees a password ──
async function resetPassword(userId) {
  const u = allUsers.find(x => x.id === userId); if (!u) return;
  if (!confirm(`Send a one-time password-reset link to ${u.username}?\n\nTheir current password and all sessions are invalidated immediately.`)) return;
  try {
    const res  = await authFetch(`/admin/user/${userId}/reset-password`, { method: 'POST' });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(errorText(data, 'Failed'));
    alert(`✅ Reset link e-mailed to ${data.username}. It is single-use and expires shortly.` +
          (data.dev_link ? `\n\n[DEV ONLY] ${data.dev_link}` : ''));
  } catch (e) { alert('Error: ' + e.message); }
}
