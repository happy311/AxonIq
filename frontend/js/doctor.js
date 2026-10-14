/**
 * AxonIQ — Doctor portal (external script; CSP forbids inline scripts).
 * Same XSS rules as admin.js: delegated data-action events, every interpolated value through esc(),
 * session lives in an HttpOnly cookie. Utilities (escapeHtml, authFetch, errorText) from utils.js.
 */
const API = window.location.origin;
const esc = escapeHtml;
const cls = v => String(v ?? '').replace(/[^A-Za-z0-9_-]/g, '');
const num = v => (Number.isFinite(Number(v)) ? Number(v) : 0);
let patients = [];
let assigned = [];
let selected = null;

function fmtDate(iso) {
  try { return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' }); } catch { return String(iso); }
}
function fmtMB(b) { return (num(b) / (1024 * 1024)).toFixed(1) + ' MB'; }

const ACTIONS = {
  docLogin:   () => docLogin(),
  docSignOut: () => docSignOut(),
  docSearch:  () => docSearch(),
  pickPatient: el => pickPatient(num(el.dataset.id), el.dataset.tab === 'mri' ? 'mri' : 'symptoms'),
};
document.addEventListener('click', ev => {
  const el = ev.target.closest('[data-action]');
  if (el && Object.prototype.hasOwnProperty.call(ACTIONS, el.dataset.action)) ACTIONS[el.dataset.action](el, ev);
});

window.addEventListener('DOMContentLoaded', async () => {
  document.getElementById('docPass').addEventListener('keydown', e => { if (e.key === 'Enter') docLogin(); });
  document.getElementById('patientQuery').addEventListener('keydown', e => { if (e.key === 'Enter') docSearch(); });
  try {
    const res = await authFetch('/auth/me');
    if (res.ok) { const me = await res.json(); if (me.role === 'doctor') showPortal(me); }
  } catch { /* stay on login */ }
});

async function docLogin() {
  const user = document.getElementById('docUser').value.trim();
  const pass = document.getElementById('docPass').value;
  const err = document.getElementById('authError');
  err.style.display = 'none';
  if (!user || !pass) { err.textContent = 'Fill in all fields'; err.style.display = 'block'; return; }
  const form = new URLSearchParams(); form.append('username', user); form.append('password', pass);
  try {
    const res = await fetch(`${API}/auth/login`, {
      method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      credentials: 'same-origin', body: form,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(errorText(data, 'Login failed'));
    if (data.role !== 'doctor') {
      await fetch(`${API}/auth/logout`, { method: 'POST', credentials: 'same-origin' });
      throw new Error('This portal is for approved doctors only.');
    }
    document.getElementById('docPass').value = '';
    showPortal(await (await authFetch('/auth/me')).json());
  } catch (e) { err.textContent = e.message; err.style.display = 'block'; }
}

async function docSignOut() {
  try { await fetch(`${API}/auth/logout`, { method: 'POST', credentials: 'same-origin' }); } catch {}
  document.getElementById('adminShell').classList.remove('visible');
  document.getElementById('authOverlay').style.display = 'flex';
  document.getElementById('patientList').innerHTML = '';
  patients = []; assigned = []; selected = null;
  document.getElementById('assignedList').innerHTML = '';
}

function showPortal(me) {
  document.getElementById('authOverlay').style.display = 'none';
  document.getElementById('adminShell').classList.add('visible');
  document.getElementById('docWho').textContent = 'Dr. ' + me.username;
  loadAssigned();
}

const patientItem = p => `
  <div class="user-item ${p.id === selected ? 'active' : ''}" data-action="pickPatient" data-id="${num(p.id)}">
    <div class="user-item-name">${esc(p.email)}${p.latest_tier ? `<span class="conclusion-tier tier-${cls(p.latest_tier)}" style="font-size:9px;padding:1px 6px;margin-left:4px;">${esc(String(p.latest_tier).replace('_', ' '))}</span>` : ''}</div>
    <div class="user-item-meta">${esc(p.username)} · ${num(p.session_count)} assessments · ${num(p.scan_count)} MRI</div>
  </div>`;

async function loadAssigned() {
  const list = document.getElementById('assignedList');
  try {
    const res = await authFetch('/doctor/my-patients');
    if (res.status === 401 || res.status === 403) { docSignOut(); return; }
    assigned = (await res.json()).patients || [];
    document.getElementById('assignedCount').textContent = assigned.length;
    list.innerHTML = assigned.length ? assigned.map(patientItem).join('')
      : '<div style="padding:12px 16px;font-size:13px;color:var(--muted)">No patients assigned yet</div>';
  } catch { list.innerHTML = ''; }
}

async function docSearch() {
  const q = document.getElementById('patientQuery').value.trim();
  const list = document.getElementById('patientList');
  list.innerHTML = '<div class="loading"><div class="spin"></div> Searching…</div>';
  try {
    const res = await authFetch('/doctor/patients?email=' + encodeURIComponent(q));
    const data = await res.json().catch(() => ({}));
    if (res.status === 401 || res.status === 403) { docSignOut(); return; }
    if (!res.ok) { list.innerHTML = `<div style="padding:16px;font-size:13px;color:var(--red)">${esc(errorText(data, 'Search failed'))}</div>`; return; }
    patients = data.patients || [];
    list.innerHTML = patients.length ? patients.map(patientItem).join('') : '<div style="padding:16px;font-size:13px;color:var(--muted)">No matching patients</div>';
  } catch (e) { list.innerHTML = `<div style="padding:16px;color:var(--red)">${esc(e.message)}</div>`; }
}

async function pickPatient(id, tab) {
  selected = id;
  document.querySelectorAll('.user-item').forEach(el => el.classList.toggle('active', num(el.dataset.id) === id));
  const p = patients.find(x => x.id === id) || assigned.find(x => x.id === id);
  const main = document.getElementById('mainPanel');
  main.innerHTML = '<div class="loading"><div class="spin"></div> Loading…</div>';
  try {
    const path = tab === 'mri' ? 'mri' : 'symptoms';
    const res = await authFetch(`/doctor/patients/${id}/${path}`);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(errorText(data, 'Could not load'));
    const head = `<div class="user-header"><div class="user-avatar-lg">${esc(((p && p.email) || '?')[0].toUpperCase())}</div>
        <div class="user-header-info"><h2>${esc(p ? p.email : 'Patient')}</h2><p>${esc(p ? p.username : '')} · Joined ${esc(fmtDate((p || {}).created_at))}</p></div></div>
      <div class="tabs">
        <button class="export-btn ${tab === 'symptoms' ? 'active' : ''}" data-action="pickPatient" data-id="${num(id)}" data-tab="symptoms">Symptoms</button>
        <button class="export-btn ${tab === 'mri' ? 'active' : ''}" data-action="pickPatient" data-id="${num(id)}" data-tab="mri">MRI scans</button>
      </div>`;
    main.innerHTML = head + (tab === 'mri' ? renderMri(data.scans || []) : renderSymptoms(data.sessions || []));
  } catch (e) { main.innerHTML = `<div style="color:var(--red);padding:20px">${esc(e.message)}</div>`; }
}

const chips = arr => (arr && arr.length)
  ? `<div class="chips">${arr.map(x => `<span class="conclusion-chip">${esc(typeof x === 'string' ? x : JSON.stringify(x))}</span>`).join('')}</div>`
  : '<span style="color:var(--muted);font-size:13px">none recorded</span>';

function renderSymptoms(sessions) {
  if (!sessions.length) return '<div style="color:var(--muted);padding:20px 0">This patient has no assessments yet.</div>';
  return sessions.map(s => {
    const tier = s.concluded_tier || s.current_tier || '';
    const mri = (s.mri_results || []).map(m => `<div style="font-size:13px;margin-top:4px">${esc(fmtDate(m.created_at))} · ${esc(m.source)} · lesions: ${m.lesion_count ?? '—'} · volume: ${m.lesion_volume_mm3 != null ? esc(Number(m.lesion_volume_mm3).toFixed(0)) + ' mm³' : '—'} · DIS ${m.dis_met ? 'met' : 'not met'} · DIT ${m.dit_met ? 'met' : 'not met'}</div>`).join('');
    return `<div class="sess-block">
      <h4>${esc(fmtDate(s.created_at))} <span class="conclusion-tier tier-${cls(tier)}" style="margin-left:8px;font-size:11px;">${esc(s.concluded_label || tier)}</span></h4>
      <div class="kv">Detected symptoms</div>${chips(s.symptoms)}
      <div class="kv">Clinical features</div>${chips(s.features)}
      <div class="kv">CNS regions involved (DIS)</div>${chips(s.dis_regions)}
      <div class="kv">Prior episodes reported (DIT)</div><div style="font-size:13px">${num(s.dit_episodes)}</div>
      ${s.tier_log && s.tier_log.length ? `<div class="kv">Risk trajectory</div>${chips(s.tier_log.map(t => 'turn ' + num(t.turn) + ': ' + t.tier))}` : ''}
      ${mri ? `<div class="kv">MRI analysis</div>${mri}` : ''}
    </div>`;
  }).join('');
}

function renderMri(scans) {
  if (!scans.length) return '<div style="color:var(--muted);padding:20px 0">This patient has not submitted any MRI scans.</div>';
  return `<table class="doc-table"><thead><tr><th>Submitted</th><th>Result</th><th>Size</th><th></th></tr></thead><tbody>${scans.map(s => `<tr>
      <td>${esc(fmtDate(s.created_at))}</td>
      <td>${s.tier ? `<span class="conclusion-tier tier-${cls(s.tier)}" style="font-size:11px;">${esc(String(s.tier).replace('_', ' '))}</span>` : '—'}
          <span style="font-size:12px;color:var(--muted)"> lesions ${s.lesion_count ?? '—'}</span></td>
      <td>${esc(fmtMB(s.size_bytes))}</td>
      <td><a class="dl-btn" href="/doctor/mri/${num(s.id)}/download" download>⬇ DICOM (.zip)</a></td></tr>`).join('')}</tbody></table>
    <p class="hint">Downloads are DICOM series converted from the patient's uploaded NIfTI. Each download is logged.</p>`;
}
