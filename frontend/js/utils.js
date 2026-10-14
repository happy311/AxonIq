/**
 * AxonIQ — Shared Utilities
 * Pure helper functions — no DOM access, no side effects.
 */

function escapeHtml(t) {
  return String(t ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
                        .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

function formatTimestamp(iso) {
  if (!iso) return '';
  try {
    const d = new Date(iso), now = new Date();
    const sameDay = d.toDateString() === now.toDateString();
    return sameDay
      ? d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      : d.toLocaleDateString([], { month: 'short', day: 'numeric' })
        + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } catch { return ''; }
}

function formatMessage(text) {
  let html = escapeHtml(text);
  // Markdown bold
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  // Paragraphs
  html = html.replace(/\n\n/g, '</p><p>');
  html = html.replace(/\n/g, '<br>');
  // Highlight key clinical terms
  ['MRI','lumbar puncture','oligoclonal bands','optic neuritis',
   'Lhermitte','Uhthoff','neurology referral'
  ].forEach(kw => {
    const re = new RegExp('(?<![\\w>])' + kw.replace(/[.*+?^${}()|[\]\\]/g,'\\$&') + '(?![\\w<])', 'gi');
    html = html.replace(re, '<strong>$&</strong>');
  });
  return `<p>${html}</p>`;
}

function getCookie(name) {
  const m = document.cookie.split('; ').find(c => c.startsWith(name + '='));
  return m ? decodeURIComponent(m.split('=')[1]) : '';
}

// Cookie-authenticated fetch: same-origin credentials + CSRF double-submit header on
// state-changing requests. Never put tokens in JS-visible storage.
function authFetch(path, opts = {}) {
  const method  = (opts.method || 'GET').toUpperCase();
  const headers = { ...(opts.headers || {}) };
  if (!['GET', 'HEAD', 'OPTIONS'].includes(method)) headers['X-CSRF-Token'] = getCookie('nc_csrf');
  return fetch(`${API}${path}`, { ...opts, headers, credentials: 'same-origin' });
}

// FastAPI error payloads → one readable string (detail may be a string or a 422 list)
function errorText(data, fallback) {
  const d = data && data.detail;
  if (!d) return fallback;
  if (typeof d === 'string') return d;
  if (Array.isArray(d)) return d.map(e => e.msg || 'Invalid value').join(', ');
  return fallback;
}
