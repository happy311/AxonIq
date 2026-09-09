/**
 * AxonIQ — Shared Utilities
 * Pure helper functions — no DOM access, no side effects.
 */

function escapeHtml(t) {
  return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
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

function authFetch(path, opts = {}) {
  return fetch(`${API}${path}`, {
    ...opts,
    headers: { ...(opts.headers || {}), 'Authorization': `Bearer ${authToken}` },
  });
}
