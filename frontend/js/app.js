/**
 * AxonIQ — App Bootstrap
 * Entry point: wires all modules together after the DOM is ready.
 * Load order of JS files matters — this file must be last.
 *
 * Load order:
 *   state.js → utils.js → auth.js → shell.js →
 *   chat.js → risk-panel.js → modals.js → app.js
 */
window.addEventListener('DOMContentLoaded', () => {

  // ── Auth: Enter key on password field ──────────────────────────────────────
  const authPw = document.getElementById('authPassword');
  if (authPw) authPw.addEventListener('keydown', e => {
    if (e.key === 'Enter') submitAuth();
  });

  // ── Auth: Enter key on email field (for quick login) ───────────────────────
  const authEmail = document.getElementById('authEmail');
  if (authEmail) authEmail.addEventListener('keydown', e => {
    if (e.key === 'Enter') submitAuth();
  });

  // ── Chat input init (resize + Enter to send) ────────────────────────────────
  initChatInput();

  // ── Close user dropdown on outside click ────────────────────────────────────
  document.addEventListener('click', e => {
    const dropdown = document.getElementById('userDropdown');
    if (dropdown && !e.target.closest('.user-pill')) {
      dropdown.classList.remove('open');
    }
  });

  // ── Change Password modal: click-outside to close + Enter on last field ─────
  const cpOverlay = document.getElementById('changePwOverlay');
  if (cpOverlay) {
    cpOverlay.addEventListener('click', e => {
      if (e.target === cpOverlay) hideChangePassword();
    });
    const pwConfirm = document.getElementById('pwConfirm');
    if (pwConfirm) pwConfirm.addEventListener('keydown', e => {
      if (e.key === 'Enter') submitChangePassword();
    });
  }

  // ── Forgot Password modal: click-outside to close + Enter shortcuts ──────────
  const fpOverlay = document.getElementById('forgotOverlay');
  if (fpOverlay) {
    fpOverlay.addEventListener('click', e => {
      if (e.target === fpOverlay) hideForgotPassword();
    });
    const fpOTP = document.getElementById('fpOTP');
    if (fpOTP) fpOTP.addEventListener('keydown', e => {
      if (e.key === 'Enter') fpVerifyOTP();
    });
    const fpConfirmPw = document.getElementById('fpConfirmPw');
    if (fpConfirmPw) fpConfirmPw.addEventListener('keydown', e => {
      if (e.key === 'Enter') fpSetPassword();
    });
  }

  // ── Auto-login if valid token + user exist in localStorage ──────────────────
  if (authToken && currentUser) {
    showApp();
  }
  // Otherwise the auth overlay is already visible (default state in HTML)
});
