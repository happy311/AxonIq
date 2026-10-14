/**
 * AxonIQ — event wiring (CSP-safe: no inline handlers anywhere).
 * Markup carries data-action / data-change (+ optional data-arg); this file maps those names
 * to functions through an explicit ALLOW-LIST — attribute values are never used as code.
 */
const CLICK_ACTIONS = {
  switchTab:            el => switchTab(el.dataset.arg),
  submitAuth:           () => submitAuth(),
  showForgotPassword:   () => showForgotPassword(),
  hideForgotPassword:   () => hideForgotPassword(),
  toggleSidebar:        () => toggleSidebar(),
  toggleUserMenu:       () => toggleUserMenu(),
  newChat:              () => newChat(),
  showChangePassword:   () => showChangePassword(),
  hideChangePassword:   () => hideChangePassword(),
  submitChangePassword: () => submitChangePassword(),
  signOut:              () => signOut(),
  sendChip:             el => sendChip(el.dataset.arg),
  sendMessage:          () => sendMessage(),
  closeMriPanel:        () => closeMriPanel(),
  toggleMriPanel:       () => toggleMriPanel(),
  pickFlair:            () => document.getElementById('flairFileInput').click(),
  submitMriPair:        () => submitMriPair(),
  submitMriText:        () => submitMriText(),
  fpSendOTP:            () => fpSendOTP(),
  fpVerifyOTP:          () => fpVerifyOTP(),
  fpSetPassword:        () => fpSetPassword(),
  exportMyData:         () => exportMyData(),
  deleteMyAccount:      () => deleteMyAccount(),
};
const CHANGE_ACTIONS = {
  handleMriSlot: (el) => handleMriSlot(el, el.dataset.arg),
  roleChanged:   () => updateRoleFields(),
};

document.addEventListener('click', ev => {
  const el = ev.target.closest('[data-action]');
  const fn = el && Object.prototype.hasOwnProperty.call(CLICK_ACTIONS, el.dataset.action)
    ? CLICK_ACTIONS[el.dataset.action] : null;
  if (fn) fn(el, ev);
});
document.addEventListener('change', ev => {
  const el = ev.target.closest('[data-change]');
  const fn = el && Object.prototype.hasOwnProperty.call(CHANGE_ACTIONS, el.dataset.change)
    ? CHANGE_ACTIONS[el.dataset.change] : null;
  if (fn) fn(el, ev);
});
