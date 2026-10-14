/**
 * AxonIQ — Auth Module
 * Handles: login, register, sign out, tab switching.
 * On success calls showApp() from shell.js.
 */

function switchTab(tab) {
  currentTab = tab;
  document.getElementById('tabLogin').classList.toggle('active', tab === 'login');
  document.getElementById('tabRegister').classList.toggle('active', tab === 'register');
  document.getElementById('authSubmitBtn').textContent = tab === 'login' ? 'Sign In' : 'Create Account';
  document.querySelectorAll('.auth-register-only').forEach(el =>
    el.style.display = tab === 'register' ? 'block' : 'none'
  );
  document.getElementById('emailLabel').textContent = tab === 'login' ? 'Email or Username' : 'Email';
  clearAuthError();
  updateRoleFields();
  const fl = document.getElementById('forgotLink');
  if (fl) fl.style.display = tab === 'login' ? 'block' : 'none';
}

function updateRoleFields() {
  const isDoc = currentTab === 'register' && document.getElementById('authRole').value === 'doctor';
  document.querySelectorAll('.auth-doctor-only').forEach(el => el.style.display = isDoc ? 'block' : 'none');
}

function showAuthError(msg) {
  const e = document.getElementById('authError');
  e.textContent = msg;
  e.style.display = 'block';
}

function clearAuthError() {
  document.getElementById('authError').style.display = 'none';
}

async function submitAuth() {
  clearAuthError();
  const btn      = document.getElementById('authSubmitBtn');
  btn.disabled   = true;
  const email    = document.getElementById('authEmail').value.trim();
  const password = document.getElementById('authPassword').value;
  const username = document.getElementById('authUsername').value.trim();

  if (!email || !password) { showAuthError('Please fill in all fields.'); btn.disabled = false; return; }
  if (currentTab === 'register' && !username) { showAuthError('Please enter a username.'); btn.disabled = false; return; }

  try {
    if (currentTab === 'register') {
      if (!document.getElementById('authPrivacy').checked) {
        showAuthError('Please accept the privacy notice to create an account.');
        return;
      }
      const role = document.getElementById('authRole').value;
      const payload = { username, email, password, accept_privacy: true, role };
      if (role === 'doctor') {
        payload.license_number = document.getElementById('authLicense').value.trim();
        payload.institution    = document.getElementById('authInstitution').value.trim();
        if (!payload.license_number || !payload.institution) {
          showAuthError('Doctors must enter a licence number and hospital/clinic.');
          return;
        }
      }
      const reg = await fetch(`${API}/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify(payload),
      });
      if (!reg.ok) {
        const d = await reg.json().catch(() => ({}));
        throw new Error(errorText(d, `Server error ${reg.status}`));
      }
      if (role === 'doctor') {
        // Doctors cannot sign in until an administrator approves them.
        switchTab('login');
        document.getElementById('authPassword').value = '';
        showAuthError('Doctor registration received. An administrator must verify and approve your account before you can sign in.');
        return;
      }
      // The server answers identically whether or not the account already existed, so we
      // sign in with the credentials just typed; if that fails, say so without guessing why.
      const ok = await _login(username, password);
      if (!ok) throw new Error('Registration could not be completed. If you already have an account, sign in or use "Forgot password".');
      return;
    }
    const ok = await _login(email, password);
    if (!ok) throw new Error('Incorrect username/email or password');
  } catch (err) {
    showAuthError(err.message);
  } finally {
    btn.disabled = false;
  }
}

// Returns true on success; throws on rate-limit/lockout so the message is shown.
async function _login(identifier, password) {
  const form = new URLSearchParams();
  form.append('username', identifier);
  form.append('password', password);
  const res = await fetch(`${API}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    credentials: 'same-origin',
    body: form,
  });
  const data = await res.json().catch(() => ({}));
  if (res.status === 429) throw new Error(errorText(data, 'Too many attempts — please wait and try again.'));
  if (res.status === 403 && typeof data.detail === 'string' && data.detail !== 'PASSWORD_CHANGE_REQUIRED') {
    throw new Error(data.detail);                       // e.g. doctor awaiting approval
  }
  if (!res.ok) return false;
  if (data.role === 'doctor') { window.location.href = '/doctor'; return true; }
  if (data.role === 'admin')  { window.location.href = '/admin';  return true; }
  currentUser = { id: data.user_id, username: data.username };
  document.getElementById('authPassword').value = '';
  showApp();
  return true;
}

async function signOut() {
  try { await fetch(`${API}/auth/logout`, { method: 'POST', credentials: 'same-origin' }); } catch {}
  currentUser = null;
  sessionId   = null;
  document.getElementById('appShell').classList.remove('visible');
  document.getElementById('authOverlay').style.display = 'flex';
  clearMessages();
}
