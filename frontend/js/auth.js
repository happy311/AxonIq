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
  const fl = document.getElementById('forgotLink');
  if (fl) fl.style.display = tab === 'login' ? 'block' : 'none';
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
    let res;
    if (currentTab === 'register') {
      res = await fetch(`${API}/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, email, password }),
      });
    } else {
      const form = new URLSearchParams();
      form.append('username', email);
      form.append('password', password);
      res = await fetch(`${API}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: form,
      });
    }

    let data;
    try { data = await res.json(); } catch { throw new Error(`Server error ${res.status}`); }

    if (!res.ok) {
      let errMsg = 'Authentication failed';
      if (data.detail) {
        errMsg = typeof data.detail === 'string'
          ? data.detail
          : Array.isArray(data.detail)
            ? data.detail.map(e => e.msg || JSON.stringify(e)).join(', ')
            : JSON.stringify(data.detail);
      }
      throw new Error(errMsg);
    }

    authToken   = data.access_token;
    currentUser = { id: data.user_id, username: data.username };
    localStorage.setItem('nc_token', authToken);
    localStorage.setItem('nc_user', JSON.stringify(currentUser));
    showApp();

  } catch (err) {
    showAuthError(err.message);
  } finally {
    btn.disabled = false;
  }
}

function signOut() {
  localStorage.removeItem('nc_token');
  localStorage.removeItem('nc_user');
  authToken   = null;
  currentUser = null;
  sessionId   = null;
  document.getElementById('appShell').classList.remove('visible');
  document.getElementById('authOverlay').style.display = 'flex';
  clearMessages();
}
