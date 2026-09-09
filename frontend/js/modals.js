/**
 * AxonIQ — Modals Module
 * Handles: Change Password modal, Forgot Password (3-step OTP) modal.
 * Both modals use the .modal-overlay / .open CSS class pattern consistently.
 */

// ── Change Password ───────────────────────────────────────────────────────────

function showChangePassword() {
  document.getElementById('userDropdown').classList.remove('open');
  ['pwOld','pwNew','pwConfirm'].forEach(id => document.getElementById(id).value = '');
  document.getElementById('changePwError').style.display   = 'none';
  document.getElementById('changePwSuccess').style.display = 'none';
  document.getElementById('changePwOverlay').classList.add('open');
  document.getElementById('pwOld').focus();
}

function hideChangePassword() {
  document.getElementById('changePwOverlay').classList.remove('open');
}

async function submitChangePassword() {
  const oldPw  = document.getElementById('pwOld').value;
  const newPw  = document.getElementById('pwNew').value;
  const confPw = document.getElementById('pwConfirm').value;
  const errEl  = document.getElementById('changePwError');
  const okEl   = document.getElementById('changePwSuccess');
  errEl.style.display = 'none';
  okEl.style.display  = 'none';

  if (!oldPw || !newPw || !confPw) {
    errEl.textContent = 'Please fill in all fields.';
    errEl.style.display = 'block'; return;
  }
  if (newPw.length < 6) {
    errEl.textContent = 'New password must be at least 6 characters.';
    errEl.style.display = 'block'; return;
  }
  if (newPw !== confPw) {
    errEl.textContent = 'Passwords do not match.';
    errEl.style.display = 'block'; return;
  }

  try {
    const res  = await authFetch('/auth/change-password', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ old_password: oldPw, new_password: newPw }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Failed to change password.');
    okEl.textContent   = '✅ Password changed successfully!';
    okEl.style.display = 'block';
    setTimeout(hideChangePassword, 2000);
  } catch (e) {
    errEl.textContent   = e.message;
    errEl.style.display = 'block';
  }
}

// ── Forgot Password (3-step OTP flow) ────────────────────────────────────────

let _fpResetToken = '';

function showForgotPassword() {
  // Reset all fields
  ['fpEmail','fpOTP','fpNewPw','fpConfirmPw'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.value = '';
  });
  document.getElementById('fpError').style.display   = 'none';
  document.getElementById('fpSuccess').style.display = 'none';
  _fpGoToStep(1);
  document.getElementById('forgotOverlay').classList.add('open');
  document.getElementById('fpEmail').focus();
}

function hideForgotPassword() {
  document.getElementById('forgotOverlay').classList.remove('open');
}

function _fpGoToStep(step) {
  [1, 2, 3].forEach(n => {
    document.getElementById(`fpStep${n}`).style.display = (n === step) ? 'block' : 'none';
  });
  const titles = ['Reset Password', 'Enter Reset Code', 'Set New Password'];
  document.getElementById('fpTitle').textContent    = titles[step - 1];
  document.getElementById('fpSubtitle').textContent = [
    'Enter your registered email',
    'Check your inbox for the code',
    'Choose a new password',
  ][step - 1];
}

function _fpErr(msg) {
  const e = document.getElementById('fpError');
  e.textContent = msg;
  e.style.display = 'block';
  document.getElementById('fpSuccess').style.display = 'none';
}

function _fpOk(msg) {
  const s = document.getElementById('fpSuccess');
  s.textContent = msg;
  s.style.display = 'block';
  document.getElementById('fpError').style.display = 'none';
}

async function fpSendOTP() {
  const email = document.getElementById('fpEmail').value.trim();
  if (!email) { _fpErr('Please enter your email address.'); return; }
  document.getElementById('fpError').style.display = 'none';

  const btn = document.querySelector('#fpStep1 .auth-btn');
  const origText = btn.textContent;
  btn.textContent = 'Sending…';
  btn.disabled = true;

  try {
    const res = await fetch(`${API}/auth/forgot-password`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ email }),
    });
    const data = await res.json().catch(() => ({}));

    if (!res.ok) { _fpErr(data.detail || 'Server error — please try again.'); return; }
    if (data.dev_mode && data.dev_error) { _fpErr(data.dev_error); return; }

    document.getElementById('fpEmailShow').textContent = email;
    _fpGoToStep(2);

    if (data.dev_mode && data.dev_otp) {
      document.getElementById('fpOTP').value = data.dev_otp;
      _fpOk('Dev mode — code: ' + data.dev_otp);
    } else {
      _fpOk('Code sent! Check your inbox and spam folder.');
    }
    document.getElementById('fpOTP').focus();

  } catch {
    _fpErr('Network error — please check your connection and try again.');
  } finally {
    btn.textContent = origText;
    btn.disabled = false;
  }
}

async function fpVerifyOTP() {
  const email = document.getElementById('fpEmail').value.trim();
  const otp   = document.getElementById('fpOTP').value.trim();
  if (!otp || otp.length < 6) { _fpErr('Please enter the 6-digit code.'); return; }

  try {
    const res  = await fetch(`${API}/auth/verify-otp`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ email, otp }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Invalid or expired code.');
    _fpResetToken = data.reset_token;
    _fpGoToStep(3);
    document.getElementById('fpSuccess').style.display = 'none';
    document.getElementById('fpNewPw').focus();
  } catch (e) { _fpErr(e.message); }
}

async function fpSetPassword() {
  const newPw  = document.getElementById('fpNewPw').value;
  const confPw = document.getElementById('fpConfirmPw').value;
  if (!newPw || newPw.length < 6) { _fpErr('Password must be at least 6 characters.'); return; }
  if (newPw !== confPw) { _fpErr('Passwords do not match.'); return; }

  try {
    const res  = await fetch(`${API}/auth/reset-password`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ reset_token: _fpResetToken, new_password: newPw }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Reset failed — the link may have expired.');
    _fpOk('Password reset successfully! Redirecting to login…');
    setTimeout(() => { hideForgotPassword(); switchTab('login'); }, 2000);
  } catch (e) { _fpErr(e.message); }
}
