# AxonIQ — security notes & accepted trade-offs

**Sessions.** Browser sessions use an `HttpOnly; Secure; SameSite=Strict` cookie (`nc_session`) plus a CSRF
double-submit token (`nc_csrf` cookie ↔ `X-CSRF-Token` header) on every state-changing request made with
the cookie. `Authorization: Bearer` is still accepted for scripts/tests (not CSRF-able, skips the CSRF check).
JS never sees the session token, so an XSS bug cannot steal it (it could still *act* as the user while the
page is open — hence the strict CSP). Access tokens last `JWT_EXPIRE_MINUTES` (default 480, hard max 1440).

**CSP.** `script-src 'self'` — no inline script, no eval. `style-src 'self'` + Google Fonts CSS host; inline
`style="…"` *attributes* are allowed through `style-src-attr 'unsafe-inline'` because the UI sets many
(inline `<style>` blocks are blocked). Removing that last exception means moving every `style=` into CSS
classes. Browsers older than ~2022 that don't know `style-src-attr` will drop inline style attributes.

**Rate limits / lockout.** Per-IP limits use slowapi; per-username/email/user limits are in-memory sliding
windows — correct for the required single worker, reset on restart. Login lockout (exponential, 1→60 min after
5 consecutive failures) is stored in the DB and applies identically to non-existent usernames. Trade-off: an
attacker who knows a username can keep that account locked (denial of service); the cap is 60 min and the
password-reset flow still works.

**Reverse proxy.** The client IP comes from `request.client`, which uvicorn rewrites from `X-Forwarded-For`
only when the TCP peer matches `--forwarded-allow-ips`. Never set that to `*`.

**Encryption at rest.** `DATA_ENCRYPTION_KEY` (Fernet) protects `messages.content`, `chat_sessions.title`,
`mri_results.findings_json` and the nightly DB backup. Not column-encrypted: usernames, e-mails, timestamps,
symptom/feature metadata, tier logs → use an encrypted volume (see README_DEPLOY_AWS.md). Key loss = data loss.

**Admin.** The administrator is seeded from `ADMIN_USERNAME`/`ADMIN_PASSWORD` at startup (`api/core/bootstrap.py`),
*before* the server accepts requests. It never promotes an existing registered account (startup aborts instead), so
registering a free username can't confer admin. `python make_admin.py <user>` remains as an explicit alternative.
The env password is plaintext in your environment/secret store — protect it like `JWT_SECRET`.
`admin_audit` is append-only via SQLite triggers (someone with raw file access could still drop the triggers —
ship the table to your log pipeline if you need tamper-evidence).

**Doctors.** Self-registration as `role=doctor` creates a `pending` account that cannot sign in (login returns 403 only
after the correct password, so nothing leaks to guessers). The administrator permits/removes doctors in `/admin`; removal
bumps `token_version` so live sessions die at once, and every request re-checks `doctor_status`. Doctors can search
patients by e-mail (min 3 chars, patients only), see per-session symptoms/tier/McDonald/MRI summaries (never chat
transcripts), and download scans. Every search/view/download is written to `admin_audit` before data is returned (request
refused if the audit write fails); search text is not stored. Downloads are rate-limited (10/hour/doctor).
Trade-off: any approved doctor can look up any patient — there is no per-patient consent/assignment. Consider adding one.
The doctor-visibility disclosure is part of privacy notice `2026-10b`; patients who registered earlier consented to the old
text — re-consent them before enabling doctors in production.

**Stored MRI scans.** Validated uploads are now kept (chunked Fernet encryption, random filenames, 0600) in `MRI_STORE_DIR`
so doctors can download them; they're deleted with the session / account / retention purge. Without `DATA_ENCRYPTION_KEY`
(dev only) they are stored unencrypted. Downloads are DICOM series *converted from NIfTI* (geometry from the header, no
invented scanner metadata, patient shown only as `AXQ<id>`) — not original scanner DICOM.

**Patient → doctor assignment.** When a patient's MRI is stored, `assign_patient()` (api/database.py) gives them to the approved
doctor with the fewest currently-assigned patients (ties: whoever was assigned longest ago, then lowest id), inside one
`BEGIN IMMEDIATE` transaction so simultaneous uploads can't pick the same doctor. It is sticky (a patient keeps their doctor).
Patients with an MRI but no approved doctor wait and are assigned when a doctor is approved / at startup. Removing a doctor
releases their patients and re-runs the assignment. Doctors see their list under "My assigned patients"; search still reaches
every patient (assignment is routing, not an access restriction).

**Accepted uploads.** `.nii`, `.nii.gz`, or a `.zip` of DICOM slices (the file type is verified from the bytes, not the name).
A DICOM zip is converted to one NIfTI volume in a sandboxed subprocess (`api/mri_engine/dicom_import.py`: RLIMIT memory/CPU/output
size, scrubbed env, nothing extracted to disk, member names never used as paths). Rejected: encrypted zips, >5000 files, members
>128 MB, per-member compression ratio >250×, total unzipped over `MRI_MAX_DECOMPRESSED_BYTES`, non-MR data, <8 slices, duplicate
or missing slice positions, undecodable (compressed-transfer-syntax) pixel data. The largest MR series is used. The converted
NIfTI then goes through the same validator, encrypted storage and analysis as a direct NIfTI upload; the original zip (which
carries patient name/ID) is deleted immediately and never stored. Doctors therefore download a re-exported, de-identified DICOM,
not the original study.
