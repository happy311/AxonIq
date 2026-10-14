/**
 * AxonIQ — App State
 * Single source of truth for ALL shared state.
 * No module should store its own state — always read/write here.
 */
const API = window.location.origin;

// Auth
// The session token lives in an HttpOnly cookie (JS can never read it → XSS cannot steal it).
// Only non-sensitive display info is kept here, in memory; it is re-fetched via /auth/me on load.
let currentUser = null;

// Session
let sessionId   = null;
let isWaiting   = false;
let currentTab  = 'login';

// MRI state — mirrors backend AgentState
let mriRequested  = false;   // agent asked user for MRI this session
let mriSubmitted  = false;   // user has already submitted MRI text
let mriAnalysed   = false;   // backend confirmed MRI was analysed
