/**
 * AxonIQ — App State
 * Single source of truth for ALL shared state.
 * No module should store its own state — always read/write here.
 */
const API = window.location.origin;

// Auth
let authToken   = localStorage.getItem('nc_token') || null;
let currentUser = JSON.parse(localStorage.getItem('nc_user') || 'null');

// Session
let sessionId   = null;
let isWaiting   = false;
let currentTab  = 'login';

// MRI state — mirrors backend AgentState
let mriRequested  = false;   // agent asked user for MRI this session
let mriSubmitted  = false;   // user has already submitted MRI text
let mriAnalysed   = false;   // backend confirmed MRI was analysed
