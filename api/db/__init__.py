"""
AxonIQ — Database Layer (Repository Pattern)

Single source of truth: api/database.py contains all SQLite logic.
This package re-exports everything from database.py under clean domain names,
making it easy to migrate to per-domain files in future without changing importers.

Usage:
    from api.db import save_message, get_messages, create_user ...
    (identical to importing from api.database directly)
"""
from api.database import (
    # Infrastructure
    init_db,
    migrate_add_admin_column,
    migrate_add_logs_table,
    migrate_add_otp_table,
    get_conn,
    DB_PATH,

    # Users
    create_user,
    get_user_by_id,
    get_user_by_username,
    get_user_by_email,
    set_admin,
    update_user_password,

    # Sessions
    create_db_session as create_session,
    get_user_sessions,
    update_session_title,
    touch_session,
    delete_db_session as delete_session,
    get_session_owner,
    save_conclusion,
    get_conclusion,
    migrate_add_session_state,
    migrate_add_clinical_state,
    migrate_add_tier_log,
    migrate_add_nifti_queue,
    get_session_state,
    update_session_state,
    get_tier_log,
    get_session_export_data,
    get_all_sessions_admin,
    get_all_users_admin,
    get_user_sessions_with_messages,

    # NIfTI queue (DB-backed, multi-worker safe)
    store_nifti_path,
    pop_nifti_path,
    store_nifti_paths,
    pop_nifti_paths,
    has_queued_nifti_paths,

    # MRI background job tracking (async upload → poll → result)
    migrate_add_mri_jobs,
    migrate_add_mri_case_id,
    create_mri_job,
    set_mri_job_status,
    set_mri_job_case_id,
    get_mri_job,

    # Messages
    save_message,
    get_messages,
    get_message_count,

    # Logs
    log_action,
    get_user_logs,
    get_all_logs,

    # OTP
    save_otp,
    get_valid_otp,
    mark_otp_used,
)

__all__ = [
    # Infrastructure
    "init_db", "migrate_add_admin_column", "migrate_add_logs_table",
    "migrate_add_otp_table", "get_conn", "DB_PATH",
    # Users
    "create_user", "get_user_by_id", "get_user_by_username",
    "get_user_by_email", "set_admin", "update_user_password",
    # Sessions
    "create_session", "get_user_sessions", "update_session_title",
    "touch_session", "delete_session", "get_session_owner",
    "save_conclusion", "get_conclusion",
    "migrate_add_session_state", "migrate_add_clinical_state",
    "migrate_add_tier_log", "migrate_add_nifti_queue",
    "get_session_state", "update_session_state",
    "get_tier_log", "get_session_export_data",
    "get_all_sessions_admin", "get_all_users_admin", "get_user_sessions_with_messages",
    # NIfTI queue
    "store_nifti_path", "pop_nifti_path",
    "store_nifti_paths", "pop_nifti_paths", "has_queued_nifti_paths",
    # MRI background job tracking
    "migrate_add_mri_jobs", "migrate_add_mri_case_id",
    "create_mri_job", "set_mri_job_status", "set_mri_job_case_id", "get_mri_job",
    # Messages
    "save_message", "get_messages", "get_message_count",
    # Logs
    "log_action", "get_user_logs", "get_all_logs",
    # OTP
    "save_otp", "get_valid_otp", "mark_otp_used",
]
