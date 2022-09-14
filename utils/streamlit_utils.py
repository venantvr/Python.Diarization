from streamlit.scriptrunner import get_script_run_ctx as get_report_ctx


def get_session() -> str:
    """
    Function that returns a unique identifier per user session
    """

    session_id = get_report_ctx().session_id.replace('-', '_')

    return session_id
