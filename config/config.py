# config/config.py
import os

# Default sample/refresh interval in seconds.
# The Streamlit UI can override this with the sidebar slider,
# but this is the fallback / container default.
SAMPLE_INTERVAL = int(os.getenv("SENTRA_SAMPLE_INTERVAL", "2"))

# Default retention helper constants (used for purge buttons)
ONE_HOUR = 60 * 60
ONE_DAY = ONE_HOUR * 24
ONE_WEEK = ONE_DAY * 7

# Where to store the SQLite DB.
# In Docker we mount /data as a volume, so default there is /data/sentra.db.
# On local dev (Windows etc.) /data may not exist, so we fall back to ./sentra.db.
DEFAULT_DB_PATH = os.getenv("SENTRA_DB_PATH", "/data/sentra.db")


def get_db_path() -> str:
    """
    Returns a usable absolute path to the SQLite DB file.
    Ensures parent directory exists.
    """
    candidate = DEFAULT_DB_PATH

    # If /data doesn't exist (like on Windows dev), just drop sentra.db in CWD.
    if candidate.startswith("/data") and not os.path.exists("/data"):
        candidate = os.path.abspath("./sentra.db")

    parent = os.path.dirname(candidate)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

    return candidate
