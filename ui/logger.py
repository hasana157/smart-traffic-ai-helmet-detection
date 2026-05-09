import pandas as pd
import os
from datetime import datetime


def save_log(data: dict, path: str) -> None:
    """Append a violation record to a CSV log file, creating it if needed."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = pd.DataFrame([data])
        file_exists = os.path.isfile(path)
        df.to_csv(
            path,
            mode="a",
            header=not file_exists,
            index=False,
            encoding="utf-8",
        )
    except Exception as e:
        print(f"[Logger] Failed to save log: {e}")


def is_duplicate(new_v: dict, v_list: list, threshold: int = 3) -> bool:
    """
    Return True if the new violation is too similar to a recent one.

    Checks:
    - Same violation type
    - Within `threshold` seconds of the last entry of the same type
    """
    if not v_list:
        return False

    same_type = [v for v in v_list if v.get("Type") == new_v.get("Type")]
    if not same_type:
        return False

    last = same_type[-1]
    try:
        t1 = datetime.strptime(new_v["Time"], "%H:%M:%S")
        t2 = datetime.strptime(last["Time"], "%H:%M:%S")
        return abs((t1 - t2).total_seconds()) < threshold
    except (ValueError, KeyError):
        return False


def load_log(path: str) -> pd.DataFrame:
    """Load a violation CSV log. Returns empty DataFrame on missing/corrupt file."""
    if not os.path.isfile(path):
        return pd.DataFrame(columns=["Time", "Type", "Confidence", "Detail"])
    try:
        return pd.read_csv(path, encoding="utf-8")
    except Exception as e:
        print(f"[Logger] Failed to load log: {e}")
        return pd.DataFrame(columns=["Time", "Type", "Confidence", "Detail"])
