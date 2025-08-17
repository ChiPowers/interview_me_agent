import requests
import trafilatura


def fetch_and_clean(url: str, timeout: int = 15) -> str:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
    except Exception:
        return ""
    txt = trafilatura.extract(r.text, include_tables=False, include_formatting=False) or ""
    return txt.strip()