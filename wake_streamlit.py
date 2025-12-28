import os
import sys
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

STREAMLIT_URL = os.environ.get("STREAMLIT_APP_URL", "").strip()
if not STREAMLIT_URL:
    print("ERROR: STREAMLIT_APP_URL env var is required")
    sys.exit(2)

# Text expected when the real app is rendered (h1 by default, override via APP_SENTINEL_TEXT)
APP_SENTINEL_TEXT = os.environ.get("APP_SENTINEL_TEXT", "Interview Chivon Powers").strip().lower()

# The wake button text Streamlit shows when asleep
WAKE_BUTTON_XPATH = "//button[contains(., 'Yes, get this app back up')]"

# What we consider “awake”: any of these indicates the Streamlit UI rendered.
# Streamlit tends to include one or more of these across versions.
AWAKE_SELECTORS = [
    (By.CSS_SELECTOR, '[data-testid="stAppViewContainer"]'),
    (By.CSS_SELECTOR, '[data-testid="stHeader"]'),
    (By.CSS_SELECTOR, '#root'),
    (By.CSS_SELECTOR, 'div.stApp'),
]

ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", "wake_artifacts")).resolve()


def save_artifacts(driver: webdriver.Chrome, prefix: str) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    png_path = ARTIFACT_DIR / f"{prefix}-{ts}.png"
    html_path = ARTIFACT_DIR / f"{prefix}-{ts}.html"

    try:
        driver.save_screenshot(str(png_path))
        print(f"Saved screenshot: {png_path}")
    except WebDriverException as e:
        print(f"Could not save screenshot: {e}")

    try:
        html_path.write_text(driver.page_source, encoding="utf-8")
        print(f"Saved HTML: {html_path}")
    except Exception as e:
        print(f"Could not save HTML: {e}")


def has_wake_button(driver: webdriver.Chrome) -> bool:
    try:
        return bool(driver.find_elements(By.XPATH, WAKE_BUTTON_XPATH))
    except Exception:
        return False


def wait_for_app_ready(driver: webdriver.Chrome, timeout_s: int) -> bool:
    """
    Wait for Streamlit container, app-specific sentinel text, and absence of the wake button.
    Prevents false positives on the generic sleep page.
    """
    end = time.time() + timeout_s
    while time.time() < end:
        if wait_for_awake(driver, timeout_s=3):
            sentinel_ok = (not APP_SENTINEL_TEXT) or (APP_SENTINEL_TEXT in driver.page_source.lower())
            if sentinel_ok and not has_wake_button(driver):
                return True
        time.sleep(2)
    return False


def wait_for_awake(driver: webdriver.Chrome, timeout_s: int) -> bool:
    """Return True if any 'awake' selector appears within timeout."""
    end = time.time() + timeout_s
    last_err = None
    while time.time() < end:
        for by, sel in AWAKE_SELECTORS:
            try:
                elems = driver.find_elements(by, sel)
                if elems:
                    return True
            except Exception as e:
                last_err = e
        time.sleep(1.5)
    if last_err:
        print(f"Last selector error: {last_err}")
    return False


def main() -> None:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(60)

    try:
        print(f"Opening: {STREAMLIT_URL}")
        driver.get(STREAMLIT_URL)

        # First, see if the app is already awake.
        if wait_for_app_ready(driver, timeout_s=25):
            print("✅ App appears awake (container + sentinel detected).")
            return

        # If not awake yet, try clicking the wake button if present.
        wait = WebDriverWait(driver, 25)
        try:
            btn = wait.until(EC.element_to_be_clickable((By.XPATH, WAKE_BUTTON_XPATH)))
            print("Wake button found — clicking…")
            btn.click()
        except TimeoutException:
            print("No wake button detected. App may be loading slowly or page differs.")
            save_artifacts(driver, prefix="no-wake-button")
            # Still attempt to wait longer for awake state
            if wait_for_awake(driver, timeout_s=90):
                print("✅ App became awake after waiting.")
                return
            print("❌ App did not become awake after waiting.")
            sys.exit(1)

        # After click: give it a moment and then wait for the app to actually render.
        time.sleep(2)

        if wait_for_app_ready(driver, timeout_s=120):
            print("✅ Wake succeeded (container + sentinel detected after click).")
            return

        # If we reach here, click happened but app still didn't render.
        print("❌ Wake click happened, but Streamlit UI never rendered within 120s.")
        save_artifacts(driver, prefix="clicked-but-not-awake")
        sys.exit(1)

    finally:
        driver.quit()


if __name__ == "__main__":
    main()
