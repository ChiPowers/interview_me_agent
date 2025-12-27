import os
import sys
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

STREAMLIT_URL = os.environ.get("STREAMLIT_APP_URL", "").strip()
if not STREAMLIT_URL:
    print("ERROR: STREAMLIT_APP_URL env var is required")
    sys.exit(2)

WAKE_BUTTON_XPATH = "//button[contains(., 'Yes, get this app back up')]"

def main():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(options=options)

    try:
        print(f"Opening: {STREAMLIT_URL}")
        driver.get(STREAMLIT_URL)

        wait = WebDriverWait(driver, 25)

        try:
            # If the app is asleep, Streamlit shows a “wake up” button that must be clicked.
            btn = wait.until(EC.element_to_be_clickable((By.XPATH, WAKE_BUTTON_XPATH)))
            print("Wake button found — clicking…")
            btn.click()

            # Confirm it disappears (usually means wake process started)
            wait.until(EC.invisibility_of_element_located((By.XPATH, WAKE_BUTTON_XPATH)))
            print("Wake button disappeared ✅ (app should be spinning up)")
        except TimeoutException:
            # If no button appears, the app is probably already awake
            print("No wake button detected ✅ (app likely already awake)")

    finally:
        driver.quit()

if __name__ == "__main__":
    main()
