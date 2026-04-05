import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 1. Your list of 91 trains
trains = ['12673']

# 2. Initialize the browser
options = webdriver.ChromeOptions()
driver = webdriver.Chrome(options=options)

master_data = []

for train in trains:
    print(f"Scraping data for train: {train}...")
    try:
        driver.get("https://enquiry.indianrail.gov.in/mntes/")

        # --- BROWSER INTERACTION ---
        input_box = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "trainNo"))
        )
        input_box.clear()
        input_box.send_keys(train)

        search_btn = driver.find_element(
            By.CSS_SELECTOR, ".svg-inline--fa.fa-search.fa-w-16")
        search_btn.click()

        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CLASS_NAME, "w3-card-2"))
        )

        # Buffer to let older days fully load into the DOM
        time.sleep(3)

        # --- YOUR EXACT EXTRACTING PATTERN (UNTOUCHED) ---
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        station_rows = soup.find_all('div', class_='w3-card-2')

        for row in station_rows:
            # 1. Extract Station Name
            name_div = row.find(
                'div', style=lambda s: s and 'flex:1' in s and 'float:left' in s)

            if name_div:
                station_name = name_div.find('b').text.strip()
            else:
                station_name = "N/A"

            # 2. Extract Delay
            arrival_div = row.find(
                'div', style=lambda s: s and 'width:100px' in s and 'float:left' in s)

            if arrival_div:
                delay_span = arrival_div.find('span', class_='w3-round')
                delay = delay_span.text.strip() if delay_span else "On Time / 0 Min"
            else:
                delay = "N/A"

            master_data.append({
                'Train Number': train,
                'Station Name': station_name,
                'Delay': delay
            })

    except Exception as e:
        print(f"Failed to scrape train {train}. Error: {e}")
        continue

    time.sleep(3)

driver.quit()

# --- THE MAGIC PANDAS FILTERING ---
print("\n--- Organizing Data ---")
df = pd.DataFrame(master_data)

if df.empty:
    print("ALERT: The dataframe is still empty! The website might be blocking the load or the elements aren't showing up.")
else:
    # 1. Identify the 'N/A' divider rows (True/False column)
    is_divider = (df['Station Name'] == 'N/A') & (df['Delay'] == 'N/A')

    # 2. Count the blocks! Every time it hits a divider (True), the Run Block number goes up by 1.
    # Block 0 = Current day (4th April - glitchy layout)
    # Block 1 = 3rd April
    # Block 2 = 2nd April
    # Block 3 = 1st April
    df['Run Block'] = is_divider.cumsum()

    # 3. Filter the dataframe to KEEP ONLY Blocks 1, 2, and 3
    df_cleaned = df[df['Run Block'].isin([1, 2, 3])]

    # 4. Delete the empty "N/A" divider rows themselves so the final sheet is spotless
    df_cleaned = df_cleaned[df_cleaned['Station Name'] != 'N/A']

    # Save to CSV
    df_cleaned.to_csv('train_delays_cleaned.csv', index=False)
    print("Done! Perfectly filtered data saved to train_delays_cleaned.csv")
