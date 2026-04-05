import csv
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
# Added this to handle the Enter key
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# ==========================================
# 1. SETUP YOUR VARIABLES HERE
# ==========================================
WEBSITE_URL = "https://www.railyatri.in/time-table"
SEARCH_BAR_XPATH = "//input[@id='trainname']"  # Change to actual ID or XPath

# Your full list of 91 trains goes here
train_numbers = ['12673', '12674', '12681', '12682', '12637', '12638', '12631', '12632', '20665', '20666', '12621', '12622', '12839', '12840', '12433', '12434', '12295', '12296', '12841', '12842', '12625', '12626', '12951', '12952', '22301', '22302', '12001', '12002', '12903', '12904', '17057', '17058', '22651', '22652', '12163', '12164', '12679', '12680', '12925', '12926', '12137', '12138', '12509', '12510', '16339',
                 '16340', '16127', '16128', '22601', '22602', '12009', '12010', '12019', '12020', '12049', '12050', '12235', '12236', '12245', '12246', '12259', '12260', '12261', '12262', '12301', '12302', '12305', '12306', '12313', '12314', '12423', '12424', '12953', '12954', '12957', '12958', '22221', '22222', '22225', '22226', '11013', '11014', '11027', '11028', '12367', '12368', '15959', '15960', '12201', '12202']

# ==========================================
# 2. INITIALIZE SELENIUM WEBDRIVER
# ==========================================
options = webdriver.ChromeOptions()
# options.add_argument('--headless') # Uncomment this if you want it to run silently
driver = webdriver.Chrome(options=options)

# ==========================================
# 3. SCRAPING LOGIC
# ==========================================
with open('train_schedules.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Train_Number', 'Day', 'Station_Code', 'Time'])

    for train_no in train_numbers:
        print(f"Scraping schedule for Train: {train_no}...")

        try:
            # Load the main page
            driver.get(WEBSITE_URL)

            # Wait for the search bar to appear
            search_bar = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, SEARCH_BAR_XPATH))
            )

            # Clear it, enter the train number, and press ENTER
            search_bar.clear()
            search_bar.send_keys(train_no)
            search_bar.send_keys(Keys.ENTER)  # Mimics pressing the Enter key

            # CRITICAL: Wait for the dynamic schedule container to load
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located(
                    (By.CLASS_NAME, 'trainstopage_timeline_listing___s1lT'))
            )

            # Grab the fully rendered HTML
            html_content = driver.page_source
            soup = BeautifulSoup(html_content, 'html.parser')

            # Find all blocks that represent a single Day's timeline
            day_blocks = soup.find_all(
                'div', class_='trainstopage_timeline_listing___s1lT')

            if not day_blocks:
                print(f"  -> No schedule data found for {train_no}. Skipping.")
                continue

            for block in day_blocks:
                # Extract the Day
                day_header = block.find(
                    'div', class_='trainstopage_day_count__rWMg4')
                current_day = day_header.text.strip().upper() if day_header else "UNKNOWN"

                # Find all individual station rows within this day
                station_rows = block.find_all('div', class_='css-zoser8')

                for row in station_rows:
                    # Extract the Station Code
                    code_element = row.find(
                        'p', class_='trainstopage_grey_4a__9i_Q8')
                    station_code = code_element.text.strip() if code_element else ""

                    # Extract the Arrival time / START text
                    arrival_element = row.find(
                        'p', class_='trainstopage_fs11___by8V')
                    arrival_text = arrival_element.text.strip() if arrival_element else ""

                    # Apply the START condition rule
                    if arrival_text == "START":
                        departure_element = row.find('div', class_='dly-time')
                        target_time = departure_element.text.strip() if departure_element else ""
                    else:
                        target_time = arrival_text

                    # Write to CSV
                    if station_code and target_time:
                        writer.writerow(
                            [train_no, current_day, station_code, target_time])

            print(f"  -> Success!")

        except TimeoutException:
            print(f"  -> Timeout: Could not load data for train {train_no}.")
        except NoSuchElementException:
            print(
                f"  -> Error: Search elements not found on the page for train {train_no}.")
        except Exception as e:
            print(f"  -> An unexpected error occurred: {e}")

        # Short sleep to avoid hammering their server
        time.sleep(3)

# Close the browser when done
driver.quit()
print("\nScraping complete! Data saved to 'train_schedules.csv'.")
