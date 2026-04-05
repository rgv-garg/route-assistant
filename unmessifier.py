import pandas as pd
import re

# 1. READ AND REPAIR THE BROKEN CSV LINES
# We open it as a text file first to stitch together those broken station names
with open('messy_data.csv', 'r') as file:
    raw_lines = file.readlines()

fixed_lines = []
for line in raw_lines:
    line = line.strip()
    if not line or "Station Name" in line:  # Skip empty lines and the header
        continue

    # If the line starts with a 5-digit train number (e.g., "12673,")
    if re.match(r'^\d{5}', line):
        fixed_lines.append(line)
    else:
        # It doesn't start with a train number! This means it's the broken second-half
        # of the previous row (like "MGR CTL"). We stitch it back onto the last line.
        if fixed_lines:
            fixed_lines[-1] += " " + line

# 2. PARSE INTO CLEAN DATA
data = []
for line in fixed_lines:
    parts = line.split(',')
    if len(parts) >= 3:
        train = parts[0].strip()
        station = parts[1].strip()
        delay = parts[2].strip()

        # Drop the useless N/A rows completely
        if station != 'N/A' and delay != 'N/A':
            data.append({
                'Train Number': train,
                'Station Name': station,
                'Delay': delay
            })

# 3. GROUP BY DAYS USING THE "ORIGIN STATION" LOGIC
cleaned_data = []
current_train = None
origin_station = None
day_block = 0

for row in data:
    train = row['Train Number']
    station = row['Station Name']

    # If we switch to a new train, reset our trackers
    if train != current_train:
        current_train = train
        origin_station = station  # The very first valid station is our Origin!
        day_block = 1

    # If we see the Origin Station again on the same train, a new day's block has started!
    elif station == origin_station:
        day_block += 1

    row['Day Block'] = day_block
    cleaned_data.append(row)

# 4. SAVE TO 3 SEPARATE FILES
df = pd.DataFrame(cleaned_data)

# Based on your 6-day data structure:
# Block 4 = 3rd April (Pink), Block 5 = 2nd April (Green), Block 6 = 1st April (Yellow)
target_blocks = {
    4: "3rd_April",
    5: "2nd_April",
    6: "1st_April"
}

print("Organizing your 91 trains...")

for block_num, date_name in target_blocks.items():
    # Filter for just that specific block
    df_filtered = df[df['Day Block'] == block_num]

    # Drop the internal Day Block column so the final sheet looks clean
    df_filtered = df_filtered.drop(columns=['Day Block'])

    if not df_filtered.empty:
        filename = f"Cleaned_Train_Data_{date_name}.csv"
        df_filtered.to_csv(filename, index=False)
        print(f"Success! Saved {len(df_filtered)} rows to {filename}")
    else:
        print(f"Warning: No data found for {date_name} (Block {block_num})")

print("\nDone! You now have 3 perfect Excel/CSV files.")
