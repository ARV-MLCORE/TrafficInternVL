import json

# --- CONFIGURATION ---
# IMPORTANT: Please ensure this path is correct. Using an absolute path is best if you're unsure.
file_path = '/workspace/Park/LLaMA-Factory/data/aicity_train_data.json' 

# Define the old prompts (the core text to find)
old_pedestrian_text = "Please describe the interested pedestrian in the video."
old_vehicle_text = "Please describe the interested vehicle in the video."

# Define the new, full prompts to replace them with
new_pedestrian_prompt = "This picture shows the relationship between the pedestrian in the green box and the vehicle in the blue box. Describe the pedestrian in the green box or the pedestrian closest to the vehicle based on age, height, clothing, line of sight, relative position to the vehicle, movement status, weather conditions and road environment."
new_vehicle_prompt = "This picture shows the relationship between the vehicle in the blue box and the pedestrian in the green box. Describe the vehicle in the blue box or the vehicle closest to the pedestrian based on the relative position to the pedestrian, driving status, weather conditions and road environment. And describe the age, height, clothing of the pedestrian."

# --- SCRIPT LOGIC ---
replacements_made = 0

try:
    print(f"--> Attempting to read file: '{file_path}'")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print("--> File read successfully.")

    # Iterate through each record in the JSON data
    for i, item in enumerate(data):
        if 'conversations' in item:
            for conv_turn in item['conversations']:
                if conv_turn.get('from') == 'human':
                    current_value = conv_turn.get('value', '')
                    
                    # Check if the pedestrian text is in the current value
                    if old_pedestrian_text in current_value:
                        # Reconstruct the value, preserving the <image> tag
                        conv_turn['value'] = f"<image>\n{new_pedestrian_prompt}"
                        replacements_made += 1
                        print(f"    [Record {i+1}] Replaced pedestrian prompt.")
                        
                    # Check if the vehicle text is in the current value
                    elif old_vehicle_text in current_value:
                        conv_turn['value'] = new_vehicle_prompt
                        replacements_made += 1
                        print(f"    [Record {i+1}] Replaced vehicle prompt.")

    # Only write to the file if changes were actually made
    if replacements_made > 0:
        print(f"\n--> Found and replaced a total of {replacements_made} prompts.")
        print("--> Writing changes back to the file...")
        with open(file_path, 'w', encoding='utf-8') as f:
            # Use indent=2 to keep the JSON format readable
            json.dump(data, f, indent=2, ensure_ascii=False)
        print("--> File successfully updated!")
    else:
        print("\n--> No matching prompts were found to replace. The file has not been changed.")

except FileNotFoundError:
    print(f"\n--- ERROR ---")
    print(f"The file '{file_path}' was not found.")
    print("Please check that the path is correct and that you are running the script from the right directory.")
except json.JSONDecodeError:
    print(f"\n--- ERROR ---")
    print(f"The file '{file_path}' is not a valid JSON file. Please check its contents for errors.")
except Exception as e:
    print(f"\n--- ERROR ---")
    print(f"An unexpected error occurred: {e}")