import json
import argparse
import os
from tqdm import tqdm

# --- 1. กำหนด Prompt ใหม่ตามลำดับที่จะใช้ ---
# Prompt ของ Vehicle จะขึ้นก่อน และมี <image> tag
PROMPT_VEHICLE_FIRST = """<image>
This picture shows the relationship between the vehicle in the blue box and the pedestrian in the green box. Describe the vehicle in the blue box or the vehicle closest to the pedestrian based on the relative position to the pedestrian, driving status, weather conditions and road environment. And describe the age, height, and clothing of the pedestrian."""

# Prompt ของ Pedestrian จะมาทีหลัง และไม่มี <image> tag
PROMPT_PEDESTRIAN_SECOND = """This picture shows the relationship between the pedestrian in the green box and the vehicle in the blue box. Describe the pedestrian in the green box or the pedestrian closest to the vehicle based on age, height, clothing, line of sight, relative position to the vehicle, movement status, weather conditions and road environment."""

# --- 2. สร้างฟังก์ชันหลักเพื่อห่อหุ้มตรรกะ ---
def swap_conversation_order(input_file, output_file):
    """
    Reads a JSON dataset, swaps the order of pedestrian and vehicle conversation turns,
    and saves the result to a new file.
    """
    swapped_records = 0
    
    try:
        print(f"--> Attempting to read input file: '{input_file}'")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"--> File read successfully. Found {len(data)} records.")

        # --- 3. ตรรกะการสลับลำดับ ---
        print("--> Starting conversation swap process...")
        for item in tqdm(data, desc="Swapping conversations"):
            if 'conversations' in item and len(item['conversations']) == 4:
                try:
                    # ดึงคำตอบ (caption) เดิมออกมาเก็บไว้ก่อน
                    # คำตอบของ Pedestrian อยู่ตำแหน่งที่ 2 (index 1)
                    ped_caption = item['conversations'][1]['value']
                    # คำตอบของ Vehicle อยู่ตำแหน่งที่ 4 (index 3)
                    veh_caption = item['conversations'][3]['value']
                    
                    # สร้าง list บทสนทนาใหม่ทั้งหมดตามลำดับที่ต้องการ
                    swapped_conversations = [
                        # Turn 1: Vehicle
                        {"from": "human", "value": PROMPT_VEHICLE_FIRST},
                        {"from": "gpt", "value": veh_caption},
                        
                        # Turn 2: Pedestrian
                        {"from": "human", "value": PROMPT_PEDESTRIAN_SECOND},
                        {"from": "gpt", "value": ped_caption}
                    ]
                    
                    # นำบทสนทนาที่สลับแล้วไปเขียนทับของเดิมใน record นั้นๆ
                    item['conversations'] = swapped_conversations
                    swapped_records += 1

                except (IndexError, KeyError) as e:
                    # ถ้าโครงสร้างข้อมูลบาง record ไม่ตรงตามคาด ให้ข้ามไป
                    print(f"Warning: Skipping a record due to unexpected format: {e}")
                    continue

        # --- 4. การบันทึกไฟล์ (ที่ปลอดภัยกว่า) ---
        if swapped_records > 0:
            print(f"\n--> Successfully swapped conversations in {swapped_records} records.")
            print(f"--> Writing modified data to new file: '{output_file}'")
            
            # สร้างโฟลเดอร์ถ้ายังไม่มี
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print("--> New file successfully created!")
        else:
            print("\n--> No records were swapped. The output file was not created.")

    except FileNotFoundError:
        print(f"\n--- ERROR: File Not Found ---")
        print(f"The input file '{input_file}' was not found.")
    except Exception as e:
        print(f"\n--- UNEXPECTED ERROR ---")
        print(f"An error occurred: {e}")

# --- 5. การรับค่าจาก Command Line ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Swap the order of conversation turns in the dataset.")
    parser.add_argument('--input-file', type=str, required=True, help='Path to the source JSON file.')
    parser.add_argument('--output-file', type=str, required=True, help='Path to save the new, swapped JSON file.')
    
    args = parser.parse_args()
    
    # แก้ไขกรณีที่ระบุ output file แบบไม่มีโฟลเดอร์
    if not os.path.dirname(args.output_file):
        args.output_file = os.path.join('.', args.output_file)

    swap_conversation_order(args.input_file, args.output_file)