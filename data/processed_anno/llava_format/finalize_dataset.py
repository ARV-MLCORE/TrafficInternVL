import json
import argparse
import os
from tqdm import tqdm

def transform_to_final_format(input_file, output_file):
    """
    Reads a JSON file in the intermediate format and transforms it into the final
    format with only "conversations" and "images" keys.

    Args:
        input_file (str): Path to the source JSON file.
        output_file (str): Path to save the final transformed JSON file.
    """
    print(f"Loading data from: {input_file}")
    
    # 1. อ่านไฟล์ JSON ต้นทาง
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            source_data = json.load(f)
        print(f"Found {len(source_data)} records to transform.")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_file}. The file might be corrupted.")
        return

    # 2. เตรียม List ว่างสำหรับเก็บข้อมูลที่แปลงแล้ว
    final_dataset = []

    # 3. วนลูปเพื่อแปลงข้อมูลทีละรายการ
    print("Transforming data into the final format...")
    for item in tqdm(source_data, desc="Processing records"):
        # สร้าง Dictionary ใหม่ตามรูปแบบที่ต้องการ
        new_item = {
            # คัดลอก "conversations" มาโดยตรง
            "conversations": item.get("conversations", []),
            
            # นำค่าของ "image" (ที่เป็น string) มาใส่ใน List ใหม่
            # แล้วกำหนดให้กับ key "images"
            "images": [item.get("image", "")]
        }
        
        # เพิ่มข้อมูลที่แปลงแล้วลงใน List สุดท้าย
        final_dataset.append(new_item)

    # 4. บันทึก List สุดท้ายเป็นไฟล์ JSON ใหม่
    print(f"Saving final dataset to: {output_file}")
    
    # สร้างโฟลเดอร์ถ้ายังไม่มี
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # ใช้ indent=2 หรือ 4 เพื่อให้ไฟล์อ่านง่าย
        json.dump(final_dataset, f, indent=2, ensure_ascii=False)

    print("Transformation complete!")
    print(f"Total records in the final dataset: {len(final_dataset)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transform dataset to the final LLaVA format.")
    
    parser.add_argument('--input-file', type=str, required=True,
                        help='Path to the source JSON file (e.g., wts_bdd_local_train.json)')
    
    parser.add_argument('--output-file', type=str, required=True,
                        help='Path to save the final JSON file (e.g., final_dataset_train.json)')

    args = parser.parse_args()
    
    transform_to_final_format(args.input_file, args.output_file)