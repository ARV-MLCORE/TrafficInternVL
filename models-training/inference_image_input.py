import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import os
import json
from tqdm import tqdm
import argparse
import multiprocessing as mp


def worker_process_conversational(args):
    """
    This function is executed by each individual process.
    It now processes prompts sequentially, maintaining conversational context.
    The second prompt is asked based on the context of the first prompt and answer.
    """
    # Unpack arguments
    gpu_id, video_chunk, model_path, test_data_dir, backup_dir = args

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    process_pbar = tqdm(total=len(video_chunk), position=gpu_id, desc=f"GPU-{gpu_id}")

    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).cuda().eval()
    except Exception as e:
        print(f"[GPU-{gpu_id}] FATAL: Could not load model. Exiting. Error: {e}")
        return {}

    worker_results = {}

    for video_name in video_chunk:
        video_path = os.path.join(test_data_dir, video_name)
        if not os.path.isdir(video_path):
            continue

        video_backup_path = os.path.join(backup_dir, video_name)
        os.makedirs(video_backup_path, exist_ok=True)
            
        frame_files = sorted([f for f in os.listdir(video_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        video_results = []

        for frame_file in frame_files:
            image_path = os.path.join(video_path, frame_file)
            frame_label = os.path.splitext(frame_file)[0]
            
            backup_file_path = os.path.join(video_backup_path, f"{frame_label}.json")
            if os.path.exists(backup_file_path):
                try:
                    with open(backup_file_path, 'r', encoding='utf-8') as f:
                        saved_result = json.load(f)
                    video_results.append(saved_result)
                    continue 
                except Exception as e:
                    print(f"[GPU-{gpu_id}] Could not read backup {backup_file_path}, re-processing. Error: {e}")

            try:
                image = Image.open(image_path).convert("RGB")
        
                query1 = "This picture shows the relationship between the vehicle in the blue box and the pedestrian in the green box. Describe the vehicle in the blue box or the vehicle closest to the pedestrian based on the relative position to the pedestrian, driving status, weather conditions and road environment. And describe the age, height, and clothing of the pedestrian."
                query2 = "This picture shows the relationship between the pedestrian in the green box and the vehicle in the blue box. Describe the pedestrian in the green box or the pedestrian closest to the vehicle based on age, height, clothing, line of sight, relative position to the vehicle, movement status, weather conditions and road environment."
                
                # --- Step 1: Start the conversation with the first question ---
                messages = [
                    {"role": "user", "content": [{"type": "image", "url": image_path}, {"type": "text", "text": query1}]}
                ]
                inputs1 = processor.apply_chat_template(messages, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
                
                output_ids1 = model.generate(
                    **inputs1,
                    max_new_tokens=1024,
                    do_sample=False
                )
                response_vehicle = processor.decode(output_ids1[0], skip_special_tokens=True).split("assistant\n")[-1].strip()
                
                # --- Step 2: Append the model's first answer and the user's second question to the conversation history ---
                messages.append({"role": "assistant", "content": [{"type": "text", "text": response_vehicle}]})

                messages.append({"role": "user", "content": [{"type": "text", "text": query2}]})

        
                inputs2_contextual = processor.apply_chat_template(messages, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

                output_ids2_contextual = model.generate(
                    **inputs2_contextual,
                    max_new_tokens=1024,
                    do_sample=False
                )
                response_pedestrian = processor.decode(output_ids2_contextual[0], skip_special_tokens=True).split("assistant\n")[-1].strip()
                
                frame_result = {
                    "labels": [frame_label],
                    "caption_pedestrian": response_pedestrian,
                    "caption_vehicle": response_vehicle
                }

                with open(backup_file_path, 'w', encoding='utf-8') as f:
                    json.dump(frame_result, f, indent=2, ensure_ascii=False)

                video_results.append(frame_result)

            except Exception as e:
                print(f"[GPU-{gpu_id}] Failed to process {image_path}: {e}")
                video_results.append({"labels": [frame_label], "caption_pedestrian": f"Error: {e}", "caption_vehicle": f"Error: {e}"})
            
        worker_results[video_name] = video_results
        process_pbar.update(1)

    process_pbar.close()
    return worker_results

def main(args):
    mp.set_start_method("spawn", force=True)

    num_gpus = args.num_gpus
    print(f"Using {num_gpus} GPUs for inference.")
    print(f"Intermediate backup files will be saved in: {args.backup_dir}")

    all_video_folders = sorted([d for d in os.listdir(args.test_data_dir) if os.path.isdir(os.path.join(args.test_data_dir, d))])
    
    if not all_video_folders:
        print("No video folders found in the test data directory. Exiting.")
        return

    video_chunks = [[] for _ in range(num_gpus)]
    for i, video_folder in enumerate(all_video_folders):
        video_chunks[i % num_gpus].append(video_folder)

    worker_args = [
        (gpu_id, video_chunks[gpu_id], args.model_path, args.test_data_dir, args.backup_dir) 
        for gpu_id in range(num_gpus)
    ]
    
    print("Starting multiprocessing pool...")
    with mp.Pool(processes=num_gpus) as pool:
        # ### MODIFIED: Call the new conversational function ###
        list_of_results = pool.map(worker_process_conversational, worker_args)
    print("All workers have finished.")

    final_results = {}
    for partial_result in list_of_results:
        final_results.update(partial_result)

    print(f"Saving final combined results to: {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print("\nInference complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multiprocessing Inference Pipeline with Backup/Resume and Conversational Context")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the merged model directory.")
    parser.add_argument("--test_data_dir", type=str, required=True, help="Directory containing video subfolders.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the single FINAL output JSON file.")
    parser.add_argument("--backup_dir", type=str, required=True, help="Directory to store intermediate per-frame JSON backups.")
    parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs to use for parallel processing.")
    
    args = parser.parse_args()
    main(args)