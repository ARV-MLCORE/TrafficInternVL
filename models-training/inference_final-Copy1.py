import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import os
import json
from tqdm import tqdm
import argparse
import multiprocessing as mp
from collections import defaultdict

# =================================================================================
# 1. Worker Function: Final Corrected Version
# =================================================================================
def worker_process(args):
    """
    This is the final, corrected version of the worker process. It incorporates fixes for:
    1. Data type mismatch (bfloat16).
    2. 'string indices must be integers' by ensuring consistent content structure.
    3. 'sequence length is longer...' by sending only text in the follow-up turn.
    """
    # Unpack arguments for this worker
    gpu_id, task_chunk, model_path, backup_dir = args

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    process_pbar = tqdm(total=len(task_chunk), position=gpu_id, desc=f"GPU-{gpu_id}")

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

    # Loop through each task assigned to this worker
    for task in task_chunk:
        scenario_id = task.get('scenario', 'unknown_scenario')
        
        video_backup_path = os.path.join(backup_dir, scenario_id.split('_event_')[0])
        os.makedirs(video_backup_path, exist_ok=True)
        backup_file_path = os.path.join(video_backup_path, f"{scenario_id}.json")

        if os.path.exists(backup_file_path):
            try:
                with open(backup_file_path, 'r', encoding='utf-8') as f:
                    saved_result = json.load(f)
                worker_results[scenario_id] = saved_result
                process_pbar.update(1)
                continue
            except Exception as e:
                print(f"[GPU-{gpu_id}] Could not read backup {backup_file_path}, re-processing. Error: {e}")

        try:
            conversations_from_file = task['conversations']
            print(f"conversations_from_file: {conversations_from_file}")
            question_types = task.get('question_types', ['vehicle', 'pedestrian'])
            print(f"question_types: {question_types}")

            prompt1_user_turn = conversations_from_file[0][0] if isinstance(conversations_from_file[0], list) else conversations_from_file[0]
            print(f"prompt1_user_turn: {prompt1_user_turn}")
            prompt2_user_turn = conversations_from_file[1][0] if isinstance(conversations_from_file[1], list) else conversations_from_file[1]
            print(f"prompt2_user_turn: {prompt2_user_turn}")
            
            conversation_history = []

            # --- Round 1: Ask the first question ---
            conversation_history.append(prompt1_user_turn)
            print(f"conversation_history_round1:{conversation_history}")
            
            # FIX 1: Added dtype=torch.bfloat16
            inputs1 = processor.apply_chat_template(conversation_history, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
            
            output_ids1 = model.generate(**inputs1, max_new_tokens=512, do_sample=False)
            response1 = processor.decode(output_ids1[0], skip_special_tokens=True).split("assistant\n")[-1].strip()
            
            # FIX 2: Correct 'string indices' error by wrapping the assistant's response in a consistent structure.
            assistant_turn = {"role": "assistant", "content": [{"type": "text", "text": response1}]}
            conversation_history.append(assistant_turn)
            print(f"conversation_history_after_assistance:{conversation_history}")

            # --- Round 2: Ask the second question (with context) ---
            # FIX 3: Correct 'sequence length' error by sending ONLY the text from the second prompt.
            prompt2_text_content = next((item for item in prompt2_user_turn['content'] if item['type'] == 'text'), None)
            print(f"prprompt2_text_content :{prompt2_text_content }")

            if not prompt2_text_content:
                 raise ValueError(f"No text content found in the second prompt for task {scenario_id}")

            # Append only the text part as the new user turn. The model remembers the images from the first turn.
            conversation_history.append({"role": "user", "content": [prompt2_text_content]})
            print(f"conversation_history_after 2nd prompt :{conversation_history }")
            
            # FIX 1: Added dtype=torch.bfloat16
            inputs2 = processor.apply_chat_template(conversation_history, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
            
            output_ids2 = model.generate(**inputs2, max_new_tokens=512, do_sample=False)
            response2 = processor.decode(output_ids2[0], skip_special_tokens=True).split("assistant\n")[-1].strip()

            final_result = {
                "labels": [scenario_id],
                f"caption_{question_types[0]}": response1,
                f"caption_{question_types[1]}": response2
            }
            print(f"final_result:{final_result}")
            
            with open(backup_file_path, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, indent=2, ensure_ascii=False)
            
            worker_results[scenario_id] = final_result

        except Exception as e:
            print(f"!!! [GPU-{gpu_id}] FAILED to process task {scenario_id}: {e}")
            print("--- DATA FOR FAILED TASK ---")
            print(json.dumps(task, indent=2))
            print("--------------------------")
            worker_results[scenario_id] = {"labels": [scenario_id], "caption_vehicle": f"Error: {e}", "caption_pedestrian": f"Error: {e}"}
        
        process_pbar.update(1)

    process_pbar.close()
    return worker_results

# =================================================================================
# 2. Main Function: MODIFIED to read a JSONL file
# =================================================================================
def main(args):
    mp.set_start_method("spawn", force=True)

    # --- MODIFIED: Read tasks from the JSONL file line by line ---
    try:
        tasks = []
        with open(args.test_set_json, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    tasks.append(json.loads(line.strip()))
        print(f"Loaded {len(tasks)} tasks from {args.test_set_json}")
    except Exception as e:
        print(f"FATAL: Could not read test set file {args.test_set_json}. Error: {e}")
        return

    num_gpus = args.num_gpus
    print(f"Using {num_gpus} GPUs for inference.")
    print(f"Intermediate backup files will be saved in: {args.backup_dir}")
    
    task_chunks = [[] for _ in range(num_gpus)]
    for i, task in enumerate(tasks):
        task_chunks[i % num_gpus].append(task)

    worker_args = [
        (gpu_id, task_chunks[gpu_id], args.model_path, args.backup_dir) 
        for gpu_id in range(num_gpus)
    ]
    
    print("Starting multiprocessing pool...")
    with mp.Pool(processes=num_gpus) as pool:
        list_of_results = pool.map(worker_process, worker_args)
    print("All workers have finished.")

    final_results = {}
    for partial_result in list_of_results:
        final_results.update(partial_result)

    print(f"Saving final combined results to: {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print("\nInference complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conversational Multiprocessing Inference from Test Set")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the merged model directory.")
    parser.add_argument("--test_set_json", type=str, required=True, help="Path to the JSONL file containing test prompts.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the single FINAL output JSON file.")
    parser.add_argument("--backup_dir", type=str, required=True, help="Directory to store intermediate per-task JSON backups.")
    parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs to use for parallel processing.")
    
    args = parser.parse_args()
    main(args)