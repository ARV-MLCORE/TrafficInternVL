from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import os
import json
from tqdm import tqdm
import argparse
import multiprocessing as mp
from collections import defaultdict
from datetime import datetime
import shutil
import torch
import re

# The final, correct version for Task 2, adapted with the proven logic from Task 1.
def worker_process(args):
    gpu_id, question_chunks, model_path, backup_dir, debug_limit = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    total_tasks = len(question_chunks)
    process_pbar = tqdm(total=total_tasks, position=gpu_id, desc=f"GPU-{gpu_id}")

    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).cuda().eval()
        print(f"[GPU-{gpu_id}] Model loaded successfully")
    except Exception as e:
        print(f"[GPU-{gpu_id}] FATAL: Could not load model. Exiting. Error: {e}")
        return {}

    processed_count = 0
    results = []
    
    for task in question_chunks:
        question_id = task.get('question_id', f'unknown_{processed_count}')
        backup_file_path = os.path.join(backup_dir, f"{question_id}.json")
        
        # Backup checking logic (no changes needed)
        if os.path.exists(backup_file_path):
            try:
                with open(backup_file_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                if 'correct' in existing_data and existing_data.get('correct') != "error":
                    if processed_count < debug_limit:
                        print(f"[GPU-{gpu_id}] Skipping {question_id} - already completed")
                    results.append(existing_data)
                    process_pbar.update(1)
                    processed_count += 1
                    continue
            except Exception as e:
                print(f"[GPU-{gpu_id}] Corrupted backup {question_id}, re-processing. Error: {e}")

        final_result = None
        try:
            # =======================================================================
            # ### THE FINAL ADAPTATION FROM TASK 1 ###
            # =======================================================================
            
            original_conversations = task.get('conversations')
            if not original_conversations:
                raise ValueError("Task has a missing 'conversations' field.")

            # Create a new, perfectly structured list for the processor.
            reformatted_conversations = []
            for msg in original_conversations:
                role = msg.get('role')
                content = msg.get('content')

                if not role or content is None:
                    raise KeyError("A message is missing 'role' or 'content'.")

                # The GOLDEN RULE from Task 1: Ensure ALL content is a list of dicts.
                if isinstance(content, str):
                    # This handles the 'system' role by wrapping its string content.
                    reformatted_msg = {
                        "role": role,
                        "content": [{"type": "text", "text": content}]
                    }
                elif isinstance(content, list):
                    # This handles the 'user' role which is already a list.
                    reformatted_msg = msg
                else:
                    raise TypeError(f"Unsupported content type: {type(content).__name__}")
                
                reformatted_conversations.append(reformatted_msg)
            
            # Now, use the perfectly consistent, reformatted conversations.
            inputs = processor.apply_chat_template(
                reformatted_conversations, add_generation_prompt=True, 
                tokenize=True, return_dict=True, return_tensors="pt"
            ).to(model.device, dtype=torch.bfloat16)
            
            if inputs['input_ids'].shape[1] > 8192:
                raise ValueError(f"Input too long ({inputs['input_ids'].shape[1]} tokens)")
                
            output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            response = processor.decode(output_ids[0], skip_special_tokens=True).split("assistant\n")[-1].strip()
            
            answer_choice = extract_answer_choice(response)

            final_result = { "id": question_id, "correct": answer_choice.upper() if answer_choice else "unknown", "raw_response": response, "metadata": { "scenario": task.get('scenario', 'unknown'), "phase_num": task.get('phase_num', 'unknown'), "question_type": task.get('question_type', 'unknown'), "processed_by_gpu": gpu_id, "timestamp": str(datetime.now()), "status": "completed" } }
            results.append(final_result)

        except Exception as e:
            # Enhanced error logging remains useful
            print(f"\n!!! [GPU-{gpu_id}] FAILED to process {question_id}. Error Type: {type(e).__name__}, Message: {e}")
            print("--- Problematic Task Data ---")
            print(json.dumps(task, indent=2, ensure_ascii=False))
            print("-----------------------------\n")
            final_result = { "id": question_id, "correct": "error", "error": str(e), "metadata": { "scenario": task.get('scenario', 'unknown'), "phase_num": task.get('phase_num', 'unknown'), "question_type": task.get('question_type', 'unknown'), "processed_by_gpu": gpu_id, "timestamp": str(datetime.now()), "status": "failed" } }
            results.append(final_result)

        # Saving logic (no changes needed)
        if final_result:
            try:
                temp_file = f"{backup_file_path}.tmp.{gpu_id}"
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(final_result, f, indent=2, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())
                shutil.move(temp_file, backup_file_path)
            except Exception as save_error:
                print(f"!!! [GPU-{gpu_id}] FAILED to save {question_id}: {save_error}")
                if os.path.exists(temp_file):
                    try: os.remove(temp_file)
                    except: pass
        
        process_pbar.update(1)
        processed_count += 1
        if processed_count % 10 == 0 and torch.cuda.is_available(): torch.cuda.empty_cache()

    process_pbar.close()
    print(f"[GPU-{gpu_id}] Finished processing {processed_count} tasks")
    return results


def extract_answer_choice(response):
    """
    Extract the answer choice (A, B, C, or D) from the model's response.
    """
    # Look for patterns like "A.", "B)", "(C)", "D", etc.
    patterns = [
        r'\b([ABCD])\.',  # A., B., C., D.
        r'\b([ABCD])\)',  # A), B), C), D)
        r'\(([ABCD])\)',  # (A), (B), (C), (D)
        r'\b([ABCD])\b',  # Just A, B, C, D as standalone
        r'answer\s*:?\s*([ABCD])',  # "answer: A" or "answer A"
        r'choice\s*:?\s*([ABCD])',  # "choice: A" or "choice A"
        r'option\s*:?\s*([ABCD])',  # "option: A" or "option A"
    ]
    
    # Try each pattern
    for pattern in patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            return matches[0].upper()
    
    # If no clear pattern found, look for the first occurrence of A, B, C, or D
    for char in ['A', 'B', 'C', 'D']:
        if char in response.upper():
            return char
    
    return None


def aggregate_results(backup_dir, output_file):
    """
    Reads all individual backup JSONs and aggregates them into the final format.
    """
    print("\n--- Starting Aggregation Step ---")
    final_results = []

    # Get all backup files
    backup_files = [f for f in os.listdir(backup_dir) if f.endswith('.json')]
    
    for filename in tqdm(backup_files, desc="Aggregating results"):
        file_path = os.path.join(backup_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Skip failed tasks or those without proper answers
            if "error" in data or data.get("metadata", {}).get("status") == "failed":
                continue
                
            if "correct" not in data or data["correct"] == "error":
                continue

            # Format the entry as per the requested output
            formatted_entry = {
                "id": data["id"],
                "correct": data["correct"]
            }
            
            final_results.append(formatted_entry)

        except json.JSONDecodeError:
            print(f"Warning: Corrupted JSON file skipped: {file_path}")
        except Exception as e:
            print(f"Warning: Could not process file {file_path}. Error: {e}")

    # Save the final aggregated file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        print(f"\nAggregation complete. Final results saved to: {output_file}")
        print(f"Total processed questions: {len(final_results)}")
    except Exception as e:
        print(f"\nFATAL: Could not write final output file to {output_file}. Error: {e}")


def main(args):
    mp.set_start_method("spawn", force=True)

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

    # Create backup directory
    os.makedirs(args.backup_dir, exist_ok=True)
    
    num_gpus = args.num_gpus
    print(f"Using {num_gpus} GPUs for inference.")
    print(f"Backup files will be saved in: {args.backup_dir}")

    if args.debug_first_n > 0:
        print(f"*** DEBUG MODE ON: Detailed output for the first {args.debug_first_n} tasks per GPU. ***")

    # Distribute tasks across GPUs (round-robin)
    task_chunks = [[] for _ in range(num_gpus)]
    
    for i, task in enumerate(tasks):
        gpu_id = i % num_gpus
        task_chunks[gpu_id].append(task)
    
    # Print distribution info
    for gpu_id in range(num_gpus):
        print(f"GPU-{gpu_id}: {len(task_chunks[gpu_id])} tasks")

    # Prepare worker arguments
    worker_args = [
        (gpu_id, task_chunks[gpu_id], args.model_path, args.backup_dir, args.debug_first_n) 
        for gpu_id in range(num_gpus)
    ]
    
    print("\nStarting multiprocessing pool for inference...")
    with mp.Pool(processes=num_gpus) as pool:
        all_results = pool.map(worker_process, worker_args)
    print("All workers have finished inference.")

    # Aggregate results
    aggregate_results(args.backup_dir, args.output_file)

    print("\nInference and aggregation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 2 Multiple-Choice Question Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the merged model directory.")
    parser.add_argument("--test_set_json", type=str, required=True, help="Path to the JSONL file containing test prompts.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the final aggregated output JSON file.")
    parser.add_argument("--backup_dir", type=str, required=True, help="Directory to store per-question JSON backups.")
    parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs to use for parallel processing.")
    parser.add_argument("--debug_first_n", type=int, default=0, help="Print detailed output for the first N tasks per GPU.")
    
    args = parser.parse_args()
    main(args)