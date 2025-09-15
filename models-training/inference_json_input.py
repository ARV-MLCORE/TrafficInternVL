import torch
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

def worker_process(args):
    """
    Scenario-based worker: Each GPU processes complete scenarios
    """
    gpu_id, scenario_chunks, model_path, backup_dir, debug_limit = args

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Calculate total tasks for progress bar
    total_tasks = sum(len(phases) for _, phases in scenario_chunks)
    process_pbar = tqdm(total=total_tasks, position=gpu_id, desc=f"GPU-{gpu_id}")

    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).cuda().eval()
        print(f"[GPU-{gpu_id}] Model loaded successfully")
    except Exception as e:
        print(f"[GPU-{gpu_id}] FATAL: Could not load model. Exiting. Error: {e}")
        return {}

    processed_count = 0
    
    # Process each scenario assigned to this GPU
    for scenario_id, scenario_tasks in scenario_chunks:
        print(f"[GPU-{gpu_id}] Processing scenario: {scenario_id} ({len(scenario_tasks)} phases)")
        
        # Create scenario folder once
        scenario_backup_folder = os.path.join(backup_dir, scenario_id)
        os.makedirs(scenario_backup_folder, exist_ok=True)
        
        # Process each phase in this scenario
        for task in scenario_tasks:
            phase_num = task.get('phase_num', 'unknown_phase')
            backup_file_path = os.path.join(scenario_backup_folder, f"{phase_num}.json")
            
            # Check if this phase is already completed
            if os.path.exists(backup_file_path):
                try:
                    with open(backup_file_path, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                        # Validate completeness
                        if ('labels' in existing_data and 
                            len([k for k in existing_data.keys() if k.startswith('caption_')]) >= 2 and
                            'error' not in existing_data):
                            if processed_count < debug_limit:
                                print(f"[GPU-{gpu_id}] Skipping {scenario_id} phase {phase_num} - already completed")
                            process_pbar.update(1)
                            processed_count += 1
                            continue
                        else:
                            print(f"[GPU-{gpu_id}] Incomplete data in {scenario_id} phase {phase_num}, re-processing")
                except Exception as e:
                    print(f"[GPU-{gpu_id}] Corrupted backup {scenario_id} phase {phase_num}, re-processing. Error: {e}")

            # Process this phase
            final_result = None
            try:
                conversations_from_file = task['conversations']
                question_types = task.get('question_types', ['vehicle', 'pedestrian'])
                prompt1_user_turn = conversations_from_file[0][0] if isinstance(conversations_from_file[0], list) else conversations_from_file[0]
                prompt2_user_turn = conversations_from_file[1][0] if isinstance(conversations_from_file[1], list) else conversations_from_file[1]
                
                conversation_history = []

                # --- Round 1 ---
                conversation_history.append(prompt1_user_turn)
                
                if processed_count < debug_limit:
                    print(f"\n===== [GPU-{gpu_id} | {scenario_id} Phase {phase_num}] - Round 1 =====")
                    print(json.dumps(conversation_history, indent=2))

                inputs1 = processor.apply_chat_template(
                    conversation_history, add_generation_prompt=True, 
                    tokenize=True, return_dict=True, return_tensors="pt"
                ).to(model.device, dtype=torch.bfloat16)
                
                if processed_count < debug_limit:
                    print(f"[GPU-{gpu_id}] Token count round 1: {inputs1['input_ids'].shape[1]}")
                    
                if inputs1['input_ids'].shape[1] > 8192:
                    raise ValueError(f"Input too long ({inputs1['input_ids'].shape[1]} tokens)")
                    
                output_ids1 = model.generate(**inputs1, max_new_tokens=512, do_sample=False)
                response1 = processor.decode(output_ids1[0], skip_special_tokens=True).split("assistant\n")[-1].strip()
                assistant_turn = {"role": "assistant", "content": [{"type": "text", "text": response1}]}
                conversation_history.append(assistant_turn)

                # --- Round 2 ---
                prompt2_text_content = next((item for item in prompt2_user_turn['content'] if item['type'] == 'text'), None)
                if not prompt2_text_content:
                    raise ValueError(f"No text content in prompt2 for {scenario_id}")
                    
                prompt2_text_content['text'] = prompt2_text_content['text'].replace('<image>', '').strip()
                conversation_history.append({"role": "user", "content": [prompt2_text_content]})

                if processed_count < debug_limit:
                    print(f"\n===== [GPU-{gpu_id} | {scenario_id} Phase {phase_num}] - Round 2 =====")
                    print(json.dumps(conversation_history, indent=2))

                inputs2 = processor.apply_chat_template(
                    conversation_history, add_generation_prompt=True, 
                    tokenize=True, return_dict=True, return_tensors="pt"
                ).to(model.device, dtype=torch.bfloat16)
                
                if inputs2['input_ids'].shape[1] > 8192:
                    raise ValueError(f"Input too long ({inputs2['input_ids'].shape[1]} tokens)")
                    
                output_ids2 = model.generate(**inputs2, max_new_tokens=512, do_sample=False)
                response2 = processor.decode(output_ids2[0], skip_special_tokens=True).split("assistant\n")[-1].strip()

                final_result = {
                    "labels": [f"{scenario_id}_phase{phase_num}"], # Will be cleaned up later
                    f"caption_{question_types[0]}": response1,
                    f"caption_{question_types[1]}": response2,
                    "metadata": {
                        "scenario_id": scenario_id,
                        "phase_num": phase_num,
                        "processed_by_gpu": gpu_id,
                        "timestamp": str(datetime.now()),
                        "status": "completed"
                    }
                }

            except Exception as e:
                print(f"!!! [GPU-{gpu_id}] FAILED to process {scenario_id} phase {phase_num}: {e}")
                final_result = {
                    "labels": [f"{scenario_id}_phase{phase_num}"],
                    "error": str(e),
                    "metadata": {
                        "scenario_id": scenario_id,
                        "phase_num": phase_num,
                        "processed_by_gpu": gpu_id,
                        "timestamp": str(datetime.now()),
                        "status": "failed"
                    }
                }

            # Save result atomically
            if final_result:
                try:
                    temp_file = f"{backup_file_path}.tmp.{gpu_id}"
                    with open(temp_file, 'w', encoding='utf-8') as f:
                        json.dump(final_result, f, indent=2, ensure_ascii=False)
                        f.flush()
                        os.fsync(f.fileno())
                    
                    shutil.move(temp_file, backup_file_path)
                    
                    if processed_count < debug_limit:
                        print(f"[GPU-{gpu_id}] Saved: {scenario_id} phase {phase_num}")
                        
                except Exception as save_error:
                    print(f"!!! [GPU-{gpu_id}] FAILED to save {scenario_id} phase {phase_num}: {save_error}")
                    if os.path.exists(temp_file):
                        try:
                            os.remove(temp_file)
                        except:
                            pass
            
            process_pbar.update(1)
            processed_count += 1
            
            if processed_count % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"[GPU-{gpu_id}] Completed scenario: {scenario_id}")

    process_pbar.close()
    print(f"[GPU-{gpu_id}] Finished processing {processed_count} tasks")
    return {}


def aggregate_results(all_scenario_ids, backup_dir, output_file):
    """
    Reads all individual backup JSONs and aggregates them into a single
    final output file in the desired format.
    """
    print("\n--- Starting Aggregation Step ---")
    final_output = {}

    for scenario_id in tqdm(all_scenario_ids, desc="Aggregating scenarios"):
        scenario_folder = os.path.join(backup_dir, scenario_id)
        
        if not os.path.isdir(scenario_folder):
            print(f"Warning: Backup folder for scenario '{scenario_id}' not found. Skipping.")
            continue
            
        phase_results = []
        result_files = [f for f in os.listdir(scenario_folder) if f.endswith('.json')]

        for filename in result_files:
            file_path = os.path.join(scenario_folder, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Skip failed tasks
                if "error" in data or data.get("metadata", {}).get("status") == "failed":
                    continue

                # Get phase number for sorting and for the new label
                phase_num = data.get("metadata", {}).get("phase_num")
                if phase_num is None:
                    # Fallback to parsing from filename if metadata is missing
                    phase_num = os.path.splitext(filename)[0]
                
                # Format the entry as per the requested output
                formatted_entry = {
                    "labels": [str(phase_num)],
                    # This sort key will be removed later
                    "_sort_key": int(phase_num) 
                }
                
                # Copy all caption fields
                for key, value in data.items():
                    if key.startswith("caption_"):
                        formatted_entry[key] = value

                phase_results.append(formatted_entry)

            except json.JSONDecodeError:
                print(f"Warning: Corrupted JSON file skipped: {file_path}")
            except Exception as e:
                print(f"Warning: Could not process file {file_path}. Error: {e}")

        # Sort the phases for this scenario in descending order (e.g., 4, 3, 2, 1, 0)
        phase_results.sort(key=lambda x: x["_sort_key"], reverse=True)
        
        # Clean up the temporary sort key before adding to the final output
        for entry in phase_results:
            del entry["_sort_key"]
            
        if phase_results:
            final_output[scenario_id] = phase_results

    # Save the final aggregated file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        print(f"\nAggregation complete. Final results saved to: {output_file}")
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

    # Group tasks by scenario
    scenarios = defaultdict(list)
    for task in tasks:
        scenario_id = task.get('scenario', 'unknown_scenario')
        scenarios[scenario_id].append(task)
    
    print(f"Found {len(scenarios)} unique scenarios")
    # ... (rest of the initial print statements remain the same)

    num_gpus = args.num_gpus
    print(f"Using {num_gpus} GPUs for inference.")
    print(f"Backup files will be saved in: {args.backup_dir}")

    if args.debug_first_n > 0:
        print(f"*** DEBUG MODE ON: Detailed output for the first {args.debug_first_n} tasks per GPU. ***")

    # Distribute scenarios across GPUs (round-robin)
    scenario_chunks = [[] for _ in range(num_gpus)]
    scenario_list = list(scenarios.items())
    
    scenario_list.sort(key=lambda x: len(x[1]), reverse=True)
    
    for i, (scenario_id, scenario_tasks) in enumerate(scenario_list):
        gpu_id = i % num_gpus
        scenario_chunks[gpu_id].append((scenario_id, scenario_tasks))
    
    # ... (rest of the distribution print statements remain the same)

    # Prepare worker arguments
    worker_args = [
        (gpu_id, scenario_chunks[gpu_id], args.model_path, args.backup_dir, args.debug_first_n) 
        for gpu_id in range(num_gpus)
    ]
    
    print("\nStarting multiprocessing pool for inference...")
    with mp.Pool(processes=num_gpus) as pool:
        pool.map(worker_process, worker_args)
    print("All workers have finished inference.")

    # --- NEW: Call the aggregation function after all processes are done ---
    all_scenario_ids = list(scenarios.keys())
    aggregate_results(all_scenario_ids, args.backup_dir, args.output_file)

    print("\nInference and aggregation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scenario-based Multiprocessing Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the merged model directory.")
    parser.add_argument("--test_set_json", type=str, required=True, help="Path to the JSONL file containing test prompts.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the final aggregated output JSON file.")
    parser.add_argument("--backup_dir", type=str, required=True, help="Directory to store per-scenario JSON backups.")
    parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs to use for parallel processing.")
    parser.add_argument("--debug_first_n", type=int, default=0, help="Print detailed output for the first N tasks per GPU.")
    
    args = parser.parse_args()
    main(args)