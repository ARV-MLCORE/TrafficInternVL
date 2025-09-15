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

def group_questions_by_scenario_phase(tasks):
    """
    Group questions by scenario and phase to build chat history.
    Returns a dictionary with (scenario, phase) as keys and list of questions as values.
    """
    grouped = defaultdict(list)
    for task in tasks:
        scenario = task.get('scenario')
        phase = task.get('phase_num')
        key = (scenario, phase)
        grouped[key].append(task)
    
    # Sort questions within each group by question_id for consistency
    for key in grouped:
        grouped[key].sort(key=lambda x: x.get('question_id', ''))
    
    return grouped

def build_chat_history_conversations(questions_group):
    """
    Build chat history conversations for a group of questions from the same scenario and phase.
    Format: system prompt -> images -> system -> question1 -> assistant1 -> question2 -> assistant2 -> ...
    """
    if not questions_group:
        return []
    
    # Get the first question to extract system prompt and images
    first_question = questions_group[0]
    original_conversations = first_question.get('conversations', [])
    
    if len(original_conversations) < 2:
        raise ValueError("Invalid conversation structure")
    
    # Extract system prompt and user content (images + first question)
    system_msg = original_conversations[0]  # system role
    first_user_msg = original_conversations[1]  # user role with images and question
    
    # Build the chat history conversations
    chat_conversations = []
    
    # Add system prompt
    chat_conversations.append(system_msg)
    
    # Extract images from the first user message
    user_content = first_user_msg.get('content', [])
    images = []
    first_question_text = ""
    
    for content_item in user_content:
        if content_item.get('type') == 'image':
            images.append(content_item)
        elif content_item.get('type') == 'text':
            first_question_text = content_item.get('text', '')
    
    # Add images and first question
    first_user_content = images + [{'type': 'text', 'text': first_question_text}]
    chat_conversations.append({
        'role': 'user',
        'content': first_user_content
    })
    
    # Add placeholder for first assistant response (will be generated)
    chat_conversations.append({
        'role': 'assistant',
        'content': [{'type': 'text', 'text': 'PLACEHOLDER_RESPONSE_1'}]
    })
    
    # Add subsequent questions and placeholder responses
    for i, question in enumerate(questions_group[1:], start=2):
        # Extract question text from the user content
        question_conversations = question.get('conversations', [])
        if len(question_conversations) >= 2:
            question_user_msg = question_conversations[1]
            question_content = question_user_msg.get('content', [])
            
            question_text = ""
            for content_item in question_content:
                if content_item.get('type') == 'text':
                    question_text = content_item.get('text', '')
                    break
            
            # Add user question (without images, as they're already included)
            chat_conversations.append({
                'role': 'user',
                'content': [{'type': 'text', 'text': question_text}]
            })
            
            # Add placeholder for assistant response
            chat_conversations.append({
                'role': 'assistant',
                'content': [{'type': 'text', 'text': f'PLACEHOLDER_RESPONSE_{i}'}]
            })
    
    return chat_conversations

def worker_process(args):
    gpu_id, scenario_phase_groups, model_path, backup_dir, debug_limit = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    total_groups = len(scenario_phase_groups)
    process_pbar = tqdm(total=total_groups, position=gpu_id, desc=f"GPU-{gpu_id}")

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
    all_results = []
    
    for (scenario, phase), questions_group in scenario_phase_groups:
        group_id = f"{scenario}_phase_{phase}"
        
        if processed_count < debug_limit:
            print(f"[GPU-{gpu_id}] Processing group {group_id} with {len(questions_group)} questions")
        
        try:
            # Build chat history conversations for this scenario-phase group
            chat_conversations = build_chat_history_conversations(questions_group)
            
            # Process each question in the group sequentially to build chat history
            group_results = []
            current_conversations = chat_conversations.copy()
            
            for question_idx, question in enumerate(questions_group):
                question_id = question.get('question_id', f'unknown_{processed_count}_{question_idx}')
                backup_file_path = os.path.join(backup_dir, f"{question_id}.json")
                
                # Check if already processed
                if os.path.exists(backup_file_path):
                    try:
                        with open(backup_file_path, 'r', encoding='utf-8') as f:
                            existing_data = json.load(f)
                        if 'correct' in existing_data and existing_data.get('correct') != "error":
                            if processed_count < debug_limit:
                                print(f"[GPU-{gpu_id}] Skipping {question_id} - already completed")
                            group_results.append(existing_data)
                            
                            # Update chat history with the existing response
                            if question_idx * 2 + 1 < len(current_conversations):
                                current_conversations[question_idx * 2 + 1]['content'][0]['text'] = existing_data.get('raw_response', 'Unknown response')
                            continue
                    except Exception as e:
                        print(f"[GPU-{gpu_id}] Corrupted backup {question_id}, re-processing. Error: {e}")
                
                # Prepare conversations for current question (up to this point in chat history)
                question_conversations = current_conversations[:question_idx * 2 + 2]  # Include up to current question
                
                # Format conversations for the processor
                reformatted_conversations = []
                for msg in question_conversations:
                    role = msg.get('role')
                    content = msg.get('content')

                    if not role or content is None:
                        raise KeyError("A message is missing 'role' or 'content'.")

                    # Ensure ALL content is a list of dicts
                    if isinstance(content, str):
                        reformatted_msg = {
                            "role": role,
                            "content": [{"type": "text", "text": content}]
                        }
                    elif isinstance(content, list):
                        reformatted_msg = msg
                    else:
                        raise TypeError(f"Unsupported content type: {type(content).__name__}")
                    
                    reformatted_conversations.append(reformatted_msg)

            #     # ==================== DEBUGGING CODE START ====================
            # # Pretty-print the chat history being sent to the model for this specific question
            # if processed_count < debug_limit: # Only print for the groups you're debugging
            #     import json
            #     print("+"*80)
            #     print(f"[DEBUG GPU-{gpu_id}] Processing Question ID: {question_id}")
            #     print(f"[DEBUG GPU-{gpu_id}] Scenario: {scenario}, Phase: {phase}, Question Index in Chat: {question_idx}")
            #     print(f"[DEBUG GPU-{gpu_id}] --- CONVERSATIONAL HISTORY FOR THIS TURN ---")
            #     # The `reformatted_conversations` is exactly what the model will see
            #     print(json.dumps(reformatted_conversations, indent=2))
            #     print("+"*80)
            # # ===================== DEBUGGING CODE END =====================
                # Generate response
                inputs = processor.apply_chat_template(
                    reformatted_conversations, add_generation_prompt=True, 
                    tokenize=True, return_dict=True, return_tensors="pt"
                ).to(model.device, dtype=torch.bfloat16)
                
                if inputs['input_ids'].shape[1] > 8192:
                    raise ValueError(f"Input too long ({inputs['input_ids'].shape[1]} tokens)")
                    
                output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
                response = processor.decode(output_ids[0], skip_special_tokens=True).split("assistant\n")[-1].strip()
                
                answer_choice = extract_answer_choice(response)

                final_result = {
                    "id": question_id,
                    "correct": answer_choice.upper() if answer_choice else "unknown",
                    "raw_response": response,
                    "metadata": {
                        "scenario": question.get('scenario', 'unknown'),
                        "phase_num": question.get('phase_num', 'unknown'),
                        "question_type": question.get('question_type', 'unknown'),
                        "processed_by_gpu": gpu_id,
                        "timestamp": str(datetime.now()),
                        "status": "completed",
                        "question_index_in_chat": question_idx
                    }
                }
                
                group_results.append(final_result)
                
                # Update chat history with the actual response for next questions
                if question_idx * 2 + 1 < len(current_conversations):
                    current_conversations[question_idx * 2 + 1]['content'][0]['text'] = response
                
                # Save individual result
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
            
            all_results.extend(group_results)
            
        except Exception as e:
            print(f"\n!!! [GPU-{gpu_id}] FAILED to process group {group_id}. Error Type: {type(e).__name__}, Message: {e}")
            # Create error results for all questions in the group
            for question in questions_group:
                question_id = question.get('question_id', f'unknown_{processed_count}')
                error_result = {
                    "id": question_id,
                    "correct": "error",
                    "error": str(e),
                    "metadata": {
                        "scenario": question.get('scenario', 'unknown'),
                        "phase_num": question.get('phase_num', 'unknown'),
                        "question_type": question.get('question_type', 'unknown'),
                        "processed_by_gpu": gpu_id,
                        "timestamp": str(datetime.now()),
                        "status": "failed"
                    }
                }
                all_results.append(error_result)
        
        process_pbar.update(1)
        processed_count += 1
        if processed_count % 5 == 0 and torch.cuda.is_available(): 
            torch.cuda.empty_cache()

    process_pbar.close()
    print(f"[GPU-{gpu_id}] Finished processing {processed_count} scenario-phase groups")
    return all_results

def extract_answer_choice(response):
    """
    Extract the answer choice (A, B, C, or D) from the model's response.
    """
    patterns = [
        r'\b([ABCD])\.',  # A., B., C., D.
        r'\b([ABCD])\)',  # A), B), C), D)
        r'\(([ABCD])\)',  # (A), (B), (C), (D)
        r'\b([ABCD])\b',  # Just A, B, C, D as standalone
        r'answer\s*:?\s*([ABCD])',  # "answer: A" or "answer A"
        r'choice\s*:?\s*([ABCD])',  # "choice: A" or "choice A"
        r'option\s*:?\s*([ABCD])',  # "option: A" or "option A"
    ]
    
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

    backup_files = [f for f in os.listdir(backup_dir) if f.endswith('.json')]
    
    for filename in tqdm(backup_files, desc="Aggregating results"):
        file_path = os.path.join(backup_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if "error" in data or data.get("metadata", {}).get("status") == "failed":
                continue
                
            if "correct" not in data or data["correct"] == "error":
                continue

            formatted_entry = {
                "id": data["id"],
                "correct": data["correct"]
            }
            
            final_results.append(formatted_entry)

        except json.JSONDecodeError:
            print(f"Warning: Corrupted JSON file skipped: {file_path}")
        except Exception as e:
            print(f"Warning: Could not process file {file_path}. Error: {e}")

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

    # Group questions by scenario and phase
    grouped_questions = group_questions_by_scenario_phase(tasks)
    print(f"Grouped into {len(grouped_questions)} scenario-phase combinations")
    
    # Print some statistics
    total_questions = sum(len(questions) for questions in grouped_questions.values())
    print(f"Total questions: {total_questions}")
    
    if args.debug_first_n > 0:
        print(f"\n*** Sample scenario-phase groups: ***")
        for i, ((scenario, phase), questions) in enumerate(list(grouped_questions.items())[:3]):
            print(f"  {scenario} phase {phase}: {len(questions)} questions")

    # Create backup directory
    os.makedirs(args.backup_dir, exist_ok=True)
    
    num_gpus = args.num_gpus
    print(f"\nUsing {num_gpus} GPUs for inference.")
    print(f"Backup files will be saved in: {args.backup_dir}")

    if args.debug_first_n > 0:
        print(f"*** DEBUG MODE ON: Detailed output for the first {args.debug_first_n} groups per GPU. ***")

    # Distribute scenario-phase groups across GPUs
    scenario_phase_items = list(grouped_questions.items())
    group_chunks = [[] for _ in range(num_gpus)]
    
    for i, item in enumerate(scenario_phase_items):
        gpu_id = i % num_gpus
        group_chunks[gpu_id].append(item)
    
    # Print distribution info
    for gpu_id in range(num_gpus):
        total_questions_gpu = sum(len(questions) for _, questions in group_chunks[gpu_id])
        print(f"GPU-{gpu_id}: {len(group_chunks[gpu_id])} groups, {total_questions_gpu} total questions")

    # Prepare worker arguments
    worker_args = [
        (gpu_id, group_chunks[gpu_id], args.model_path, args.backup_dir, args.debug_first_n) 
        for gpu_id in range(num_gpus)
    ]
    
    print("\nStarting multiprocessing pool for inference with chat history...")
    with mp.Pool(processes=num_gpus) as pool:
        all_results = pool.map(worker_process, worker_args)
    print("All workers have finished inference.")

    # Aggregate results
    aggregate_results(args.backup_dir, args.output_file)

    print("\nInference with chat history complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 2 Multiple-Choice Question Inference with Chat History")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the merged model directory.")
    parser.add_argument("--test_set_json", type=str, required=True, help="Path to the JSONL file containing test prompts.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the final aggregated output JSON file.")
    parser.add_argument("--backup_dir", type=str, required=True, help="Directory to store per-question JSON backups.")
    parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs to use for parallel processing.")
    parser.add_argument("--debug_first_n", type=int, default=0, help="Print detailed output for the first N groups per GPU.")
    
    args = parser.parse_args()
    main(args)