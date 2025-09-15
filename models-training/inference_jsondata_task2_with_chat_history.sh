#!/bin/bash
python /workspace/Park/LLaMA-Factory/inference_json_input_task2_with_chat_history.py \
    --model_path /workspace/Park/LLaMA-Factory/model_repository/InternVL-38B_task2/exp2/InternVL-38B-AICity-Simple-Merged/ \
    --test_set_json /workspace/processed_data_subtask2_best_view/wts_dataset_test_subtask2_best_view.jsonl\
    --output_file /workspace/Park/LLaMA-Factory/result_backup/task2/exp2_task2_with_history/final_aggregated_results.json \
    --num_gpus 6 \
    --backup_dir /workspace/Park/LLaMA-Factory/result_backup/task2/exp2_task2_with_history \
    > full_log_task2_with_history.txt 2>&1