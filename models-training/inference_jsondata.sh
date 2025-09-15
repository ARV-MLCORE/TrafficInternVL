python inference_final.py \
    --model_path ./model_repository/InternVL-38B/exp3/InternVL-38B-AICity-Simple-Merged/ \
    --test_set_json /workspace/processed_data/wts_dataset_test.jsonl \
    --output_file /workspace/Park/LLaMA-Factory/result_backup/exp3/final_aggregated_results.json \
    --num_gpus 6 \
    --backup_dir /workspace/Park/LLaMA-Factory/result_backup/exp3 \
    > full_log.txt 2>&1