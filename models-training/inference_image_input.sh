python inference_image_input.py \
    --model_path ./model_repository/InternVL-38B/exp4/InternVL-38B-AICity-Simple-Merged/ \
    --test_data_dir /workspace/Park/AICITY2024_Track2_AliOpenTrek_CityLLaVA/data_preprocess/data/generate_test_frames/bbox_local \
    --output_file /workspace/Park/LLaMA-Factory/result_backup/exp4/final_aggregated_results.json \
    --num_gpus 4 \
    --backup_dir /workspace/Park/LLaMA-Factory/result_backup/exp4 \