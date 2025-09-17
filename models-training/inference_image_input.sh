python3 inference_image_input.py \
    --model_path /home/deepzoom/TrafficInternVL/models-training/model_repository/InternVL-8B_task1/github-test/InternVL-8B-AICity-Simple-Merged/ \
    --test_data_dir /home/deepzoom/TrafficInternVL/data-preparation/task1/data/generate_test_frames/bbox_local \
    --output_file /home/deepzoom/TrafficInternVL/result_backup/test_forgithub/final_aggregated_results.json \
    --num_gpus 1 \
    --backup_dir /home/deepzoom/TrafficInternVL/LLaMA-Factory/result_backup_forgithub/exp1 \