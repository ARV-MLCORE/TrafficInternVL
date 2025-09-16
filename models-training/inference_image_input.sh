python3 inference_image_input.py \
    --model_path /home/deepzoom/arv-aicity2-data/Park/LLaMA-Factory/model_repository/InternVL-38B/exp4/InternVL-38B-AICity-Simple-Merged/ \
    --test_data_dir /home/deepzoom/TrafficInternVL/data-preparation/task1/data/generate_test_frames/bbox_local \
    --output_file ./result_backup/test_forgithub/final_aggregated_results.json \
    --num_gpus 1 \
    --backup_dir ./LLaMA-Factory/result_backup_forgithub/exp4 \