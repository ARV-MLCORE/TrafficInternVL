CUDA_VISIBLE_DEVICES=0 llamafactory-cli export \
    --model_name_or_path "OpenGVLab/InternVL3-8B-hf" \
    --trust_remote_code \
    --adapter_name_or_path ./model/InternVL-8B/r8-task1-github-test/ \
    --export_dir ./model_repository/InternVL-8B_task1/github-test/InternVL-8B-AICity-Simple-Merged \
    --export_size 40 \
    --export_device cpu