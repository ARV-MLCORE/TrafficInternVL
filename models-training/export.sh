CUDA_VISIBLE_DEVICES=0 llamafactory-cli export \
    --model_name_or_path "OpenGVLab/InternVL3-38B-hf" \
    --trust_remote_code \
    --adapter_name_or_path ./model/InternVL-38B/r8-task2-exp2/ \
    --export_dir ./model_repository/InternVL-38B_task2/exp2/InternVL-38B-AICity-Simple-Merged \
    --export_size 40 \
    --export_device cpu