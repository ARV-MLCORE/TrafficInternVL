
# Disable conflicting tokenizer parallelism and reduce the number of workers
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
    --preprocessing_num_workers 12 \
    --stage 'sft' \
    --do_train true \
    --model_name_or_path "OpenGVLab/InternVL3-8B-hf" \
    --trust_remote_code \
    --finetuning_type lora \
    --lora_rank 64 \
    --lora_target all \
    --dataset aicity_local_dataset \
    --template intern_vl \
    --cutoff_len 4096 \
    --output_dir model/InternVL-38B/r8-task1-github-test \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 100 \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --bf16