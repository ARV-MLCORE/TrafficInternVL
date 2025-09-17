#!/bin/bash

num_worker=32
root="/home/deepzoom/arv-aicity2-data/Park/AICITY2024_Track2_AliOpenTrek_CityLLaVA/data_preprocess/data"
save_folder="/home/deepzoom/TrafficInternVL/data-preparation/task1/processed_anno" # Store json files 
splits=("train" "val")
scale=1.5

# Activate llava-env
source env/bin/activate

for split in "${splits[@]}"; do
    python3 extract_wts_frame_bbox_anno.py --root $root --save-folder $save_folder/frame_bbox_anno --split $split
    python3 extract_bdd_frame_bbox_anno.py --root $root --save-folder $save_folder/frame_bbox_anno --split $split
done

for file in "$save_folder/frame_bbox_anno"/*train*; do
    python3 draw_bbox_on_frame.py --worker $num_worker --anno $file --scale $scale
done

for file in "$save_folder/frame_bbox_anno"/*val*; do
    python3 draw_bbox_on_frame.py --worker $num_worker --anno $file --scale $scale
done

for split in "${splits[@]}"; do
    python3 transform_format.py \
        --root $root \
        --save-folder $save_folder/internvl_format \
        --split $split \
        --wts-global-image-path $root/WTS/bbox_local \
        --bdd-global-image-path $root/BDD_PC_5k/bbox_local \
        --output-tag local
done

# generate shortQA
# API_KEY="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"
# MODEL="Openai"

# python shortQA_split.py --model $MODEL --api-key $API_KEY
# python shortQA_merge.py

# # data filter
# python add_stage_prompt.py
# python filiter_data_by_area.py
# python check_image.py

# echo " Trainsets prepared."