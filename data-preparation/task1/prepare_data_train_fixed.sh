#!/bin/bash

num_worker=32
root="/home/deepzoom/arv-aicity2-data/Park/AICITY2024_Track2_AliOpenTrek_CityLLaVA/data_preprocess/data/"
save_folder="/home/deepzoom/arv-aicity2-data/final_repo_for_github/data/processed_anno/" # Store json files 
splits=("train" "val")
scale=1.5

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
    python3 transform_llava_format.py \
        --root $root \
        --save-folder $save_folder/llava_format \
        --split $split \
        --wts-global-image-path $root/WTS/bbox_global \
        --bdd-global-image-path $root/BDD_PC_5k/bbox_global \
        --output-tag global
done

# generate shortQA
# API_KEY="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"
# MODEL="Openai"

# python3 shortQA_split.py --model $MODEL --api-key $API_KEY
# python3 shortQA_merge.py

# # data filter
# python3 add_stage_prompt.py
# python3 filiter_data_by_area.py
# python3 check_image.py

# echo " Trainsets prepared."