CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1 python fusion_main_mimic3.py \
--dim 256 --dropout 0.3 --layers 2 \
--mode train \
--epochs 100 --batch_size 8 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs paired_ehr_note \
--task in-hospital-mortality \
--fusion_type drfuse --layer_after 4 \
--labels_set mortality \
--normalizer_state /disk1/fwu/myProjects/MedFuse/data_mimic3/ihm_ts.normalizer \
--save_dir checkpoints/mortality/mimic3/paired/drfuse \
--ehr_data_dir /disk1/fwu/myProjects/MedFuse/data_mimic3

# nohup sh scripts/mortality/mimic3/paired/drfuse.sh > logs/ihm_mimic3/paired/drfuse.log 2>&1 &