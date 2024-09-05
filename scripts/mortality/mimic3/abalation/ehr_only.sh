CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1 python fusion_main_mimic3.py \
--dim 256 --dropout 0.3 --layers 2 \
--vision_backbone resnet34 \
--mode train \
--epochs 100 --batch_size 16 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs partial_ehr \
--fusion_type uni_ehr \
--task in-hospital-mortality \
--labels_set mortality \
--save_dir checkpoints/mortality/mimic3/abalation/ehr_only \
--normalizer_state /disk1/fwu/myProjects/MedFuse/data_mimic3/ihm_ts.normalizer \
--ehr_data_dir /disk1/fwu/myProjects/MedFuse/data_mimic3 
# nohup sh scripts/mortality/mimic3/abalation/ehr_only.sh > logs/ihm_mimic3/abalation/ehr_only.log 2>&1 &