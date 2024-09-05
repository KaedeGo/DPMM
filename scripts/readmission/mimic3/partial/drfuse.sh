CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1 python fusion_main_mimic3.py \
--dim 256 --dropout 0.3 --layers 2 \
--mode train \
--epochs 100 --batch_size 8 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs partial_ehr_note \
--task readmission \
--fusion_type drfuse --layer_after 4 \
--labels_set readm \
--normalizer_state /disk1/fwu/myProjects/MedFuse/data_mimic3/readm_ts.normalizer \
--ehr_data_dir /disk1/fwu/myProjects/MedFuse/data_mimic3 \
--save_dir checkpoints/readm/mimic3/partial/drfuse

# nohup sh scripts/readmission/mimic3/partial/drfuse.sh > logs/readm_mimic3/partial/drfuse.log 2>&1 &