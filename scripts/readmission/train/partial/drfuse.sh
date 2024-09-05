CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1 python fusion_main.py \
--dim 256 --dropout 0.3 --layers 2 \
--mode train \
--epochs 100 --batch_size 32 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs partial_ehr_cxr \
--task readmission \
--fusion_type drfuse --layer_after 4 \
--labels_set readm \
--normalizer_state /disk1/fwu/myProjects/MedFuse/data/readm_ts.normalizer \
--save_dir checkpoints/readm/partial/drfuse

# nohup sh scripts/readmission/train/partial/drfuse.sh > logs/readm_mimic4/partial/drfuse.log 2>&1 &