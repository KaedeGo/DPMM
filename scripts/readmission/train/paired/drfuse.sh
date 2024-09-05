CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1 python fusion_main.py \
--dim 256 --dropout 0.3 --layers 2 \
--mode train \
--epochs 100 --batch_size 32 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs paired_ehr_cxr \
--task readmission \
--fusion_type drfuse --layer_after 4 \
--labels_set readm \
--normalizer_state /disk1/fwu/myProjects/MedFuse/data/readm_ts.normalizer \
--save_dir checkpoints/readm/paired/drfuse
# nohup sh scripts/readmission/train/paired/drfuse.sh > logs/readm_mimic4/paired/drfuse.log 2>&1 &