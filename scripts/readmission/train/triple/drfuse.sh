CUDA_VISIBLE_DEVICES=4 CUDA_LAUNCH_BLOCKING=1 python fusion_main_3d.py \
--dim 256 --dropout 0.3 --layers 2 \
--mode train \
--epochs 100 --batch_size 16 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs paired_ehr_cxr \
--task readmission \
--fusion_type drfuse --layer_after 4 \
--labels_set readm \
--normalizer_state /data1/fwu/myProjects/MedFuse/data/readm_ts.normalizer \
--save_dir checkpoints/readm/triple/drfuse
# nohup sh scripts/readmission/train/triple/drfuse.sh > logs/readm_mimic4/triple/drfuse.log 2>&1 &