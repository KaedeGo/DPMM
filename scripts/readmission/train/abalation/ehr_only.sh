CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1 python fusion_main.py \
--dim 256 --dropout 0.3 --layers 2 \
--vision_backbone resnet34 \
--mode train \
--epochs 100 --batch_size 32 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs partial_ehr \
--fusion_type uni_ehr \
--task readmission \
--labels_set readm \
--save_dir checkpoints/readm/abalation/ehr_only \
--normalizer_state /disk1/fwu/myProjects/MedFuse/data/readm_ts.normalizer 

# nohup sh scripts/readmission/train/abalation/ehr_only.sh > logs/readm_mimic4/abalation/ehr_only.log 2>&1 &