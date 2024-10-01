CUDA_VISIBLE_DEVICES=3 CUDA_LAUNCH_BLOCKING=1 python fusion_main.py \
--dim 256 --dropout 0 --layers 2 \
--vision_backbone resnet34 \
--mode train \
--epochs 100 --batch_size 32 --lr 1e-04 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs paired_ehr_cxr \
--data_ratio 1.0 \
--task readmission \
--labels_set readm \
--fusion_type copula \
--copula_fuse_type lstm \
--copula 1 \
--replace_w_align kl \
--normalizer_state /disk1/fwu/myProjects/MedFuse/data/readm_ts.normalizer \
--save_dir checkpoints/readm/abalation/copula_paired_w_kl
# nohup sh scripts/readmission/train/abalation/copula_paired_w_align.sh > logs/readm_mimic4/abalation/copula_paired_w_kl.log 2>&1 &