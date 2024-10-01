CUDA_VISIBLE_DEVICES=3 CUDA_LAUNCH_BLOCKING=1 python fusion_main.py \
--dim 256 --dropout 0 --layers 2 \
--vision_backbone resnet34 \
--mode train \
--epochs 100 --batch_size 32 --lr 1e-04 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs paired_ehr_cxr \
--data_ratio 1.0 \
--task in-hospital-mortality \
--labels_set mortality \
--fusion_type copula \
--copula 1 \
--copula_fuse_type lstm \
--replace_w_align kl \
--normalizer_state /disk1/fwu/myProjects/MedFuse/data/ihm_ts.normalizer \
--save_dir checkpoints/mortality/abalation/copula_paired_w_align_kl
# nohup sh scripts/mortality/train/abalation/copula_paired_w_align.sh > logs/ihm_mimic4/abalation/copula_paired_w_kl.log 2>&1 &