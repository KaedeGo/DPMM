CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python fusion_main.py \
--dim 256 --dropout 0 --layers 2 \
--vision_backbone resnet34 \
--mode train \
--epochs 100 --batch_size 32 --lr 8e-05 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs partial_ehr_cxr \
--data_ratio 1.0 \
--task in-hospital-mortality \
--labels_set mortality \
--fusion_type copula \
--copula_fuse_type na \
--normalizer_state /disk1/fwu/myProjects/MedFuse/data/ihm_ts.normalizer \
--save_dir checkpoints/mortality/abalation/copula_partial_wo_fusion
# nohup sh scripts/mortality/train/abalation/copula_partial_wo_fusion.sh > logs/ihm_mimic4/abalation/copula_partial_wo_fusion.log 2>&1 &