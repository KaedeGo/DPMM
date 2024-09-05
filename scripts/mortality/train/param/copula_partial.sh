CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 python fusion_main.py \
--dim 256 --dropout 0.3 --layers 2 \
--vision_backbone resnet34 \
--mode train \
--epochs 100 --batch_size 32 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs partial_ehr_cxr \
--data_ratio 1.0 \
--task in-hospital-mortality \
--labels_set mortality \
--fusion_type copula \
--copula_fuse_type lstm \
--copula 0.00001 \
--normalizer_state /disk1/fwu/myProjects/MedFuse/data/ihm_ts.normalizer \
--save_dir checkpoints/mortality/partial/copula1
# nohup sh scripts/mortality/train/param/copula_partial.sh > logs/ihm_mimic4/partial/copula1.log 2>&1 &