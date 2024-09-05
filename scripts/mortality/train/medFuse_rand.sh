CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1 python fusion_main.py \
--dim 256 --dropout 0.3 --layers 2 \
--vision_backbone resnet34 \
--copula 0 \
--mode train \
--epochs 100 --batch_size 32 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs paired_ehr_cxr \
--data_ratio 1.0 \
--task in-hospital-mortality \
--labels_set mortality \
--fusion_type lstm \
--normalizer_state /disk1/fwu/myProjects/MedFuse/data/ihm_ts.normalizer \
--save_dir checkpoints/mortality/medFuseRand_paired

# --lr 9.382e-05 \
# nohup sh scripts/mortality/train/medFuse_rand.sh > ihm_medFuseRand_paired.log 2>&1 &