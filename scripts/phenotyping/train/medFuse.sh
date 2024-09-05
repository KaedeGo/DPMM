CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1 python fusion_main.py \
--dim 256 --dropout 0.3 --layers 2 \
--vision_backbone resnet34 \
--mode train \
--epochs 200 --batch_size 32 --lr 0.00007 \
--vision_num_classes 25 --num_classes 25 \
--data_pairs paired_ehr_cxr \
--task phenotyping \
--labels_set pheno \
--fusion_type lstm \
--normalizer_state /disk1/fwu/myProjects/MedFuse/data/pheno_ts.normalizer \
--save_dir checkpoints/phenotyping/medFuse_paired

# nohup sh scripts/phenotyping/train/medFuse.sh > phe_medFuse_paired.log 2>&1 &