CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 python fusion_main.py \
--dim 256 --dropout 0.3 --layers 2 \
--vision_backbone resnet34 \
--mode train \
--epochs 200 --batch_size 32 --lr 0.00007 \
--vision_num_classes 25 --num_classes 25 \
--data_pairs partial_ehr_cxr \
--data_ratio 1.0 \
--task phenotyping_48h \
--labels_set pheno \
--fusion_type copula \
--copula_fuse_type lstm \
--copula 0.00005 \
--copula_resample False \
--normalizer_state /disk1/fwu/myProjects/MedFuse/data/pheno_ts.normalizer \
--save_dir checkpoints/phenotyping/abalation/copula_partial_wo_resample.5

# nohup sh scripts/phenotyping/train/abalation/copula_partial_wo_resample.sh > logs/phe_mimic4/abalation/copula_partial_wo_resample.log 2>&1 &
