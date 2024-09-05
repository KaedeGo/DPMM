CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1 python fusion_main.py \
--dim 256 --dropout 0.3 --layers 2 \
--vision_backbone resnet34 \
--mode train \
--epochs 200 --batch_size 32 --lr 0.00007 \
--vision_num_classes 25 --num_classes 25 \
--data_pairs partial_ehr_cxr \
--data_ratio 1.0 \
--task phenotyping_48h \
--labels_set pheno \
--fusion_type unified \
--normalizer_state /disk1/fwu/myProjects/MedFuse/data/pheno_ts.normalizer \
--save_dir checkpoints/phenotyping/partial/unified

# nohup sh scripts/phenotyping/train/partial/unified.sh > logs/phe_mimic4/partial/unified.log 2>&1 &
