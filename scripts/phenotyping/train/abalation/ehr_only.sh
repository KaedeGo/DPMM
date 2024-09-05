CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 python fusion_main.py \
--dim 256 --dropout 0.3 --layers 2 \
--vision_backbone resnet34 \
--mode train \
--epochs 200 --batch_size 32 --lr 0.00053985 \
--vision_num_classes 25 --num_classes 25 \
--task phenotyping_48h \
--labels_set pheno \
--data_pairs partial_ehr \
--fusion_type uni_ehr \
--normalizer_state /disk1/fwu/myProjects/MedFuse/data/pheno_ts.normalizer \
--save_dir checkpoints/phenotyping/abalation/ehr_only 

# nohup sh scripts/phenotyping/train/abalation/ehr_only.sh > logs/phe_mimic4/abalation/ehr_only.log 2>&1 &
