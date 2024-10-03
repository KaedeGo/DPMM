from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import argparse
import os
import imp
import re
from trainers.fusion_trainer import FusionTrainer
from trainers.mmtm_trainer import MMTMTrainer
from trainers.daft_trainer import DAFTTrainer
from trainers.drfuse_trainer import DrFuseTrainer
from trainers.copula_trainer import CopulaTrainer

from ehr_utils.preprocessing import Discretizer, Normalizer
from datasets_mf.ehr_dataset import get_datasets
from datasets_mf.cxr_dataset import get_cxr_datasets
from datasets_mf.fusion import load_cxr_ehr
from pathlib import Path
from paths import *
import torch

import wandb

from arguments import args_parser

parser = args_parser()
# add more arguments here ...
args = parser.parse_args()
print(args)

if args.missing_token is not None:
    from trainers.fusion_tokens_trainer import FusionTokensTrainer as FusionTrainer

seed = 1002
torch.manual_seed(seed)
np.random.seed(seed)

def read_timeseries(args):
    path = f'{args.ehr_data_dir}/{args.task}/train/14991576_episode3_timeseries.csv'
    ret = []
    with open(path, "r") as tsfile:
        header = tsfile.readline().strip().split(',')
        assert header[0] == "Hours"
        for line in tsfile:
            mas = line.strip().split(',')
            ret.append(np.array(mas))
    return np.stack(ret)
    

discretizer = Discretizer(timestep=float(args.timestep),
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')


discretizer_header = discretizer.transform(read_timeseries(args))[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
normalizer_state = args.normalizer_state
if normalizer_state is None:
    normalizer_state = 'normalizers/ph_ts{}.input_str:previous.start_time:zero.normalizer'.format(args.timestep)
    normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
normalizer.load_params(normalizer_state)

ehr_train_ds, ehr_val_ds, ehr_test_ds = get_datasets(discretizer, normalizer, args)

cxr_train_ds, cxr_val_ds, cxr_test_ds = get_cxr_datasets(args)

train_dl, val_dl, test_dl = load_cxr_ehr(args, ehr_train_ds, ehr_val_ds, cxr_train_ds, cxr_val_ds, ehr_test_ds, cxr_test_ds)

# for lr in [0.0001, 0.0005, 0.001]:
    # for dr in [0, 0.05, 0.1, 0.2, 0.3, 0.4]:
# for temperature in [0.001, 0.01]:
for rho_scale in [-2.5, -3, -4]:
    for K in [2, 3, 5]:
                if "partial" in args.data_pairs:
                    pair_type = "partial"
                else:
                    pair_type = "paired"

                if "mimic3" in args.ehr_data_dir:
                    dataset_name = "mimic3"
                else:
                    dataset_name = "mimic4"

                # args.lr = lr
                # args.dropout = dr
                args.K = K
                args.rho_scale = rho_scale
                args.temperature = temperature

                name = f"{dataset_name}_{args.fusion_type}_{pair_type}_{args.labels_set}_{args.copula_family}_rho{rho_scale}_K{K}_temp{args.temperature}"
                # name = f"{args.fusion_type}_{pair_type}_{args.labels_set}_dr{dr}"
                # name = f"{args.fusion_type}_{pair_type}_{args.labels_set}_lr{lr}"
                # name = f"{args.fusion_type}_{pair_type}_{args.labels_set}_{args.copula_family}_temp{args.temperature}"

                args.save_dir = f"checkpoints/phenotyping/paired/copula/{name}"
                path = Path(args.save_dir)
                path.mkdir(parents=True, exist_ok=True)

                config = vars(args)
                mode = "online"

                wandb.init(name=name,
                           project='MedFuse',
                           notes="",
                           mode=mode,
                           config=config,
                           tags=["copula", args.copula_family, pair_type]
                )

                with open(f"{args.save_dir}/args.txt", 'w') as results_file:
                    for arg in vars(args):
                        print(f"  {arg:<40}: {getattr(args, arg)}")
                        results_file.write(f"  {arg:<40}: {getattr(args, arg)}\n")


                if args.fusion_type == 'mmtm':
                    trainer = MMTMTrainer(
                        train_dl,
                        val_dl,
                        args,
                        test_dl=test_dl
                        )
                elif args.fusion_type == 'daft':
                        trainer = DAFTTrainer(train_dl,
                        val_dl,
                        args,
                        test_dl=test_dl)
                elif args.fusion_type == 'drfuse':
                    trainer = DrFuseTrainer(
                        train_dl,
                        val_dl,
                        args,
                        test_dl=test_dl
                        )
                elif args.fusion_type == 'copula':
                    trainer = CopulaTrainer(
                        train_dl,
                        val_dl,
                        args,
                        test_dl=test_dl
                        )
                else:
                    trainer = FusionTrainer(
                        train_dl,
                        val_dl,
                        args,
                        test_dl=test_dl
                        )

                if args.mode == 'train':
                    print("==> training")

                    trainer.train()
                    trainer.args.load_state = args.save_dir + '/best_checkpoint.pth.tar'
                    trainer.load_state()
                    trainer.eval()

                    wandb.finish()
                    args.load_state = None

                elif args.mode == 'eval':
                    trainer.eval()
                else:
                    raise ValueError("not Implementation for args.mode")
