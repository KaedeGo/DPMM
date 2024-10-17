import torch.nn as nn
import torch
import numpy as np
from models.loss import CosineLoss, KLDivLoss, DirichletProcessLoss
from models.multimodal_fusion import MultimodalFusion, CrossAttentionFusion
import torch.nn.functional as F


class DP_Fusion(nn.Module):
    def __init__(self, args, ehr_model, note_model):

        super(DP_Fusion, self).__init__()
        self.args = args
        self.ehr_model = ehr_model
        self.note_model = note_model

        target_classes = self.args.num_classes
        lstm_in = self.ehr_model.feats_dim
        lstm_out = self.note_model.feats_dim
        projection_in = self.note_model.feats_dim

        self.projection = nn.Linear(projection_in, lstm_in)
        feats_dim = 2 * self.ehr_model.feats_dim

        self.fused_cls = nn.Sequential(
            nn.Linear(feats_dim, self.args.num_classes), nn.Sigmoid()
        )

        self.align_loss = CosineLoss()
        self.kl_loss = KLDivLoss()
        self.dp_loss = DirichletProcessLoss(K=args.K, M=2, rho_scale=args.rho_scale)

        self.lstm_fused_cls = nn.Sequential(
            nn.Linear(lstm_out, target_classes), nn.Sigmoid()
        )

        self.lstm_fusion_layer = nn.LSTM(
            lstm_in, lstm_out, batch_first=True, dropout=0.0
        )

        self.cross_attention_fusion = CrossAttentionFusion(in_ts_size=lstm_in, in_cxr_size=lstm_in)

        self.mha_fused_cls = nn.Sequential(
            nn.Linear(feats_dim, target_classes),
            nn.Sigmoid()
        )

    def forward(self, x, seq_lengths=None, token=None, mask=None, pairs=None):

        ehr_preds, ehr_feats = self.ehr_model(x, seq_lengths)
        note_preds, _, note_feats = self.note_model(token, mask)
        projected = self.projection(note_feats)

        # normalize the ehr_feats&note_feats
        if self.args.dp_normalize_feats:
            ehr_feats = F.normalize(ehr_feats, p=2, dim=1)
            projected = F.normalize(projected, p=2, dim=1)

        if self.args.dp_resample:
            n_samples = len(projected[list(~np.array(pairs))])
            if n_samples > 0:
                note_samples = self.dp_loss.rsample(
                    n_samples=torch.zeros(n_samples).size()
                )
                projected[list(~np.array(pairs))] = note_samples.detach()
        else:
            projected[list(~np.array(pairs))] = 0

        if self.args.replace_w_align:
            dp_loss = self.align_loss(ehr_feats, projected)
        else:
            dp_loss = self.dp_loss(ehr_feats, projected)

        if self.args.dp_fuse_type == "lstm":
            if len(ehr_feats.shape) == 1:
                feats = ehr_feats[None, None, :]
                feats = torch.cat([feats, projected[:, None, :]], dim=1)
            else:
                feats = ehr_feats[:, None, :]
                feats = torch.cat([feats, projected[:, None, :]], dim=1)
            seq_lengths = np.array([1] * len(seq_lengths))
            seq_lengths[pairs] = 2

            feats = torch.nn.utils.rnn.pack_padded_sequence(
                feats, seq_lengths, batch_first=True, enforce_sorted=False
            )

            x, (ht, _) = self.lstm_fusion_layer(feats)

            out = ht.squeeze(0)

            fused_preds = self.lstm_fused_cls(out)
        elif self.args.dp_fuse_type == 'mha':
            if len(ehr_feats.shape) == 1:
                feats = ehr_feats[None,:]
            fusion_feat = self.cross_attention_fusion(ehr_feats, projected)
            fused_preds = self.mha_fused_cls(fusion_feat)
        else:
            feats = torch.cat([ehr_feats, projected], dim=1)
            fused_preds = self.fused_cls(feats)

        return {
            "ehr_feats": ehr_feats,
            "note_feats": projected,
            "dp": fused_preds,
            "dp_loss": dp_loss,
        }
