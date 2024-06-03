import torch
import torch.nn as nn
from torch.nn import functional as F


def init_weights(m):
    try:
        if type(m) == nn.ConvTranspose2d:
            nn.init.normal_(m.weight,std=0.001)
        elif type(m) == nn.Conv2d:
            nn.init.normal_(m.weight,std=0.001)
            nn.init.constant_(m.bias, 0)
        elif type(m) == nn.BatchNorm2d:
            nn.init.constant_(m.weight,1)
            nn.init.constant_(m.bias,0)
        elif type(m) == nn.Linear:
            nn.init.normal_(m.weight,std=0.01)
            nn.init.constant_(m.bias,0)
    except AttributeError:
        pass


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=2048+3, nhead=1, dim_feedforward=2048+3, kdim=256+3+3, vdim=256+3+3, dropout=0.0, activation="gelu"):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim_feedforward, nhead, dropout=dropout, kdim=kdim, vdim=vdim)
        self.linear = nn.Linear(d_model, dim_feedforward)
        self.linear1 = nn.Linear(dim_feedforward, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim_feedforward)
        self.norm1 = nn.LayerNorm(dim_feedforward)
        self.norm2 = nn.LayerNorm(dim_feedforward)
        self.activation = F.gelu

    def forward(self, src_q:torch.Tensor, src_k:torch.Tensor, src_v:torch.Tensor)->torch.Tensor:
        src_q, src_k, src_v = src_q.permute(1,0,2), src_k.permute(1,0,2), src_v.permute(1,0,2)
        src = src_q
        src2, _ = self.cross_attn(src_q, src_k, src_v)

        src = src + self.dropout(src2)
        src = self.norm(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src.permute(1,0,2)


class CRFormer(nn.Module):
    def __init__(self, dim=256+3):
        super().__init__()
        self.SA_Transformer_r, self.SA_Transformer_l, self.SA_Transformer_obj = [], [], []
        self.CA_Transformer_r, self.CA_Transformer_l, self.CA_Transformer_obj = [], [], []
        self.Transformer_r, self.Transformer_l, self.Transformer_obj = [], [], []
        self.num_layers = 4
        # 其中256可能是特征维度，加上额外的3维可能是因为加入了其他信息（如位置编码
        self.dim = dim

        transformers = [self.SA_Transformer_r, self.SA_Transformer_l, self.SA_Transformer_obj, 
                        # self.CA_Transformer_r, self.CA_Transformer_l, self.CA_Transformer_obj,
                        self.Transformer_r, self.Transformer_l, self.Transformer_obj]
        ca_transformers = [self.CA_Transformer_r, self.CA_Transformer_l, self.CA_Transformer_obj]
        
        for _ in range(self.num_layers):
            feat_dim = self.dim
            for transformer in transformers:
                transformer.append(TransformerEncoderLayer(d_model=feat_dim, nhead=1, dim_feedforward=feat_dim, kdim=feat_dim, vdim=feat_dim))
            feat_dim = feat_dim + 1
            for transformer in ca_transformers:
                transformer.append(TransformerEncoderLayer(d_model=feat_dim, nhead=1, dim_feedforward=feat_dim, kdim=feat_dim, vdim=feat_dim))
        #在__init__方法中创建的Transformer层列表转换为nn.ModuleList。这样做的好处是，nn.ModuleList可以方便地管理模型的子模块，使它们在训练过程中能够正确地更新权重。      
        self.SA_Transformer_r = nn.ModuleList(self.SA_Transformer_r)
        self.SA_Transformer_l = nn.ModuleList(self.SA_Transformer_l)
        self.SA_Transformer_obj = nn.ModuleList(self.SA_Transformer_obj)

        self.CA_Transformer_r = nn.ModuleList(self.CA_Transformer_r)
        self.CA_Transformer_l = nn.ModuleList(self.CA_Transformer_l)
        self.CA_Transformer_obj = nn.ModuleList(self.CA_Transformer_obj)

        self.Transformer_r = nn.ModuleList(self.Transformer_r)
        self.Transformer_l = nn.ModuleList(self.Transformer_l)
        self.Transformer_obj = nn.ModuleList(self.Transformer_obj)

    def init_weights(self):
        self.apply(init_weights)
        
    def forward(self, pts_r_feat, pts_l_feat, pts_obj_feat, dist_ro, dist_lo, dist_or, dist_ol):
        # pts [Batch_size, Feats, PointFSs]; dist [Batch_size, PointsDist]
        pts_r_feat = pts_r_feat.transpose(1,2)
        pts_l_feat = pts_l_feat.transpose(1,2)
        pts_obj_feat = pts_obj_feat.transpose(1,2)
        
        dist_ro = dist_ro.unsqueeze(1).transpose(1,2)
        dist_lo = dist_lo.unsqueeze(1).transpose(1,2)
        dist_or = dist_or.unsqueeze(1).transpose(1,2)
        dist_ol = dist_ol.unsqueeze(1).transpose(1,2)

        feat_r = pts_r_feat.clone()
        feat_l = pts_l_feat.clone()
        feat_o = pts_obj_feat.clone()

        features_ro = torch.cat([pts_r_feat, dist_ro], dim=2)
        features_lo = torch.cat([pts_l_feat, dist_lo], dim=2)
        features_or = torch.cat([pts_obj_feat, dist_or], dim=2)
        features_ol = torch.cat([pts_obj_feat, dist_ol], dim=2)

        for i in range(self.num_layers):
            feat_r = self.SA_Transformer_r[i](feat_r, feat_r, feat_r)
            feat_l = self.SA_Transformer_l[i](feat_l, feat_l, feat_l)
            feat_o = self.SA_Transformer_obj[i](feat_o, feat_o, feat_o)

        for i in range(self.num_layers):
            features_ro_ = self.CA_Transformer_r[i](features_ro, features_or, features_or)
            features_lo_ = self.CA_Transformer_l[i](features_lo, features_ol, features_ol)
            features_or_ = self.CA_Transformer_obj[i](features_or, features_ro, features_ro)
            features_ol_ = self.CA_Transformer_obj[i](features_ol, features_lo, features_lo)

            features_ro, features_lo, features_or, features_ol = features_ro_ , features_lo_, features_or_, features_ol_
        
        out_feat_r = feat_r + features_ro[:,:,:-1]
        out_feat_l = feat_l + features_lo[:,:,:-1]
        out_feat_o = feat_o + features_or[:,:,:-1] + features_ol[:,:,:-1]

        out_feat_r = out_feat_r.transpose(1,2)
        out_feat_l = out_feat_l.transpose(1,2)
        out_feat_o = out_feat_o.transpose(1,2)

        return out_feat_r, out_feat_l, out_feat_o