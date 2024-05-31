import torch
import torch.nn as nn

class Downsampler(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.upsampling = torch.nn.Linear(in_dim, out_dim)

    def forward(self, pred_vertices_sub):
        pred_vertices = self.upsampling(pred_vertices_sub)
        return pred_vertices


class RegressHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv1d(input_dim, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1, 1),
        )

    def forward(self, x):
        dist = self.network(x).permute(0, 2, 1)[:, :, 0]
        return dist

