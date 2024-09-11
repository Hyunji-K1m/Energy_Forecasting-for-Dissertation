import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from linear_attention_transformer import LinearAttentionTransformer
import yaml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def load_config():
    with open('/content/drive/Othercomputers/My MacBook Air/Desktop/Dissertation/code/real code/CSDI_final/config/base_forecasting.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config
config = load_config()
print(config)

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

def get_linear_trans(heads=8, layers=1, channels=64, localheads=0, localwindow=0):
    return LinearAttentionTransformer(
        dim=channels,
        depth=layers,
        heads=heads,
        max_seq_len=395,  
        n_local_attn_heads=localheads, 
        local_attn_window_size=localwindow,
    )

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        projection_dim = projection_dim or embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim // 2).to(torch.float32),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = F.silu(self.projection1(x))
        x = F.silu(self.projection2(x))
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1).float()  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim).float() / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        return torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)


class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = 64

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=70,  
            embedding_dim=64,  
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        print("self.input_projection", self.input_projection)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=64,  
                    channels=self.channels,
                    diffusion_embedding_dim=64,  
                    nheads=4,  
                    is_linear=True  
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info, diffusion_step):
        B, inputdim, K, L = x.shape

        x = F.relu(self.input_projection(x.reshape(B, inputdim, K * L)))
        #print(self.input_projection)
        x = x.reshape(B, self.channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = F.relu(self.output_projection1(x.reshape(B, self.channels, K * L)))
        return self.output_projection2(x).reshape(B, K, L)


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, is_linear=False):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.is_linear = is_linear
        self.time_layer = get_linear_trans(heads=nheads, layers=1, channels=channels) if is_linear else get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_linear_trans(heads=nheads, layers=1, channels=channels) if is_linear else get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.cond_downsample = Conv1d_with_init(329, 64, 1)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(0, 2, 1) if self.is_linear else y.permute(2, 0, 1)).permute(1, 2, 0)
        return y.reshape(B, channel, K * L)

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(0, 2, 1) if self.is_linear else y.permute(2, 0, 1)).permute(1, 2, 0)
        return y.reshape(B, channel, K * L)

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, cond_dim, _, _ = cond_info.shape
        #print("cond_info shape", cond_info.shape)
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_downsample(cond_info)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip
