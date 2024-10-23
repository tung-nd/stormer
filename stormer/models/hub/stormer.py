import torch
import torch.nn as nn
from timm.models.vision_transformer import trunc_normal_, Mlp
from xformers.ops import memory_efficient_attention, unbind
from .weather_embedding import WeatherEmbedding


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Linear(1, hidden_size)

    def forward(self, t):
        return self.mlp(t.unsqueeze(-1))


class MemEffAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_bias=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """
    An transformers block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = MemEffAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.Identity()
        # self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class Stormer(nn.Module):
    def __init__(self, 
        in_img_size,
        variables,
        patch_size=2,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
    ):
        super().__init__()
        
        if in_img_size[0] % patch_size != 0:
            pad_size = patch_size - in_img_size[0] % patch_size
            in_img_size = (in_img_size[0] + pad_size, in_img_size[1])
        self.in_img_size = in_img_size
        self.variables = variables
        self.patch_size = patch_size
        
        # embedding
        self.embedding = WeatherEmbedding(
            variables=variables,
            img_size=in_img_size,
            patch_size=patch_size,
            embed_dim=hidden_size,
            num_heads=num_heads,
        )
        self.embed_norm_layer = nn.LayerNorm(hidden_size)
        
        # interval embedding
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        # backbone
        self.blocks = nn.ModuleList([
            Block(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        
        # prediction layer
        self.head = FinalLayer(hidden_size, patch_size, len(variables))

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        trunc_normal_(self.t_embedder.mlp.weight, std=0.02)
        
        # Zero-out adaLN modulation layers in blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.head.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.head.linear.weight, 0)
        nn.init.constant_(self.head.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor, h=None, w=None):
        """
        x: (B, L, V * patch_size**2)
        return imgs: (B, V, H, W)
        """
        p = self.patch_size
        v = len(self.variables)
        h = self.in_img_size[0] // p if h is None else h // p
        w = self.in_img_size[1] // p if w is None else w // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, v))
        x = torch.einsum("nhwpqv->nvhpwq", x)
        imgs = x.reshape(shape=(x.shape[0], v, h * p, w * p))
        return imgs

    def forward(self, x, variables, time_interval):
        
        x = self.embedding(x, variables) # B, L, D
        x = self.embed_norm_layer(x)

        time_interval_emb = self.t_embedder(time_interval)
        for block in self.blocks:
            x = block(x, time_interval_emb)
        
        x = self.head(x, time_interval_emb)
        x = self.unpatchify(x)
        
        return x
    
# variables = [
#     "2m_temperature",
#     "10m_u_component_of_wind",
#     "10m_v_component_of_wind",
#     "mean_sea_level_pressure",
#     "geopotential_50",
#     "geopotential_100",
#     "geopotential_150",
#     "geopotential_200",
#     "geopotential_250",
#     "geopotential_300",
#     "geopotential_400",
#     "geopotential_500",
#     "geopotential_600",
#     "geopotential_700",
#     "geopotential_850",
#     "geopotential_925",
#     "geopotential_1000",
#     "u_component_of_wind_50",
#     "u_component_of_wind_100",
#     "u_component_of_wind_150",
#     "u_component_of_wind_200",
#     "u_component_of_wind_250",
#     "u_component_of_wind_300",
#     "u_component_of_wind_400",
#     "u_component_of_wind_500",
#     "u_component_of_wind_600",
#     "u_component_of_wind_700",
#     "u_component_of_wind_850",
#     "u_component_of_wind_925",
#     "u_component_of_wind_1000",
#     "v_component_of_wind_50",
#     "v_component_of_wind_100",
#     "v_component_of_wind_150",
#     "v_component_of_wind_200",
#     "v_component_of_wind_250",
#     "v_component_of_wind_300",
#     "v_component_of_wind_400",
#     "v_component_of_wind_500",
#     "v_component_of_wind_600",
#     "v_component_of_wind_700",
#     "v_component_of_wind_850",
#     "v_component_of_wind_925",
#     "v_component_of_wind_1000",
#     "temperature_50",
#     "temperature_100",
#     "temperature_150",
#     "temperature_200",
#     "temperature_250",
#     "temperature_300",
#     "temperature_400",
#     "temperature_500",
#     "temperature_600",
#     "temperature_700",
#     "temperature_850",
#     "temperature_925",
#     "temperature_1000",
#     "specific_humidity_50",
#     "specific_humidity_100",
#     "specific_humidity_150",
#     "specific_humidity_200",
#     "specific_humidity_250",
#     "specific_humidity_300",
#     "specific_humidity_400",
#     "specific_humidity_500",
#     "specific_humidity_600",
#     "specific_humidity_700",
#     "specific_humidity_850",
#     "specific_humidity_925",
#     "specific_humidity_1000",
# ]
# from torch.utils.flop_counter import FlopCounterMode
# import numpy as np
# from stormer.utils.metrics import lat_weighted_mse
# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# import torch.distributed as dist
# import os

# # # Mock environment variables for single-node distributed training
# # os.environ['RANK'] = '0'
# # os.environ['WORLD_SIZE'] = '1'
# # os.environ['MASTER_ADDR'] = 'localhost'
# # os.environ['MASTER_PORT'] = '12355'

# # dist.init_process_group(backend='nccl')

# device = 'cuda'
# patch_size = 8

# model = ViTAdaLN(
#     in_img_size=(721,1440),
#     variables=variables,
#     patch_size=patch_size,
#     embed_norm=True,
#     hidden_size=1024,
#     depth=24,
#     num_heads=16,
#     mlp_ratio=4.0,
# ).to(device).half()
# # model = FSDP(
# #     model, 
# #     sharding_strategy="SHARD_GRAD_OP",
# #     # activation_checkpointing_policy={Block, ClimaXEmbedding},
# #     # auto_wrap_policy=Block
# # )

# x = torch.randn((1, 69, 721, 1440)).to(device, dtype=torch.half)
# y = torch.rand_like(x)
# pad_size = patch_size - 721 % patch_size
# padded_x = torch.nn.functional.pad(x, (0, 0, pad_size, 0), 'constant', 0)
# lat = np.random.randn(721)
# time_interval = torch.tensor([6]).to(dtype=x.dtype).to(device)

# flop_counter = FlopCounterMode(model, depth=2)
# with flop_counter:
#     # lat_weighted_mse(model(padded_x, variables, time_interval)[:, :, pad_size:], y, variables, lat)["w_mse_aggregate"].backward()
#     model(padded_x, variables, time_interval)