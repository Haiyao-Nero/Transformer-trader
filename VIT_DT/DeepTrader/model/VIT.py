from collections import OrderedDict
from typing import Tuple, Union
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward(self, x):
        # x is (BT, HW+1, D) // BT, N, D
        #print("In_adaptor",x.shape)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        #print("Out_Adaptor", x.shape)
        return x
    
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
    
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    # def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, bias=True):
    def __init__(self, in_features, embed_dim=768, bias=True):
        super().__init__()

        self.mlp=Mlp(in_features=in_features,out_features=embed_dim)

    def forward(self, x):
        x = self.mlp(x)
        # x = x.flatten(2).transpose(1, 2)
        return x
    
class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_head: int, num_time_frames: int, attn_mask: torch.Tensor = None, d_model = int(768), scale=1., num_tadapter=1, drop_path=0.):
        super().__init__()
        self.num_tadapter = num_tadapter
        #print("Here_RAB_0", n_head, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        #print("Here_RAB_0.1")
        self.ln_1 = LayerNorm(d_model)
        #print("Here_RAB_1")
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, 1)),  # Outputting a single value for each stock
            ("sigmoid", nn.Sigmoid())  # Squash the output to [0, 1] range
        ]))
        #print("Here_RAB_2")
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.n_head = n_head

        self.MLP_Adapter = Adapter(d_model, skip_connect=False)
        self.S_Adapter = Adapter(d_model)
        self.scale = scale
        self.T_Adapter = Adapter(d_model, skip_connect=False)
        if num_tadapter == 2:
            self.T_Adapter_in = Adapter(d_model)
        self.num_time_frames = num_time_frames # this must be 271, or no of time frames
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        ## x shape [HW+1, BT, D] => N+1, BT, D
        #print("In_Residual", x.shape)
        n, bt, d = x.shape
        ## temporal adaptation
        xt = rearrange(x, 'n (b t) d -> t (b n) d', t=self.num_time_frames)
        #print("In_Residual_temporal", xt.shape)
        if self.num_tadapter == 2:
            xt = self.T_Adapter(self.attention(self.T_Adapter_in(self.ln_1(xt))))
        else:
            xt = self.T_Adapter(self.attention(self.ln_1(xt)))
        #print("In_Residual_temporal_2:", xt.shape)
        xt = rearrange(xt, 't (b n) d -> n (b t) d', n=n)
        #print("In_Residual_temporal_3:", xt.shape)
        x = x + self.drop_path(xt)
        #print("In_Residual_temporal_4:", xt.shape)
        ## spatial adaptation
        #print("In_Residual_Spatial", x.shape)
        x = x + self.S_Adapter(self.attention(self.ln_1(x)))
        ## joint adaptation
        #print("In_Residual_Joint", x.shape)
        xn = self.ln_2(x)
        x = x + self.mlp(xn) + self.drop_path(self.scale * self.MLP_Adapter(xn))
        
        #print("Out_Residual", x.shape)
        return x

class Transformer(nn.Module):
    def __init__(self, num_time_frames, num_features: int, layers: int, heads: int, attn_mask: torch.Tensor = None, num_tadapter=1, scale=1., drop_path=0.1):
        super().__init__()
        self.num_features = num_features
        self.layers = layers
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.layers)]
        #print("IN_transformer_Num_heads:", heads)
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(heads, num_time_frames) for i in range(layers)])


    def forward(self, x: torch.Tensor):
        #print("In_transformer", x.shape)
        return self.resblocks(x)
    
# Updated - v3
class VIT(nn.Module):
    ## ViT definition in CLIP image encoder
    def __init__(self, time_series_data: int, window_len: int, num_stocks:int, num_features:int, embedding_dim: int, layers: int, heads: int, drop_path_rate, num_tadapter=1, adapter_scale=0.5, pretrained=None, d_model = 768,patch_embedding_bias=True):
        super().__init__()
        self.time_series_data = time_series_data
        self.pretrained = pretrained
        self.num_stocks = num_stocks
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.window_len = window_len
        
        # self.conv1 = nn.Conv2d(in_channels=self.num_stocks,  # Treat each stock as a separate channel
        #                          out_channels=self.num_stocks * self.embedding_dim,  # Each stock gets its own set of filters
        #                          kernel_size=(self.window_len, self.num_features),  # Embedding each feature set across all time points
        #                          stride=self.window_len,
        #                          padding=(0, 0),
        #                          groups=self.num_stocks)  # Group convolution
        self.patch_embed = PatchEmbed(
            in_features=num_features, embed_dim=embedding_dim, bias=patch_embedding_bias)
        scale = embedding_dim ** -0.5
        self.layers = layers

        self.num_time_frames = (time_series_data//self.window_len)
        self.class_embedding = nn.Parameter(scale * torch.randn(embedding_dim), requires_grad=True)
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.num_time_frames, self.num_stocks, embedding_dim))
        self.ln_pre = LayerNorm(embedding_dim)

        self.temporal_embedding = nn.Parameter(torch.zeros(1, self.num_time_frames+1, embedding_dim))

        #print("Called Transformer_vit1")
        self.transformer = Transformer(self.num_time_frames, embedding_dim, layers, heads, num_tadapter=num_tadapter, scale=adapter_scale, drop_path=drop_path_rate)

        #print("After Called Transformer_vit1")
        self.ln_post = LayerNorm(embedding_dim)

        self.probability_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid())
        self.value_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Sigmoid(),
            nn.Linear(d_model // 2, 1),
            )

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained:
            self.pretrained = pretrained
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

        ## initialize S_Adapter
        for n, m in self.transformer.named_modules():
            if 'S_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        ## initialize T_Adapter
        for n, m in self.transformer.named_modules():
            if 'T_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        ## initialize MLP_Adapter
        for n, m in self.transformer.named_modules():
            if 'MLP_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed', 'temporal_embedding'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table', 'temporal_position_bias_table'}

    def forward(self, x: torch.Tensor):
        # Retrieve the dimensions of the input tensor x: B (batch size), Num 
        #print("ViT_Clip", x.shape)
        B, N, T, D = x.size()  # Decompose input tensor shape
        x = x.view(B, N, T, D)  # Combine 'one' and 'D' dimensions
        #print("VIT_Before_Conv", x.shape)
        x = rearrange(x, 'b n t d -> (b n) t d')
        x = self.patch_embed(x)
        x = x.squeeze(-1)
        x = rearrange(x, '(b n) t d -> b n d t', n=self.num_stocks).permute(0, 1, 3, 2)
        # print("VIT_After_Conv", x.shape) # Output Tensor Shape:  torch.Size([2, 6, 100, 768])
        x = rearrange(x, 'b n t d -> (b t) n d')
        x = x + self.positional_embedding.to(x.dtype)
        n = self.num_stocks
        x = rearrange(x, '(b t) n d -> (b n) t d', t=self.num_time_frames)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 0, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.temporal_embedding
        x = rearrange(x, '(b n) t d -> (b t) n d', n=n)
        
        x = self.ln_pre(x)

        #print("Called A_VIT_shape:", x.shape)
        x = x.permute(1, 0, 2)  # NLD -> LND
        #print("called B_VIT_shape:", x.shape)
        x = self.transformer(x)
        #print("called C_VIT_shape:", x.shape)
        x = x.permute(1, 0, 2)  # LND -> NLD
        #print("called D_VIT_shape:", x.shape)
        x = self.ln_post(x)
        #print("called E_VIT_shape:", x.shape)
        time_frames = x.shape[1]
        # x = x[:, 0]
        #print("called F_VIT_shape:", x.shape)
        x = rearrange(x, '(b t) n d -> b n t d',b=B)
        #print("called G_VIT_shape:", x.shape)

        # Global average pooling across time frames and embedding dimensions
        x= x[:,:,0,...] # Assuming x is of shape [batch_size, no_of_stocks, time_frames, embedded_dim]
        #print("called H_VIT_shape:", x_avg.shape)
        x_avg = torch.mean(x,-2).squeeze()

        # MLP to compute scores
        probability_scores = self.probability_mlp(x)
        probability_scores = probability_scores.squeeze(-1)
        value = self.value_mlp(x_avg).squeeze()

        return probability_scores, value

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True