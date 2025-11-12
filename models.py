import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# -------- utils --------
def layer_init(layer, std=math.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, gain=std)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)
    def forward(self, x):
        x = x.permute(0,2,3,1); x = self.ln(x); x = x.permute(0,3,1,2)
        return x


class AttentionBottleNeck(nn.Module):
    def __init__(self, input_dim, embed_dim, num_queries, n_heads, final_hidden_dim, include_pool = False):
        super().__init__()
        self.input_norm = LayerNorm2d(embed_dim)
        self.attn_proj = nn.Conv2d(input_dim,num_queries * n_heads, 1, groups=n_heads, bias=False)
        self.val_proj = nn.Conv2d(input_dim,embed_dim,1)
        self.n_heads = n_heads
        self.num_queries = num_queries
        self.final_linear = nn.Linear(embed_dim * num_queries, final_hidden_dim)

    def forward(self,x):
        # x dim -> B x C x H x W
        B , C , H , W = x.size()
        x = self.input_norm(x)
        A_w = self.attn_proj(x)
        A_w = torch.softmax(torch.flatten(A_w,2,3),2)
        # A_w dim -> B x nquery*nhead x (H*W)
        A_w = torch.reshape(A_w, [B* self.n_heads, self.num_queries, H*W])
        # A_w dim -> B*nhead x nquery x (H*W)

        V = self.val_proj(x)
        V = torch.flatten(V,2,3)
        V = torch.reshape(V,[B * self.n_heads, C//self.n_heads, H*W])
        # V dim -> B*nhead x C/nheads  x H*W
        V = torch.permute(V, [0, 2, 1])
        # V dim -> B*nhead x H*W x C/nheads
        A = torch.bmm(A_w, V)
        # A dim -> B*nhead x nquery x C/nheads
        A = torch.reshape(A,[B, self.num_queries*C])
        o = self.final_linear(A)



        return o

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4, kernel=5):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel, padding=kernel//2, groups=dim, bias=True)
        self.ln = LayerNorm2d(dim)
        hidden = int(mlp_ratio*dim)
        self.pw1 = nn.Conv2d(dim, hidden, 1, bias=True)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden, dim, 1, bias=True)
    def forward(self, x):
        r = x
        x = self.dw(x)
        x = self.ln(x)
        x = self.pw1(x); x = self.act(x)
        x = self.pw2(x)
        return x + r

class Downsample(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.ln = LayerNorm2d(c_in)
        self.conv = nn.Conv2d(c_in, c_out, 2, stride=2, bias=True)
    def forward(self, x):
        return self.conv(self.ln(x))

# -------- Fourier PE (added after embed by addition) --------
class PosEnc2DFourier(nn.Module):
    """
    Fixed 2D Fourier features on HxW grid.
    Channels: [x, y, sin/cos bands on x and y], total Cpos = 2 + 4*bands (if include_xy).
    """
    def __init__(self, H, W, bands=8, simple_linear = True, include_xy=True):
        super().__init__()
        ys = torch.linspace(0, 1, steps=H).view(H,1).expand(H,W)
        xs = torch.linspace(0, 1, steps=W).view(1,W).expand(H,W)
        xs = xs.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        ys = ys.unsqueeze(0).unsqueeze(0)
        feats = []
        if include_xy:
            feats += [xs, ys]
        for k in range(bands):
            f = (2.0**k) * math.pi
            feats += [torch.sin(f*xs), torch.cos(f*xs)]
            feats += [torch.sin(f*ys), torch.cos(f*ys)]
        pe = torch.cat(feats, dim=1)  # (1,Cpos,H,W)
        self.register_buffer("pe", pe, persistent=False)
    @property
    def cpos(self): return self.pe.shape[1]
    def forward(self, B, device=None, dtype=None):
        pe = self.linear(self.pe)
        if device is not None and pe.device != device:
            pe = pe.to(device=device, dtype=(dtype or pe.dtype))
        return pe.expand(B, -1, -1, -1)


class EmbedWithPE(nn.Module):
    """
    Embed conv + LN, then add Fourier features projected to same channels.
    y = LN(Conv(x)); y = y + Conv1x1(PE)
    """
    def __init__(self, in_ch, out_ch, out_H, out_W, bands=8, learned_pe = True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=True)
        self.ln   = LayerNorm2d(out_ch)
        self.pe   = PosEnc2DFourier(out_H, out_W, bands=bands, include_xy=True)
        self.pe_proj = nn.Conv2d(self.pe.cpos, out_ch, kernel_size=1, bias=True)
        # report
        self.pe_params = self.pe_proj.weight.numel() + (self.pe_proj.bias.numel() if self.pe_proj.bias is not None else 0)
    def forward(self, x):
        y = self.ln(self.conv(x))
        B, _, H, W = y.shape
        pe = self.pe(B, device=y.device, dtype=y.dtype)
        y = y + self.pe_proj(pe)
        return y

# -------- Agent --------
class Agent(nn.Module):
    """
    Attributes:
      self.embed           # stem with Fourier PE added by + (projection)
      self.backbone        # ConvNeXt blocks + downsamples + 1x1 channel bottleneck
      self.flatten_linear  # Linear(flattened -> 512)  [kept separate for optimizer routing]
      self.mlp             # GELU -> Linear(512->512) -> GELU
      self.actor, self.critic
    """
    def __init__(self, envs,
                 dims=(64,128,256), depths=(2,3,4),
                 dw_kernel=5, pe_bands=8,
                 bottleneck_red=96):
        super().__init__()
        # ---- infer obs/action
        (in_ch, in_h, in_w) = envs.single_observation_space.shape
        action_dim = envs.single_action_space.n

        # ---- quick probe to know embed output H,W (halve by stride=2)
        out_h = (in_h + 2*1 - 4)//2 + 1  # conv2d formula with k=4,s=2,p=1
        out_w = (in_w + 2*1 - 4)//2 + 1

        # ---- embed (conv + LN) then add PE by +
        self.embed = EmbedWithPE(in_ch, dims[0], out_h, out_w, bands=pe_bands)

        # ---- backbone core
        stages = []
        stages.append(nn.Sequential(*[ConvNeXtBlock(dims[0], mlp_ratio=4, kernel=dw_kernel) for _ in range(depths[0])]))
        stages.append(Downsample(dims[0], dims[1]))
        stages.append(nn.Sequential(*[ConvNeXtBlock(dims[1], mlp_ratio=4, kernel=dw_kernel) for _ in range(depths[1])]))
        stages.append(Downsample(dims[1], dims[2]))
        stages.append(nn.Sequential(*[ConvNeXtBlock(dims[2], mlp_ratio=4, kernel=dw_kernel) for _ in range(depths[2])]))
        self._core = nn.Sequential(*stages)

        # ---- probe final spatial size to build bottleneck + flatten
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, in_h, in_w)
            feat = self._core(self.embed(dummy))   # (1, Cb, Hb, Wb)
            _, Cb, Hb, Wb = feat.shape

        # ---- simple 1x1 bottleneck (no explicit PE here; we rely on embed + network)
        self.bottleneck = nn.Conv2d(Cb, bottleneck_red, kernel_size=1, bias=True)

        # expose backbone as single module (embed -> core -> bottleneck is applied in forward)
        self.backbone = nn.Sequential(self._core, self.bottleneck)

        # ---- flattened dim after bottleneck
        self.flattened_dim = bottleneck_red * Hb * Wb

        # ---- separate big linear
        self.flatten_linear = layer_init(nn.Linear(self.flattened_dim, 512))
        self.mlp = nn.Sequential(
            nn.GELU(),
            layer_init(nn.Linear(512, 512)),
            nn.GELU(),
        )
        self.actor  = layer_init(nn.Linear(512, action_dim), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1),          std=1.0)

        # ---- reporting
        total_params = sum(p.numel() for p in self.parameters())
        flat_params  = self.flattened_dim * 512 + 512
        print(f"[Agent] Embed out: C={dims[0]}, H={out_h}, W={out_w} | PE proj params: {self.embed.pe_params:,}")
        print(f"[Agent] Backbone out: C={bottleneck_red}, H={Hb}, W={Wb}")
        print(f"[Agent] Flattened dim: {self.flattened_dim:,}")
        print(f"[Agent] Params in flatten_linear ({self.flattened_dim} -> 512): {flat_params:,}")
        print(f"[Agent] Total parameters (incl. heads): {total_params:,}")

    def get_split_params(self):
        muon_params = [p for p in self.backbone.parameters() if p.ndim >= 2] + \
                      [p for p in self.mlp.parameters() if p.ndim >= 2]
        muon_ids = {id(p) for p in self.parameters()}
        adam_params = [p for p in self.parameters() if id(p) not in muon_ids]

        return muon_params , adam_params
    # ---- API
    def get_value(self, x):
        x = x / 255.0
        x = self.embed(x)
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.flatten_linear(x)
        x = self.mlp(x)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x = x / 255.0
        x = self.embed(x)
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.flatten_linear(x)
        x = self.mlp(x)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
