import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torchrl.modules import BatchRenorm1d
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
    def __init__(self, H, W, bands=8, include_xy=True):
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
        pe = self.pe
        if device is not None and pe.device != device:
            pe = pe.to(device=device, dtype=(dtype or pe.dtype))
        return pe.expand(B, -1, -1, -1)

class EmbedWithPE(nn.Module):
    """
    Embed conv + LN, then add Fourier features projected to same channels.
    y = LN(Conv(x)); y = y + Conv1x1(PE)
    """
    def __init__(self, in_ch, out_ch, out_H, out_W, bands=8):
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




import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class SimpleAgent(nn.Module):
    """
    Simple MLP actor-critic.

    Muon routing rule:
      - ONLY the hidden -> hidden linear weights use Muon
      - input -> hidden weights do NOT use Muon
      - hidden -> output weights do NOT use Muon
      - all biases / norm params use Adam
    """
    def __init__(self, envs, hidden_dim=128):
        super().__init__()

        obs_dim = int(np.prod(envs.single_observation_space.shape))
        action_dim = envs.single_action_space.n

        # ----- critic -----
        self.critic_in = layer_init(nn.Linear(obs_dim, hidden_dim))
        self.critic_ln1 = nn.LayerNorm(hidden_dim)
        self.critic_mid = layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.critic_ln2 = nn.LayerNorm(hidden_dim)
        self.critic_out = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

        # ----- actor -----
        self.actor_in = layer_init(nn.Linear(obs_dim, hidden_dim))
        self.actor_ln1 = nn.LayerNorm(hidden_dim)
        self.actor_mid = layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.actor_ln2 = nn.LayerNorm(hidden_dim)
        self.actor_out = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)

        self.act = nn.GELU()

        total_params = sum(p.numel() for p in self.parameters())
        print(f"[SimpleAgent] obs_dim={obs_dim:,}, hidden_dim={hidden_dim}, action_dim={action_dim}")
        print(f"[SimpleAgent] total parameters: {total_params:,}")

    def _flatten_obs(self, x):
        # Match your other agent's preprocessing style.
        x = x.float()
        if x.ndim > 2:
            x = x.view(x.shape[0], -1)
            x = x / 255.0
        return x

    def _critic_features(self, x):
        x = self._flatten_obs(x)
        x = self.critic_in(x)     # input -> hidden (NOT Muon)
        x = self.critic_ln1(x)
        x = self.act(x)
        x = self.critic_mid(x)    # hidden -> hidden (Muon)
        x = self.critic_ln2(x)
        x = self.act(x)
        return x

    def _actor_features(self, x):
        x = self._flatten_obs(x)
        x = self.actor_in(x)      # input -> hidden (NOT Muon)
        x = self.actor_ln1(x)
        x = self.act(x)
        x = self.actor_mid(x)     # hidden -> hidden (Muon)
        x = self.actor_ln2(x)
        x = self.act(x)
        return x

    def get_split_params(self):
        """
        Only hidden->hidden weight matrices go to Muon.
        Everything else goes to Adam.
        """
        muon_params = [
            self.actor_mid.weight,
            self.critic_mid.weight,
        ]

        muon_ids = {id(p) for p in muon_params}
        adam_params = [p for p in self.parameters() if id(p) not in muon_ids]

        return muon_params, adam_params

    def get_value(self, x):
        x = self._critic_features(x)
        return self.critic_out(x)

    def get_action_and_value(self, x, action=None):
        actor_x = self._actor_features(x)
        logits = self.actor_out(actor_x)

        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        critic_x = self._critic_features(x)
        value = self.critic_out(critic_x)

        return action, probs.log_prob(action), probs.entropy(), value


class ConvSimpleAgent(nn.Module):
    """
    PQN-style ConvNet actor-critic for Atari EnvPool observations,
    with input BatchRenorm2d.

    Expected input:
      x shape [batch, 4, 84, 84], uint8-ish pixels in [0, 255]

    Input normalization:
      x.float() / 255.0
      BatchRenorm2d(C)

    For Atari frame stacks:
      C = 4

    So BatchRenorm2d tracks one running distribution per stacked-frame channel,
    across batch and spatial dimensions. This is much more natural for convnets
    than flattening the whole image and using BatchRenorm1d(obs_dim).

    PQN-style encoder:
      Conv2d -> LayerNorm -> ReLU
      Conv2d -> LayerNorm -> ReLU
      Conv2d -> LayerNorm -> ReLU
      Flatten
      Linear -> LayerNorm -> ReLU

    PPO-style separate heads:
      actor_out:  Linear(hidden_dim, action_dim), std=0.01
      critic_out: Linear(hidden_dim, 1), std=1.0

    Muon routing:
      default:
        Muon gets trunk_fc.weight

      use_muon_input=True:
        also sends conv1.weight, conv2.weight, conv3.weight to Muon

      use_muon_output=True:
        also sends actor_out.weight and critic_out.weight to Muon

      Adam gets:
        BatchRenorm params,
        LayerNorm params,
        all biases,
        and anything not explicitly routed to Muon.
    """

    def __init__(
        self,
        envs,
        hidden_dim=512,
        *,
        use_muon_input=False,
        use_muon_output=False,
        brn_eps=1e-5,
        brn_momentum=0.01,
        brn_max_r=3.0,
        brn_max_d=5.0,
        brn_warmup_steps=1000,
        brn_smooth=True,
        norm = True,
    ):
        super().__init__()

        self.use_muon_input = use_muon_input
        self.use_muon_output = use_muon_output

        obs_shape = envs.single_observation_space.shape
        action_dim = envs.single_action_space.n

        if len(obs_shape) != 3:
            raise ValueError(
                f"Expected Atari image observation shape [C,H,W], got {obs_shape}"
            )

        c, h, w = obs_shape

        if c != 4:
            print(
                f"[BetterSimpleAgent/PQN-BRN2d warning] expected 4 stacked frames, got C={c}. "
                "Continuing anyway."
            )

        if h != 84 or w != 84:
            print(
                f"[BetterSimpleAgent/PQN-BRN2d warning] expected 84x84 Atari obs, got H={h}, W={w}. "
                "LayerNorm shapes assume standard Atari preprocessing."
            )

        if hidden_dim != 512:
            print(
                f"[BetterSimpleAgent/PQN-BRN2d warning] PQN usually uses hidden_dim=512. "
                f"You passed hidden_dim={hidden_dim}."
            )

        # Input BatchRenorm over image channels.
        #
        # For x shape [B, C, H, W], BatchRenorm2d(C) normalizes each channel
        # using statistics computed over B,H,W.
        #
        # For Atari stacked grayscale frames, C=4, so each frame index gets its
        # own running statistics.
        # self.input_brn = BatchRenorm2d(
        #     c,
        #     eps=brn_eps,
        #     momentum=brn_momentum,
        #     max_r=brn_max_r,
        #     max_d=brn_max_d,
        #     warmup_steps=brn_warmup_steps,
        #     smooth=brn_smooth,
        # )

        # ----- PQN-style conv encoder -----
        self.conv1 = layer_init(nn.Conv2d(c, 32, kernel_size=8, stride=4))
        self.ln1 = nn.LayerNorm([32, 20, 20])

        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.ln2 = nn.LayerNorm([64, 9, 9])

        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.ln3 = nn.LayerNorm([64, 7, 7])

        self.flatten = nn.Flatten()

        # For 84x84 Atari:
        # 84 -> conv1 -> 20
        # 20 -> conv2 -> 9
        # 9  -> conv3 -> 7
        # so final conv output is 64 * 7 * 7 = 3136.
        self.trunk_fc = layer_init(nn.Linear(64 * 7 * 7, hidden_dim))
        self.trunk_ln = nn.LayerNorm(hidden_dim)

        self.act = nn.GELU()

        # ----- Separate PPO actor/value heads -----
        self.actor_fc = layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.critic_fc = layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.actor_ln = nn.LayerNorm(hidden_dim)
        self.critic_ln = nn.LayerNorm(hidden_dim)

        self.actor_out = layer_init(
            nn.Linear(hidden_dim, action_dim),
            std=0.01,
        )
        self.critic_out = layer_init(
            nn.Linear(hidden_dim, 1),
            std=1.0,
        )

        total_params = sum(p.numel() for p in self.parameters())
        muon_params, adam_params = self.get_split_params()

        print(
            f"[BetterSimpleAgent/PQN-BRN2d] obs_shape={obs_shape}, "
            f"hidden_dim={hidden_dim}, action_dim={action_dim}"
        )
        print("[BetterSimpleAgent/PQN-BRN2d] input normalization: BatchRenorm2d(C)")
        print(f"[BetterSimpleAgent/PQN-BRN2d] total parameters: {total_params:,}")
        print(f"[BetterSimpleAgent/PQN-BRN2d] Muon parameters: {sum(p.numel() for p in muon_params):,}")
        print(f"[BetterSimpleAgent/PQN-BRN2d] Adam parameters: {sum(p.numel() for p in adam_params):,}")
        print(
            f"[BetterSimpleAgent/PQN-BRN2d] "
            f"use_muon_input={use_muon_input}, use_muon_output={use_muon_output}"
        )

    def _normalize_input(self, x):
        """
        Normalize raw Atari image input.

        Input:
          x: [B, C, H, W], usually uint8 in [0, 255]

        Output:
          x: [B, C, H, W], float normalized by /255 and BatchRenorm2d.
        """
        x = x.float() / 255.0
        # x = self.input_brn(x)
        return x

    def _features(self, x):
        x = self._normalize_input(x)

        x = self.conv1(x)
        x = self.ln1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.ln2(x)
        x = self.act(x)

        x = self.conv3(x)
        x = self.ln3(x)
        x = self.act(x)

        x = self.flatten(x)

        x = self.trunk_fc(x)
        x = self.trunk_ln(x)
        x = self.act(x)

        return x

    def get_split_params(self):
        """
        Returns:
            muon_params, adam_params

        Default:
          Muon:
            trunk_fc.weight

          Adam:
            input_brn params
            conv weights unless use_muon_input=True
            output heads unless use_muon_output=True
            all biases
            all LayerNorm params

        Optional:
          use_muon_input=True:
            conv1.weight
            conv2.weight
            conv3.weight

          use_muon_output=True:
            actor_out.weight
            critic_out.weight
        """

        muon_params = [
            self.actor_fc.weight,
            self.critic_fc.weight,
            self.trunk_fc.weight,
        ]

        if self.use_muon_input:
            muon_params.extend([
                # self.conv1.weight,
                self.conv2.weight,
                self.conv3.weight,
            ])

        if self.use_muon_output:
            muon_params.extend([
                self.actor_out.weight,
                self.critic_out.weight,
            ])

        muon_ids = {id(p) for p in muon_params}

        adam_params = [
            p for p in self.parameters()
            if id(p) not in muon_ids
        ]

        return muon_params, adam_params

    def get_value(self, x):
        features = self._features(x)
        return self.critic_out(self.act(self.critic_ln(self.critic_fc(features))))

    def get_action_and_value(self, x, action=None):
        features = self._features(x)

        logits = self.actor_out(self.act(self.actor_ln(self.actor_fc(features))))
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        value = self.critic_out(self.act(self.critic_ln(self.critic_fc(features))))

        return action, probs.log_prob(action), probs.entropy(), value


import math
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


NormType = Literal["layer_norm", "batch_norm", "none"]


class ChannelLayerNorm2d(nn.Module):
    """
    Flax nn.LayerNorm() on NHWC conv activations normalizes over the last dim,
    i.e. channels at each spatial location.

    PyTorch Conv2d uses NCHW, so this module applies LayerNorm over C by
    temporarily moving channels to the last dimension.
    """

    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NCHW -> NHWC
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # NHWC -> NCHW
        return x.permute(0, 3, 1, 2)


def make_conv_norm(norm_type: NormType, channels: int) -> nn.Module:
    if norm_type == "layer_norm":
        return ChannelLayerNorm2d(channels)
    if norm_type == "batch_norm":
        return nn.BatchNorm2d(channels)
    if norm_type == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm_type: {norm_type}")


def make_linear_norm(norm_type: NormType, features: int) -> nn.Module:
    if norm_type == "layer_norm":
        return nn.LayerNorm(features)
    if norm_type == "batch_norm":
        return nn.BatchNorm1d(features)
    if norm_type == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm_type: {norm_type}")


def init_he_normal(module: nn.Module) -> None:
    """
    Rough PyTorch equivalent of Flax he_normal for conv/dense layers.
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def init_lecun_normal_linear(layer: nn.Linear) -> None:
    """
    Flax Dense default is closer to LeCun normal than He normal.
    The PureJaxQL implementation uses he_normal for the trunk, but leaves
    the final Q head at the Flax Dense default.
    """
    fan_in = layer.weight.shape[1]
    std = 1.0 / math.sqrt(fan_in)
    nn.init.normal_(layer.weight, mean=0.0, std=std)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class PQNAtariNetwork(nn.Module):
    """
    PyTorch implementation similar to purejaxql/pqn_atari.py.

    Expected input:
        obs: uint8 or float tensor of shape (B, C, H, W), usually (B, 4, 84, 84)

    Output:
        q_values: tensor of shape (B, num_actions)
    """

    def __init__(
        self,
        envs,
        norm_type: NormType = "none",
        norm_input: bool = False,
        use_muon_input = True,
        use_muon_output = False,
    ):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        action_dim = envs.single_action_space.n
        c, h, w = obs_shape

        self.num_actions = action_dim
        self.norm_type = norm_type
        self.norm_input = norm_input
        self.use_muon_input = use_muon_input
        self.use_muon_output = use_muon_output
        # PureJaxQL optionally uses BatchNorm on the input before /255.
        # For PyTorch, this is BatchNorm2d over the stacked-frame channels.
        self.input_norm = nn.BatchNorm2d(c) if norm_input else nn.Identity()

        self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4, padding=0)
        if norm_type == "LayerNorm":
            self.norm1 = make_conv_norm(norm_type, 32)
            self.norm2 = make_conv_norm(norm_type, 64)
            self.norm3 = make_conv_norm(norm_type, 64)
        elif norm_type == "batch_norm":
            self.norm1 = nn.BatchNorm2d(c)
            self.norm2 = nn.BatchNorm2d(c)
            self.norm3 = nn.BatchNorm2d(c)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
            self.norm3 = nn.Identity()

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)

        # For standard Atari 84x84 input:
        # 84 -> conv8/s4 -> 20
        # 20 -> conv4/s2 -> 9
        # 9  -> conv3/s1 -> 7
        # flatten = 64 * 7 * 7 = 3136
        self.fc = nn.Linear(64 * 7 * 7, 512)
        self.fc_norm = make_linear_norm(norm_type, 512)

        self.q_head = nn.Linear(512, action_dim)

        self.apply(init_he_normal)
        init_lecun_normal_linear(self.q_head)

    def features(self, obs: torch.Tensor) -> torch.Tensor:
        # EnvPool/CleanRL Atari obs usually arrives as uint8 NCHW.
        x = obs

        if x.dtype != torch.float32:
            x = x.float()

        # Match PureJaxQL ordering: optional input norm, then scale by 255.
        x = self.input_norm(x)
        x = x / 255.0

        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc_norm(self.fc(x)))
        return x

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.features(obs)
        return self.q_head(x)

    @torch.no_grad()
    def greedy_action(self, obs: torch.Tensor) -> torch.Tensor:
        q_values = self(obs)
        return torch.argmax(q_values, dim=-1)

    @torch.no_grad()
    def epsilon_greedy_action(self, obs: torch.Tensor, epsilon: float) -> torch.Tensor:
        q_values = self(obs)
        greedy = torch.argmax(q_values, dim=-1)
        random_actions = torch.randint(
            low=0,
            high=self.num_actions,
            size=greedy.shape,
            device=obs.device,
        )
        explore = torch.rand(greedy.shape, device=obs.device) < epsilon
        return torch.where(explore, random_actions, greedy)

    def get_split_params(self):
        """
        Returns:
            muon_params, adam_params

        For this PQNAtariNetwork, the actual learnable modules are:

            input_norm   optional BatchNorm2d
            conv1        Conv2d(c -> 32)
            norm1        conv norm
            conv2        Conv2d(32 -> 64)
            norm2        conv norm
            conv3        Conv2d(64 -> 64)
            norm3        conv norm
            fc           Linear(3136 -> 512)
            fc_norm      linear norm
            q_head       Linear(512 -> action_dim)

        Default Muon routing:
            Muon:
                fc.weight

            Adam:
                all biases
                all norm params
                input_norm params
                conv weights unless use_muon_input=True
                q_head.weight unless use_muon_output=True

        Optional:
            use_muon_input=True:
                also route conv2.weight and conv3.weight to Muon
                conv1.weight stays Adam by default because it is an input layer

            use_muon_output=True:
                also route q_head.weight to Muon
        """

        muon_params = [
            self.fc.weight,
        ]

        if self.use_muon_input:
            muon_params.extend([
                # Keep conv1.weight out of Muon by default because it is the input layer.
                self.conv2.weight,
                self.conv3.weight,
            ])

        if self.use_muon_output:
            muon_params.extend([
                self.q_head.weight,
            ])

        muon_ids = {id(p) for p in muon_params}

        adam_params = [
            p for p in self.parameters()
            if id(p) not in muon_ids
        ]

        # Optional sanity checks. These catch duplicated/missing params.
        adam_ids = {id(p) for p in adam_params}
        all_ids = {id(p) for p in self.parameters()}

        assert muon_ids.isdisjoint(adam_ids), "Parameter appears in both Muon and Adam groups."
        assert muon_ids | adam_ids == all_ids, "Some parameters are missing from optimizer groups."

        return muon_params, adam_params
