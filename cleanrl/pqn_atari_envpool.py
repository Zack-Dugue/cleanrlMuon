# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/pqn/#pqn_atari_envpoolpy
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import envpool
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from optimizers import AdaMuonWithAuxAdam, MuonWithAuxAdam, BGD, SingleDeviceNorMuonWithAuxAdam
from models import PQNAtariNetwork


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """wandb tag"""
    wandb_tag: str = None
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # optional multi-gpu launcher compatibility
    device: str = None

    env_id: str = "Breakout-v5"
    """the id of the environment"""
    total_timesteps: int = 10_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 128
    """the number of parallel game environments"""
    num_steps: int = 8
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the Q-network"""
    max_grad_norm: float = 10.0
    """the maximum norm for the gradient clipping"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.001
    """the fraction of `total_timesteps` it takes from start_e to end_e"""
    q_lambda: float = 0.65
    """the lambda for the Q-Learning algorithm"""

    # optimizer parity with your PPO script
    optimizer: str = "Adam"  # ["SGD", "Adam", "Muon", "NorMuon", "AdaMuon", "BGD", "NystromSGD"]
    momentum: float = 0.95

    # Nyström Jacobian-whitened SGD
    nystrom_use_nystrom: bool = True
    """True uses a rank-R Nyström Gram; False uses the exact minibatch Gram"""
    nystrom_directions: int = 64
    """number of randomly selected Nyström anchor samples"""
    nystrom_use_spatial_mean: bool = True
    """True uses the fast spatial-mean Conv2d approximation; False keeps exact spatial covariance"""
    nystrom_use_mean_gradient_nullspace: bool = True
    """restore the ordinary MSE gradient outside the retained Nyström subspace"""
    nystrom_use_full_natural_gradient: bool = False
    """False uses inverse-square-root whitening; True uses the full damped inverse"""
    nystrom_damping: float = 1.0e-4
    """ridge relative to the mean diagonal of J J^T / minibatch_size"""
    nystrom_eigenvalue_tolerance: float = 1.0e-7
    """relative spectral threshold used to remove numerical zero directions"""
    nystrom_conv_sample_chunk_size: int = 32
    """sample chunk used for exact per-sample Conv2d Jacobian construction"""
    nystrom_weight_decay: float = 0.0
    """coupled SGD weight decay used only by NystromSGD"""
    nystrom_log_every: int = 10
    """log Nyström diagnostics every this many optimizer updates"""

    # model optimizer-routing kwargs
    use_muon_input: bool = True
    use_muon_output: bool = False

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    @property
    def __class__(self):
        return super().__class__

    def __ne__(self, value, /):
        return super().__ne__(value)


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - infos["terminated"]
        self.episode_lengths *= 1 - infos["terminated"]
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        if "lives" not in infos:
            infos["lives"] = np.zeros(self.num_envs, dtype=np.int32)
        return (
            observations,
            rewards,
            dones,
            infos,
        )


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)



@dataclass
class ModuleFactors:
    module: nn.Module
    inputs: torch.Tensor
    output_gradients: torch.Tensor


@dataclass
class NystromDiagnostics:
    effective_rank: int
    relative_weight_change_from_mse: float
    ordinary_weight_norm: float
    whitened_weight_norm: float
    minimum_retained_eigenvalue: float
    maximum_retained_eigenvalue: float
    retained_condition_number: float


class JacobianGeometryCollector:
    """
    Build J J^T contributions from Conv2d, Linear, and LayerNorm modules.

    The collector captures unweighted selected-Q Jacobian factors during one
    backward pass. Conv2d geometry can either retain the exact spatial
    activation/error covariance or use the cheaper spatial-mean approximation.
    """

    def __init__(
        self,
        model: nn.Module,
        use_spatial_mean: bool,
        conv_sample_chunk_size: int,
    ):
        self.model = model
        self.use_spatial_mean = use_spatial_mean
        self.conv_sample_chunk_size = conv_sample_chunk_size

        self.handles = []
        self.parameter_ids = set()

        self.collecting = False
        self.batch_size = 0
        self.anchor_indices = None
        self.cross_matrix = None
        self.diagonal = None
        self.saved_inputs: Dict[nn.Module, torch.Tensor] = {}
        self.factors: List[ModuleFactors] = []

        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.LayerNorm)):
                if isinstance(module, nn.Conv2d) and module.groups != 1:
                    raise NotImplementedError(
                        "NystromSGD does not currently support grouped Conv2d."
                    )

                for parameter in module.parameters(recurse=False):
                    self.parameter_ids.add(id(parameter))

                self.handles.append(module.register_forward_hook(self._save_input))
                self.handles.append(
                    module.register_full_backward_pre_hook(
                        self._capture_output_gradient
                    )
                )

        model_parameter_ids = {
            id(parameter)
            for parameter in model.parameters()
            if parameter.requires_grad
        }
        missing = model_parameter_ids - self.parameter_ids
        if missing:
            raise RuntimeError(
                "NystromSGD only reconstructs Conv2d, Linear, and LayerNorm "
                f"parameters. Unsupported trainable parameter count: {len(missing)}"
            )

    def start(
        self,
        batch_size: int,
        anchor_indices: torch.Tensor,
        device: torch.device,
    ):
        if self.collecting:
            raise RuntimeError("The Nyström collector is already active.")
        if anchor_indices.ndim != 1:
            raise ValueError("anchor_indices must be one-dimensional.")

        self.collecting = True
        self.batch_size = batch_size
        self.anchor_indices = anchor_indices
        self.cross_matrix = torch.zeros(
            batch_size,
            anchor_indices.numel(),
            device=device,
            dtype=torch.float32,
        )
        self.diagonal = torch.zeros(
            batch_size,
            device=device,
            dtype=torch.float32,
        )
        self.saved_inputs.clear()
        self.factors.clear()

    def _save_input(self, module, inputs, output):
        if self.collecting:
            self.saved_inputs[module] = inputs[0].detach()

    def _capture_output_gradient(self, module, grad_output):
        if not self.collecting:
            return
        if module not in self.saved_inputs:
            raise RuntimeError(
                f"Missing saved input for {type(module).__name__}."
            )

        inputs = self.saved_inputs.pop(module)
        output_gradients = grad_output[0].detach()

        with torch.no_grad():
            if isinstance(module, nn.Linear):
                self._add_linear_contribution(
                    module,
                    inputs,
                    output_gradients,
                )
            elif isinstance(module, nn.Conv2d):
                self._add_conv2d_contribution(
                    module,
                    inputs,
                    output_gradients,
                )
            elif isinstance(module, nn.LayerNorm):
                self._add_layernorm_contribution(
                    module,
                    inputs,
                    output_gradients,
                )
            else:
                raise TypeError(type(module))

            self.factors.append(
                ModuleFactors(
                    module=module,
                    inputs=inputs,
                    output_gradients=output_gradients,
                )
            )

    def _add_linear_contribution(
        self,
        module: nn.Linear,
        inputs: torch.Tensor,
        output_gradients: torch.Tensor,
    ):
        activations = inputs.reshape(self.batch_size, -1).float()
        gradients = output_gradients.reshape(self.batch_size, -1).float()

        if activations.shape[1] != module.in_features:
            raise RuntimeError(
                "NystromSGD expects one Linear input vector per sample."
            )

        anchors = self.anchor_indices
        activation_cross = activations @ activations[anchors].T
        gradient_cross = gradients @ gradients[anchors].T

        self.cross_matrix.add_(activation_cross * gradient_cross)
        self.diagonal.add_(
            activations.square().sum(dim=1)
            * gradients.square().sum(dim=1)
        )

        if module.bias is not None:
            self.cross_matrix.add_(gradient_cross)
            self.diagonal.add_(gradients.square().sum(dim=1))

    @staticmethod
    def _conv_per_sample_weight_gradients(
        module: nn.Conv2d,
        inputs: torch.Tensor,
        output_gradients: torch.Tensor,
    ) -> torch.Tensor:
        unfolded = F.unfold(
            inputs.float(),
            kernel_size=module.kernel_size,
            dilation=module.dilation,
            padding=module.padding,
            stride=module.stride,
        )
        gradients_flat = output_gradients.float().flatten(start_dim=2)
        if gradients_flat.shape[-1] != unfolded.shape[-1]:
            raise RuntimeError(
                "Conv2d unfolded spatial dimensions do not match."
            )

        per_sample = torch.bmm(
            gradients_flat,
            unfolded.transpose(1, 2),
        )
        return per_sample.flatten(start_dim=1)

    def _add_conv2d_exact_contribution(
        self,
        module: nn.Conv2d,
        inputs: torch.Tensor,
        output_gradients: torch.Tensor,
    ):
        anchors = self.anchor_indices
        full_exact = (
            anchors.numel() == self.batch_size
            and torch.equal(
                anchors,
                torch.arange(
                    self.batch_size,
                    device=anchors.device,
                ),
            )
        )

        if full_exact:
            all_weight_gradients = (
                self._conv_per_sample_weight_gradients(
                    module,
                    inputs,
                    output_gradients,
                )
            )
            self.cross_matrix.add_(
                all_weight_gradients @ all_weight_gradients.T
            )
            self.diagonal.add_(
                all_weight_gradients.square().sum(dim=1)
            )
            del all_weight_gradients
            return

        anchor_weight_gradients = (
            self._conv_per_sample_weight_gradients(
                module,
                inputs[anchors],
                output_gradients[anchors],
            )
        )

        for start in range(
            0,
            self.batch_size,
            self.conv_sample_chunk_size,
        ):
            end = min(
                start + self.conv_sample_chunk_size,
                self.batch_size,
            )
            chunk_weight_gradients = (
                self._conv_per_sample_weight_gradients(
                    module,
                    inputs[start:end],
                    output_gradients[start:end],
                )
            )
            self.cross_matrix[start:end].add_(
                chunk_weight_gradients
                @ anchor_weight_gradients.T
            )
            self.diagonal[start:end].add_(
                chunk_weight_gradients.square().sum(dim=1)
            )
            del chunk_weight_gradients

        del anchor_weight_gradients

    @staticmethod
    def _conv_spatial_mean_patches(
        module: nn.Conv2d,
        inputs: torch.Tensor,
        output_height: int,
        output_width: int,
    ) -> torch.Tensor:
        if isinstance(module.padding, str):
            raise NotImplementedError(
                "Spatial-mean geometry requires numeric Conv2d padding."
            )

        inputs_float = inputs.float()
        kernel_height, kernel_width = module.kernel_size
        stride_height, stride_width = module.stride
        dilation_height, dilation_width = module.dilation
        padding_height, padding_width = module.padding

        if module.padding_mode == "zeros":
            padding_mode = "constant"
        else:
            padding_mode = module.padding_mode

        if padding_height != 0 or padding_width != 0:
            padded_inputs = F.pad(
                inputs_float,
                (
                    padding_width,
                    padding_width,
                    padding_height,
                    padding_height,
                ),
                mode=padding_mode,
            )
        else:
            padded_inputs = inputs_float

        if not padded_inputs.is_contiguous():
            padded_inputs = padded_inputs.contiguous()

        batch_size, input_channels, padded_height, padded_width = (
            padded_inputs.shape
        )
        required_height = (
            (output_height - 1) * stride_height
            + (kernel_height - 1) * dilation_height
            + 1
        )
        required_width = (
            (output_width - 1) * stride_width
            + (kernel_width - 1) * dilation_width
            + 1
        )
        if (
            required_height > padded_height
            or required_width > padded_width
        ):
            raise RuntimeError(
                "Conv2d output shape is inconsistent with its settings."
            )

        (
            batch_stride,
            channel_stride,
            height_stride,
            width_stride,
        ) = padded_inputs.stride()

        patch_view = padded_inputs.as_strided(
            size=(
                batch_size,
                input_channels,
                kernel_height,
                kernel_width,
                output_height,
                output_width,
            ),
            stride=(
                batch_stride,
                channel_stride,
                dilation_height * height_stride,
                dilation_width * width_stride,
                stride_height * height_stride,
                stride_width * width_stride,
            ),
        )
        return patch_view.mean(dim=(4, 5)).reshape(batch_size, -1)

    def _add_conv2d_spatial_mean_contribution(
        self,
        module: nn.Conv2d,
        inputs: torch.Tensor,
        output_gradients: torch.Tensor,
    ) -> torch.Tensor:
        output_gradients_float = output_gradients.float()
        output_height = output_gradients_float.shape[2]
        output_width = output_gradients_float.shape[3]

        mean_activations = self._conv_spatial_mean_patches(
            module,
            inputs,
            output_height,
            output_width,
        )
        summed_gradients = output_gradients_float.sum(dim=(2, 3))
        anchors = self.anchor_indices

        activation_cross = (
            mean_activations
            @ mean_activations[anchors].T
        )
        gradient_cross = (
            summed_gradients
            @ summed_gradients[anchors].T
        )

        self.cross_matrix.add_(
            activation_cross * gradient_cross
        )
        self.diagonal.add_(
            mean_activations.square().sum(dim=1)
            * summed_gradients.square().sum(dim=1)
        )

        del mean_activations
        return summed_gradients

    def _add_conv2d_contribution(
        self,
        module: nn.Conv2d,
        inputs: torch.Tensor,
        output_gradients: torch.Tensor,
    ):
        bias_gradients = None

        if self.use_spatial_mean:
            bias_gradients = (
                self._add_conv2d_spatial_mean_contribution(
                    module,
                    inputs,
                    output_gradients,
                )
            )
        else:
            self._add_conv2d_exact_contribution(
                module,
                inputs,
                output_gradients,
            )

        if module.bias is not None:
            anchors = self.anchor_indices
            if bias_gradients is None:
                bias_gradients = (
                    output_gradients.float().sum(dim=(2, 3))
                )

            self.cross_matrix.add_(
                bias_gradients @ bias_gradients[anchors].T
            )
            self.diagonal.add_(
                bias_gradients.square().sum(dim=1)
            )

    @staticmethod
    def _layernorm_local_gradients(
        module: nn.LayerNorm,
        inputs: torch.Tensor,
        output_gradients: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs_float = inputs.float()
        gradients_float = output_gradients.float()

        normalized_ndim = len(module.normalized_shape)
        normalized_dims = tuple(
            range(
                inputs.ndim - normalized_ndim,
                inputs.ndim,
            )
        )
        mean = inputs_float.mean(
            dim=normalized_dims,
            keepdim=True,
        )
        variance = inputs_float.var(
            dim=normalized_dims,
            unbiased=False,
            keepdim=True,
        )
        normalized_inputs = (
            (inputs_float - mean)
            * torch.rsqrt(variance + module.eps)
        )

        leading_dims = tuple(
            range(1, inputs.ndim - normalized_ndim)
        )

        scale_local = None
        bias_local = None

        if module.elementwise_affine and module.weight is not None:
            scale_local = gradients_float * normalized_inputs
            if leading_dims:
                scale_local = scale_local.sum(dim=leading_dims)
            scale_local = scale_local.reshape(inputs.shape[0], -1)

        if module.elementwise_affine and module.bias is not None:
            bias_local = gradients_float
            if leading_dims:
                bias_local = bias_local.sum(dim=leading_dims)
            bias_local = bias_local.reshape(inputs.shape[0], -1)

        return scale_local, bias_local

    def _add_layernorm_contribution(
        self,
        module: nn.LayerNorm,
        inputs: torch.Tensor,
        output_gradients: torch.Tensor,
    ):
        scale_local, bias_local = self._layernorm_local_gradients(
            module,
            inputs,
            output_gradients,
        )
        anchors = self.anchor_indices

        if scale_local is not None:
            self.cross_matrix.add_(
                scale_local @ scale_local[anchors].T
            )
            self.diagonal.add_(
                scale_local.square().sum(dim=1)
            )

        if bias_local is not None:
            self.cross_matrix.add_(
                bias_local @ bias_local[anchors].T
            )
            self.diagonal.add_(
                bias_local.square().sum(dim=1)
            )

    def finish(
        self,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        List[ModuleFactors],
    ]:
        if not self.collecting:
            raise RuntimeError("The Nyström collector was not started.")
        if self.saved_inputs:
            raise RuntimeError(
                f"{len(self.saved_inputs)} saved module inputs "
                "were not consumed by backward hooks."
            )

        cross_matrix = self.cross_matrix
        diagonal = self.diagonal
        factors = self.factors

        self.collecting = False
        self.batch_size = 0
        self.anchor_indices = None
        self.cross_matrix = None
        self.diagonal = None
        self.factors = []

        return cross_matrix, diagonal, factors

    def abort(self):
        self.collecting = False
        self.batch_size = 0
        self.anchor_indices = None
        self.cross_matrix = None
        self.diagonal = None
        self.saved_inputs.clear()
        self.factors.clear()

    def remove(self):
        for handle in self.handles:
            handle.remove()


def choose_nystrom_anchor_indices(
    batch_size: int,
    use_nystrom: bool,
    nystrom_directions: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    if not use_nystrom:
        return torch.arange(batch_size, device=device)

    if nystrom_directions < 1:
        raise ValueError(
            "nystrom_directions must be at least 1."
        )
    if nystrom_directions > batch_size:
        raise ValueError(
            "nystrom_directions cannot exceed the optimization "
            f"minibatch size: {nystrom_directions} > {batch_size}."
        )

    return torch.randperm(
        batch_size,
        device=device,
        generator=generator,
    )[:nystrom_directions]


@torch.no_grad()
def _retained_subspace_sample_weights(
    eigenvectors: torch.Tensor,
    raw_eigenvalues: torch.Tensor,
    rhs: torch.Tensor,
    ordinary_weights: torch.Tensor,
    batch_size: int,
    ridge: torch.Tensor,
    use_mean_gradient_nullspace: bool,
    use_full_natural_gradient: bool,
):
    if raw_eigenvalues.numel() == 0:
        if use_mean_gradient_nullspace:
            return (
                ordinary_weights,
                0,
                0.0,
                0.0,
                float("inf"),
            )
        return (
            torch.zeros_like(ordinary_weights),
            0,
            0.0,
            0.0,
            float("inf"),
        )

    fisher_eigenvalues = raw_eigenvalues / batch_size
    projected_rhs = eigenvectors.T @ rhs
    damped_eigenvalues = fisher_eigenvalues + ridge

    if use_full_natural_gradient:
        spectral_weights = damped_eigenvalues.reciprocal()
    else:
        spectral_weights = damped_eigenvalues.rsqrt()

    transformed_weights = (
        eigenvectors
        @ (projected_rhs * spectral_weights)
    ) / batch_size

    if use_mean_gradient_nullspace:
        ordinary_retained = (
            eigenvectors @ projected_rhs
        ) / batch_size
        sample_weights = (
            transformed_weights
            + ordinary_weights
            - ordinary_retained
        )
    else:
        sample_weights = transformed_weights

    minimum = fisher_eigenvalues.min().item()
    maximum = fisher_eigenvalues.max().item()
    condition = maximum / max(minimum, 1.0e-30)

    return (
        sample_weights,
        raw_eigenvalues.numel(),
        minimum,
        maximum,
        condition,
    )


@torch.no_grad()
def compute_nystrom_sample_weights(
    cross_matrix: torch.Tensor,
    diagonal: torch.Tensor,
    anchor_indices: torch.Tensor,
    rhs: torch.Tensor,
    use_nystrom: bool,
    use_mean_gradient_nullspace: bool,
    use_full_natural_gradient: bool,
    damping: float,
    eigenvalue_tolerance: float,
) -> Tuple[torch.Tensor, NystromDiagnostics]:
    batch_size = rhs.numel()
    ordinary_weights = rhs / batch_size

    mean_diagonal_fisher = (
        diagonal.mean() / batch_size
    ).clamp_min(1.0e-30)
    ridge = damping * mean_diagonal_fisher

    if not use_nystrom:
        gram = 0.5 * (cross_matrix + cross_matrix.T)
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)

        largest = eigenvalues.max().clamp_min(0.0)
        threshold = max(
            largest.item() * eigenvalue_tolerance,
            1.0e-20,
        )
        keep = eigenvalues > threshold

        retained_values = eigenvalues[keep]
        retained_vectors = eigenvectors[:, keep]
    else:
        anchor_matrix = cross_matrix[anchor_indices]
        anchor_matrix = 0.5 * (
            anchor_matrix + anchor_matrix.T
        )
        anchor_values, anchor_vectors = torch.linalg.eigh(
            anchor_matrix
        )

        largest_anchor = anchor_values.max().clamp_min(0.0)
        anchor_threshold = max(
            largest_anchor.item() * eigenvalue_tolerance,
            1.0e-20,
        )
        keep_anchor = anchor_values > anchor_threshold

        if not torch.any(keep_anchor):
            retained_vectors = cross_matrix.new_zeros(
                (batch_size, 0)
            )
            retained_values = cross_matrix.new_zeros((0,))
        else:
            anchor_values = anchor_values[keep_anchor]
            anchor_vectors = anchor_vectors[:, keep_anchor]

            anchor_inverse_square_root = (
                anchor_vectors
                * anchor_values.rsqrt()[None, :]
            ) @ anchor_vectors.T

            z_matrix = (
                cross_matrix @ anchor_inverse_square_root
            )
            small_matrix = z_matrix.T @ z_matrix
            small_matrix = 0.5 * (
                small_matrix + small_matrix.T
            )

            small_values, small_vectors = torch.linalg.eigh(
                small_matrix
            )
            largest_small = small_values.max().clamp_min(0.0)
            small_threshold = max(
                largest_small.item() * eigenvalue_tolerance,
                1.0e-20,
            )
            keep_small = small_values > small_threshold
            retained_values = small_values[keep_small]

            if retained_values.numel() == 0:
                retained_vectors = cross_matrix.new_zeros(
                    (batch_size, 0)
                )
            else:
                retained_vectors = (
                    z_matrix @ small_vectors[:, keep_small]
                ) * retained_values.rsqrt()[None, :]

    (
        sample_weights,
        rank,
        minimum,
        maximum,
        condition,
    ) = _retained_subspace_sample_weights(
        eigenvectors=retained_vectors,
        raw_eigenvalues=retained_values,
        rhs=rhs,
        ordinary_weights=ordinary_weights,
        batch_size=batch_size,
        ridge=ridge,
        use_mean_gradient_nullspace=(
            use_mean_gradient_nullspace
        ),
        use_full_natural_gradient=(
            use_full_natural_gradient
        ),
    )

    ordinary_norm = torch.linalg.vector_norm(
        ordinary_weights
    )
    whitened_norm = torch.linalg.vector_norm(
        sample_weights
    )
    relative_change = (
        torch.linalg.vector_norm(
            sample_weights - ordinary_weights
        )
        / ordinary_norm.clamp_min(1.0e-30)
    ).item()

    diagnostics = NystromDiagnostics(
        effective_rank=rank,
        relative_weight_change_from_mse=relative_change,
        ordinary_weight_norm=ordinary_norm.item(),
        whitened_weight_norm=whitened_norm.item(),
        minimum_retained_eigenvalue=minimum,
        maximum_retained_eigenvalue=maximum,
        retained_condition_number=condition,
    )
    return sample_weights, diagnostics


@torch.no_grad()
def assign_nystrom_reconstructed_gradients(
    factors: Sequence[ModuleFactors],
    sample_weights: torch.Tensor,
):
    batch_size = sample_weights.numel()

    for factor in factors:
        module = factor.module
        inputs = factor.inputs
        output_gradients = factor.output_gradients

        if isinstance(module, nn.Linear):
            activations = inputs.reshape(batch_size, -1).float()
            gradients = output_gradients.reshape(
                batch_size,
                -1,
            ).float()
            weighted_gradients = (
                gradients * sample_weights[:, None]
            )

            if module.weight.requires_grad:
                module.weight.grad = (
                    weighted_gradients.T @ activations
                ).to(dtype=module.weight.dtype)

            if (
                module.bias is not None
                and module.bias.requires_grad
            ):
                module.bias.grad = weighted_gradients.sum(
                    dim=0
                ).to(dtype=module.bias.dtype)

        elif isinstance(module, nn.Conv2d):
            weighted_output_gradients = (
                output_gradients.float()
                * sample_weights.view(
                    batch_size,
                    1,
                    1,
                    1,
                )
            )

            if module.weight.requires_grad:
                module.weight.grad = torch.nn.grad.conv2d_weight(
                    inputs.float(),
                    module.weight.shape,
                    weighted_output_gradients,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                ).to(dtype=module.weight.dtype)

            if (
                module.bias is not None
                and module.bias.requires_grad
            ):
                module.bias.grad = (
                    weighted_output_gradients.sum(
                        dim=(0, 2, 3)
                    ).to(dtype=module.bias.dtype)
                )

        elif isinstance(module, nn.LayerNorm):
            inputs_float = inputs.float()
            gradients_float = output_gradients.float()

            normalized_ndim = len(module.normalized_shape)
            normalized_dims = tuple(
                range(
                    inputs.ndim - normalized_ndim,
                    inputs.ndim,
                )
            )
            leading_dims = tuple(
                range(1, inputs.ndim - normalized_ndim)
            )

            mean = inputs_float.mean(
                dim=normalized_dims,
                keepdim=True,
            )
            variance = inputs_float.var(
                dim=normalized_dims,
                unbiased=False,
                keepdim=True,
            )
            normalized_inputs = (
                (inputs_float - mean)
                * torch.rsqrt(variance + module.eps)
            )

            weight_view = sample_weights.view(
                batch_size,
                *([1] * (inputs.ndim - 1)),
            )
            sum_dims = (0,) + leading_dims

            if (
                module.elementwise_affine
                and module.weight is not None
                and module.weight.requires_grad
            ):
                module.weight.grad = (
                    gradients_float
                    * normalized_inputs
                    * weight_view
                ).sum(dim=sum_dims).to(
                    dtype=module.weight.dtype
                )

            if (
                module.elementwise_affine
                and module.bias is not None
                and module.bias.requires_grad
            ):
                module.bias.grad = (
                    gradients_float * weight_view
                ).sum(dim=sum_dims).to(
                    dtype=module.bias.dtype
                )
        else:
            raise TypeError(type(module))


def nystrom_sgd_step(
    q_network: nn.Module,
    optimizer: optim.Optimizer,
    collector: JacobianGeometryCollector,
    observations: torch.Tensor,
    actions: torch.Tensor,
    targets: torch.Tensor,
    args: Args,
    device: torch.device,
    anchor_generator: torch.Generator,
):
    optimizer.zero_grad(set_to_none=True)
    minibatch_size = observations.shape[0]

    anchor_indices = choose_nystrom_anchor_indices(
        batch_size=minibatch_size,
        use_nystrom=args.nystrom_use_nystrom,
        nystrom_directions=args.nystrom_directions,
        device=device,
        generator=anchor_generator,
    )

    original_requires_grad = [
        parameter.requires_grad
        for parameter in q_network.parameters()
    ]
    collector_started = False

    try:
        for parameter in q_network.parameters():
            parameter.requires_grad_(False)

        # EnvPool Atari observations are normally uint8. Integer tensors cannot
        # require gradients, so use an equivalent floating-point copy for the
        # Jacobian-only forward pass. PQNAtariNetwork already treats its input
        # numerically (normally dividing by 255), so this does not change Q.
        geometry_observations = observations.detach()
        if not geometry_observations.is_floating_point():
            geometry_observations = geometry_observations.float()
        geometry_observations.requires_grad_(True)

        collector.start(
            batch_size=minibatch_size,
            anchor_indices=anchor_indices,
            device=device,
        )
        collector_started = True

        q_values = q_network(geometry_observations)
        selected_q = q_values.gather(
            1,
            actions.long().reshape(-1, 1),
        ).squeeze(1)

        td_gradient_signal = 2.0 * (
            selected_q.detach().float()
            - targets.float()
        )

        selected_q.sum().backward()
        cross_matrix, diagonal, factors = collector.finish()
        collector_started = False
    finally:
        if collector_started:
            collector.abort()
        for parameter, requires_grad in zip(
            q_network.parameters(),
            original_requires_grad,
        ):
            parameter.requires_grad_(requires_grad)

    sample_weights, diagnostics = (
        compute_nystrom_sample_weights(
            cross_matrix=cross_matrix,
            diagonal=diagonal,
            anchor_indices=anchor_indices,
            rhs=td_gradient_signal,
            use_nystrom=args.nystrom_use_nystrom,
            use_mean_gradient_nullspace=(
                args.nystrom_use_mean_gradient_nullspace
            ),
            use_full_natural_gradient=(
                args.nystrom_use_full_natural_gradient
            ),
            damping=args.nystrom_damping,
            eigenvalue_tolerance=(
                args.nystrom_eigenvalue_tolerance
            ),
        )
    )

    assign_nystrom_reconstructed_gradients(
        factors,
        sample_weights,
    )

    gradient_norm = nn.utils.clip_grad_norm_(
        q_network.parameters(),
        args.max_grad_norm,
    )
    optimizer.step()

    loss = (
        selected_q.detach().float() - targets.float()
    ).square().mean()

    del geometry_observations
    del cross_matrix
    del diagonal
    del factors
    del sample_weights

    return (
        loss.detach(),
        selected_q.detach(),
        gradient_norm.detach(),
        diagnostics,
    )


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if True:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            tags=[args.wandb_tag] if args.wandb_tag else None,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    # env setup
    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.num_envs,
        episodic_life=True,
        reward_clip=True,
        seed=args.seed,
    )
    # EnvPool already exposes num_envs as a read-only property.
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)
    print(f"envs.action_space type = {type(envs.action_space)}")
    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = PQNAtariNetwork(
        envs,
        use_muon_input=args.use_muon_input,
        use_muon_output=args.use_muon_output,
    ).to(device)

    MC_Method = False
    device_type = device.type
    nystrom_collector = None
    nystrom_diagnostics = None
    nystrom_optimizer_step = 0

    if args.optimizer == "NystromSGD":
        if args.nystrom_conv_sample_chunk_size < 1:
            raise ValueError(
                "nystrom_conv_sample_chunk_size must be at least 1."
            )
        if args.nystrom_log_every < 1:
            raise ValueError("nystrom_log_every must be at least 1.")
        if (
            args.nystrom_use_nystrom
            and args.nystrom_directions > args.minibatch_size
        ):
            raise ValueError(
                "nystrom_directions cannot exceed minibatch_size: "
                f"{args.nystrom_directions} > {args.minibatch_size}."
            )

        optimizer = optim.SGD(
            q_network.parameters(),
            momentum=args.momentum,
            lr=args.learning_rate,
            weight_decay=args.nystrom_weight_decay,
        )
        nystrom_collector = JacobianGeometryCollector(
            q_network,
            use_spatial_mean=args.nystrom_use_spatial_mean,
            conv_sample_chunk_size=(
                args.nystrom_conv_sample_chunk_size
            ),
        )
        if device.type == "cuda":
            anchor_generator = torch.Generator(device=device)
        else:
            anchor_generator = torch.Generator()
        anchor_generator.manual_seed(args.seed + 10_000)

    elif args.optimizer == "SGD":
        optimizer = optim.SGD(
            q_network.parameters(),
            momentum=args.momentum,
            lr=args.learning_rate,
        )
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(
            q_network.parameters(),
            betas=(args.momentum, 0.99),
            lr=args.learning_rate,
            eps=1e-5,
        )
    elif args.optimizer == "Muon":
        muon_params, aux_params = q_network.get_split_params()
        ns_steps = 2 if device_type == "cpu" else 5
        param_groups = [
            dict(
                params=muon_params,
                lr=args.learning_rate,
                momentum=args.momentum,
                weight_decay=1e-4,
                use_muon=True,
                ns_steps=ns_steps,
            ),
            dict(
                params=aux_params,
                lr=args.learning_rate ,
                momentum=args.momentum,
                weight_decay=1e-4,
                use_muon=False,
            ),
        ]
        optimizer = MuonWithAuxAdam(param_groups)
    elif args.optimizer == "NorMuon":
        muon_params, aux_params = q_network.get_split_params()
        param_groups = [
            dict(params=muon_params, lr=args.learning_rate, weight_decay=1e-4, use_muon=True),
            dict(params=aux_params, lr=args.learning_rate, weight_decay=1e-4, use_muon=False),
        ]
        optimizer = SingleDeviceNorMuonWithAuxAdam(param_groups)
    elif args.optimizer == "AdaMuon":
        muon_params, aux_params = q_network.get_split_params()
        param_groups = [
            dict(params=muon_params, lr=args.learning_rate, weight_decay=1e-4, use_muon=True),
            dict(params=aux_params, lr=args.learning_rate, weight_decay=1e-4, use_muon=False),
        ]
        optimizer = AdaMuonWithAuxAdam(param_groups)
    elif args.optimizer == "BGD":
        params = BGD.create_unique_param_groups(q_network)
        optimizer = BGD(
            params,
            std_init=0.01,
            mean_eta=args.learning_rate,
            std_eta=10,
            betas=(args.momentum, 0.999, 0.99),
            mc_iters=1,
        )
        MC_Method = True
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # Storage
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)
    avg_returns = deque(maxlen=20)

    # preserve per-parameter-group LR ratios during annealing
    for g in optimizer.param_groups:
        g["initial_lr"] = g["lr"]

    # start
    global_step = 0
    start_time = time.time()
    next_obs = torch.as_tensor(envs.reset(), device=device)
    next_done = torch.zeros(args.num_envs, device=device, dtype=torch.float32)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            for g in optimizer.param_groups:
                g["lr"] = frac * g["initial_lr"]

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            epsilon = linear_schedule(
                args.start_e,
                args.end_e,
                args.exploration_fraction * args.total_timesteps,
                global_step,
            )

            random_actions = torch.randint(0, envs.single_action_space.n, (args.num_envs,), device=device)

            with torch.no_grad():
                if MC_Method:
                    optimizer.randomize_weights(force_std=0)
                q_values = q_network(next_obs)
                max_actions = torch.argmax(q_values, dim=1)
                values[step] = q_values[torch.arange(args.num_envs, device=device), max_actions].flatten()

            explore = torch.rand((args.num_envs,), device=device) < epsilon
            action = torch.where(explore, random_actions, max_actions)
            actions[step] = action

            next_obs_np, reward, next_done_np, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.as_tensor(reward, device=device).view(-1)
            next_obs = torch.as_tensor(next_obs_np, device=device)
            next_done = torch.as_tensor(next_done_np, device=device, dtype=torch.float32)

            for idx, d in enumerate(next_done_np):
                if d and info["lives"][idx] == 0:
                    print(f"global_step={global_step}, episodic_return={info['r'][idx]}")
                    avg_returns.append(info["r"][idx])
                    writer.add_scalar("charts/avg_episodic_return", np.average(avg_returns), global_step)
                    writer.add_scalar("charts/episodic_return", info["r"][idx], global_step)
                    writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)

        # Compute Q(lambda) targets
        with torch.no_grad():
            returns = torch.zeros_like(rewards, device=device)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    if MC_Method:
                        optimizer.randomize_weights(force_std=0)
                    next_value, _ = torch.max(q_network(next_obs), dim=-1)
                    nextnonterminal = 1.0 - next_done.float()
                    returns[t] = rewards[t] + args.gamma * next_value * nextnonterminal
                else:
                    nextnonterminal = 1.0 - dones[t + 1].float()
                    next_value = values[t + 1]
                    returns[t] = (
                        rewards[t]
                        + args.gamma * (args.q_lambda * returns[t + 1] + (1 - args.q_lambda) * next_value) * nextnonterminal
                    )

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_returns = returns.reshape(-1)

        # optimize Q-network
        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                if MC_Method:
                    optimizer.randomize_weights()

                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                if args.optimizer == "NystromSGD":
                    (
                        loss,
                        old_val,
                        nystrom_gradient_norm,
                        nystrom_diagnostics,
                    ) = nystrom_sgd_step(
                        q_network=q_network,
                        optimizer=optimizer,
                        collector=nystrom_collector,
                        observations=b_obs[mb_inds],
                        actions=b_actions[mb_inds],
                        targets=b_returns[mb_inds],
                        args=args,
                        device=device,
                        anchor_generator=anchor_generator,
                    )
                    nystrom_optimizer_step += 1

                    if (
                        nystrom_optimizer_step
                        % args.nystrom_log_every
                        == 0
                    ):
                        writer.add_scalar(
                            "nystrom/effective_rank",
                            nystrom_diagnostics.effective_rank,
                            global_step,
                        )
                        writer.add_scalar(
                            "nystrom/relative_weight_change_from_mse",
                            nystrom_diagnostics.relative_weight_change_from_mse,
                            global_step,
                        )
                        writer.add_scalar(
                            "nystrom/ordinary_weight_norm",
                            nystrom_diagnostics.ordinary_weight_norm,
                            global_step,
                        )
                        writer.add_scalar(
                            "nystrom/whitened_weight_norm",
                            nystrom_diagnostics.whitened_weight_norm,
                            global_step,
                        )
                        writer.add_scalar(
                            "nystrom/retained_condition_number",
                            nystrom_diagnostics.retained_condition_number,
                            global_step,
                        )
                        writer.add_scalar(
                            "nystrom/gradient_norm",
                            nystrom_gradient_norm.item(),
                            global_step,
                        )
                else:
                    old_val = q_network(
                        b_obs[mb_inds]
                    ).gather(
                        1,
                        b_actions[mb_inds].unsqueeze(-1).long(),
                    ).squeeze()
                    loss = F.mse_loss(
                        b_returns[mb_inds],
                        old_val,
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        q_network.parameters(),
                        args.max_grad_norm,
                    )
                    if MC_Method:
                        optimizer.aggregate_grads(1)
                    optimizer.step()

        writer.add_scalar("losses/td_loss", loss.item(), global_step)
        writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
        sps = int(global_step / (time.time() - start_time))
        print("SPS:", sps)
        writer.add_scalar("charts/SPS", sps, global_step)

    if nystrom_collector is not None:
        nystrom_collector.remove()

    envs.close()