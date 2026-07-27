"""
Fresh-minibatch true-Fisher Nyström whitening on CIFAR-10.

This Colab-friendly script compares:
    1. batch_nystrom
    2. muon
    3. adam
    4. sgd

The Nyström method is deliberately nonpersistent:
    * fresh sample-gradient anchors are selected from every minibatch;
    * anchors are not orthogonalized;
    * no covariance EMA is maintained;
    * there are no anchor ages or replacements;
    * no dense persistent R x P parameter-space anchor bank is stored.

The whitening geometry can use either the empirical Fisher or the exact
categorical-model Fisher. The default is the exact categorical Fisher:

    K[i, j] = sum_c sqrt(p_i[c] p_j[c])
                    <grad log p_i[c], grad log p_j[c]>.

This is the Gram matrix of the probability-weighted score embedding and is
positive semidefinite. The geometry is used only to choose sample weights. The
actual update is reconstructed from the ordinary observed-label cross-entropy
gradients, so the learning objective itself is unchanged.

Conv geometry modes:
    "exact_chunked"
        Construct exact per-example Conv2d weight gradients in sample chunks.

    "spatial_mean"
        Average activation patches and output-gradient vectors across output
        positions before forming the Conv2d Gram contribution. The approximation is

            g_i ~= sum_p(delta_i,p) outer mean_p(patch_i,p).

        The implementation does NOT call F.unfold. It makes a zero-copy as_strided
        patch view and immediately reduces the spatial dimensions, so it never
        materializes [B, C_in * k_h * k_w, L] or per-example Conv weight gradients.

The spatial approximation changes only the geometry used to choose sample weights.
The final weighted Conv2d parameter gradient is reconstructed from the original
activations and complete output-gradient maps with torch.nn.grad.conv2d_weight.

No argparse or CLI is used. Edit the constants below and run the file in Colab.
Install the only nonstandard data dependency with:

    !pip install -q datasets
"""

import copy
import csv
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


# ============================================================
# Configuration -- edit these directly in Colab
# ============================================================

DATA_ROOT = "./data"
HF_DATASET_NAME = "uoft-cs/cifar10"
OUTPUT_DIR = "./cifar10_true_fisher_batch_whitening_v3"

METHODS = [
    "batch_nystrom",
    "muon",
    "adam",
    "sgd",
]

SEEDS = [0, 1, 2]
EPOCHS = 30
BATCH_SIZE = 128
TEST_BATCH_SIZE = 512
NUM_WORKERS = 2

# Optional small subsets for debugging. Use None for full CIFAR-10.
TRAIN_SUBSET_SIZE = None
TEST_SUBSET_SIZE = None

WEIGHT_DECAY = 5.0e-4
COSINE_DECAY = True

SGD_LR = 0.05
SGD_MOMENTUM = 0.9

ADAM_LR = 3.0e-4
ADAM_BETAS = (0.9, 0.999)

MUON_LR = 0.02
MUON_MOMENTUM = 0.95
MUON_NS_STEPS = 5
MUON_AUX_LR = 3.0e-4
MUON_AUX_BETAS = (0.9, 0.999)
MUON_EXCLUDE_INPUT_AND_OUTPUT = True

# Which conventional optimizer consumes the fresh batch-Nyström direction.
# Options: "sgd", "adam", "muon".
NYSTROM_OUTER_OPTIMIZER = "sgd"
NYSTROM_OUTER_SGD_LR = 0.05
NYSTROM_OUTER_SGD_MOMENTUM = 0.9
NYSTROM_OUTER_ADAM_LR = 3.0e-4
NYSTROM_OUTER_MUON_LR = 0.02

# Fresh minibatch Nyström geometry.
USE_NYSTROM = False
NYSTROM_DIRECTIONS = 32
USE_MEAN_GRADIENT_NULLSPACE = False
WHITENING_DAMPING = 1.0e-3
EIGENVALUE_TOLERANCE = 1.0e-6

# Geometry used to build the Gram matrix.
# "true_fisher":
#     Sum the score-gradient Gram across every class using sqrt probability
#     weights. For CIFAR-10 this requires 10 geometry forward/backward passes,
#     followed by one observed-label cross-entropy pass for reconstruction.
# "empirical_fisher":
#     Use the observed-label per-example cross-entropy gradients directly.
FISHER_GEOMETRY_MODE = "true_fisher"

# Conv2d geometry used only to construct the sample-space Gram approximation.
# Options: "exact_chunked", "spatial_mean".
CONV_GEOMETRY_MODE = "exact_chunked"
CONV_SAMPLE_CHUNK_SIZE = 16

# Optional one-time correctness test for the zero-copy patch mean. It compares the
# strided implementation with F.unfold(...).mean(dim=2) on several configurations.
RUN_CONV_GEOMETRY_SELF_TEST = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_BFLOAT16 = True
TORCH_DETERMINISTIC = True


# ============================================================
# Reproducibility and Hugging Face CIFAR-10
# ============================================================

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class HuggingFaceCifar10Dataset(Dataset):
    def __init__(self, hugging_face_dataset, transform):
        self.hugging_face_dataset = hugging_face_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hugging_face_dataset)

    def __getitem__(self, index):
        example = self.hugging_face_dataset[index]
        image = example["img"]
        label = int(example["label"])
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def make_loaders(seed: int):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    dataset = load_dataset(HF_DATASET_NAME, cache_dir=DATA_ROOT)
    train_dataset = HuggingFaceCifar10Dataset(dataset["train"], train_transform)
    test_dataset = HuggingFaceCifar10Dataset(dataset["test"], test_transform)

    if TRAIN_SUBSET_SIZE is not None:
        generator = torch.Generator().manual_seed(seed + 10_000)
        indices = torch.randperm(len(train_dataset), generator=generator)
        train_dataset = Subset(train_dataset, indices[:TRAIN_SUBSET_SIZE].tolist())

    if TEST_SUBSET_SIZE is not None:
        test_dataset = Subset(test_dataset, list(range(TEST_SUBSET_SIZE)))

    loader_generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=loader_generator,
        persistent_workers=NUM_WORKERS > 0,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        persistent_workers=NUM_WORKERS > 0,
    )

    return train_loader, test_loader


# ============================================================
# Model -- GroupNorm avoids cross-example gradient mixing
# ============================================================


class SmallCifarCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 96),
            nn.ReLU(inplace=False),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=False),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ============================================================
# Muon baseline / optional outer optimizer
# ============================================================


@torch.no_grad()
def zeropower_via_newton_schulz5(
    matrix: torch.Tensor,
    steps: int = 5,
    epsilon: float = 1.0e-7,
) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError("Newton-Schulz Muon update expects a matrix.")

    x = matrix.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T

    x = x / x.norm().clamp_min(epsilon)
    a, b, c = 3.4445, -4.7750, 2.0315

    for _ in range(steps):
        a_matrix = x @ x.T
        b_matrix = b * a_matrix + c * (a_matrix @ a_matrix)
        x = a * x + b_matrix @ x

    if transposed:
        x = x.T
    return x


class HybridMuon:
    """Muon for selected matrices and AdamW for all remaining parameters."""

    def __init__(
        self,
        model: nn.Module,
        muon_lr: float,
        aux_lr: float,
        momentum: float,
        weight_decay: float,
        ns_steps: int,
        exclude_input_and_output: bool,
    ):
        named_parameters = list(model.named_parameters())
        input_name = "features.0.weight"
        output_name = "classifier.weight"

        self.muon_parameters: List[nn.Parameter] = []
        auxiliary_parameters: List[nn.Parameter] = []

        for name, parameter in named_parameters:
            use_muon = parameter.ndim >= 2
            if exclude_input_and_output and name in {input_name, output_name}:
                use_muon = False

            if use_muon:
                self.muon_parameters.append(parameter)
            else:
                auxiliary_parameters.append(parameter)

        self.muon_group = {
            "params": self.muon_parameters,
            "lr": muon_lr,
            "base_lr": muon_lr,
        }
        self.auxiliary_optimizer = optim.AdamW(
            auxiliary_parameters,
            lr=aux_lr,
            betas=MUON_AUX_BETAS,
            weight_decay=weight_decay,
        )
        for group in self.auxiliary_optimizer.param_groups:
            group["base_lr"] = group["lr"]

        self.param_groups = [self.muon_group] + self.auxiliary_optimizer.param_groups
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.ns_steps = ns_steps
        self.state: Dict[nn.Parameter, torch.Tensor] = {}

    def zero_grad(self, set_to_none: bool = True):
        for parameter in self.muon_parameters:
            if parameter.grad is not None:
                if set_to_none:
                    parameter.grad = None
                else:
                    parameter.grad.zero_()
        self.auxiliary_optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self):
        self.auxiliary_optimizer.step()

        learning_rate = self.muon_group["lr"]
        for parameter in self.muon_parameters:
            if parameter.grad is None:
                continue

            gradient = parameter.grad.detach().float()
            momentum_buffer = self.state.get(parameter)
            if momentum_buffer is None:
                momentum_buffer = torch.zeros_like(parameter, dtype=torch.float32)
                self.state[parameter] = momentum_buffer

            momentum_buffer.mul_(self.momentum).add_(gradient)
            matrix = momentum_buffer.reshape(momentum_buffer.shape[0], -1)
            update = zeropower_via_newton_schulz5(
                matrix,
                steps=self.ns_steps,
            )

            rows, columns = update.shape
            update.mul_(math.sqrt(max(1.0, rows / max(columns, 1))))

            if self.weight_decay != 0.0:
                parameter.mul_(1.0 - learning_rate * self.weight_decay)

            parameter.add_(
                update.reshape_as(parameter).to(dtype=parameter.dtype),
                alpha=-learning_rate,
            )


# ============================================================
# Fresh minibatch sample-gradient geometry collector
# ============================================================


@dataclass
class ModuleFactors:
    module: nn.Module
    inputs: torch.Tensor
    output_gradients: torch.Tensor


@dataclass
class WhiteningDiagnostics:
    effective_rank: int
    mean_absolute_anchor_cosine: float
    maximum_absolute_anchor_cosine: float
    relative_weight_change_from_mean: float
    ordinary_weight_norm: float
    whitened_weight_norm: float
    minimum_retained_eigenvalue: float
    maximum_retained_eigenvalue: float
    retained_condition_number: float


def _pair(value) -> Tuple[int, int]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError("Expected a pair.")
        return int(value[0]), int(value[1])
    return int(value), int(value)


class BatchNystromGeometryCollector:
    """
    Build G G_anchor^T and diag(G G^T) from one backward pass.

    The selected sample anchors are fresh every minibatch and are used as they are;
    there is no anchor orthogonalization or persistent state.
    """

    def __init__(
        self,
        model: nn.Module,
        conv_geometry_mode: str,
        conv_sample_chunk_size: int,
    ):
        if conv_geometry_mode not in {"exact_chunked", "spatial_mean"}:
            raise ValueError(
                "conv_geometry_mode must be 'exact_chunked' or 'spatial_mean'."
            )
        if conv_sample_chunk_size < 1:
            raise ValueError("conv_sample_chunk_size must be at least one.")

        self.model = model
        self.conv_geometry_mode = conv_geometry_mode
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
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.GroupNorm)):
                if isinstance(module, nn.Conv2d) and module.groups != 1:
                    raise NotImplementedError("Grouped Conv2d is not implemented.")

                for parameter in module.parameters(recurse=False):
                    self.parameter_ids.add(id(parameter))

                self.handles.append(module.register_forward_hook(self._save_input))
                self.handles.append(
                    module.register_full_backward_pre_hook(
                        self._capture_output_gradient
                    )
                )

        model_parameter_ids = {id(parameter) for parameter in model.parameters()}
        missing = model_parameter_ids - self.parameter_ids
        if missing:
            raise RuntimeError(
                "The geometry collector does not cover every model parameter. "
                f"Missing parameter count: {len(missing)}"
            )

    def start(
        self,
        batch_size: int,
        anchor_indices: torch.Tensor,
        device: torch.device,
    ):
        if self.collecting:
            raise RuntimeError("Collector is already active.")
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
            raise RuntimeError(f"Missing saved input for {type(module).__name__}.")

        inputs = self.saved_inputs.pop(module)
        output_gradients = grad_output[0].detach()

        with torch.no_grad():
            if isinstance(module, nn.Linear):
                self._add_linear_contribution(module, inputs, output_gradients)
            elif isinstance(module, nn.Conv2d):
                self._add_conv2d_contribution(module, inputs, output_gradients)
            elif isinstance(module, nn.GroupNorm):
                self._add_groupnorm_contribution(module, inputs, output_gradients)
            else:
                raise TypeError(type(module))

            self.factors.append(
                ModuleFactors(
                    module=module,
                    inputs=inputs,
                    output_gradients=output_gradients,
                )
            )

    def _add_linear_contribution(self, module, inputs, output_gradients):
        activations = inputs.reshape(self.batch_size, -1).float()
        gradients = output_gradients.reshape(self.batch_size, -1).float()

        if activations.shape[1] != module.in_features:
            raise RuntimeError(
                "This implementation expects one Linear input vector per sample."
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
    def _conv_exact_weight_gradients(
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
        gradients = output_gradients.float().flatten(start_dim=2)
        if gradients.shape[-1] != unfolded.shape[-1]:
            raise RuntimeError("Conv2d unfolded spatial dimensions do not match.")

        return torch.bmm(
            gradients,
            unfolded.transpose(1, 2),
        ).flatten(start_dim=1)

    @staticmethod
    def _conv_mean_patches_strided(
        module: nn.Conv2d,
        inputs: torch.Tensor,
        output_height: int,
        output_width: int,
    ) -> torch.Tensor:
        """
        Return mean output-position patches with shape [B, C_in * k_h * k_w].

        as_strided creates only a view. The spatial mean is the first materialized
        patch tensor, so [B, C_in * k_h * k_w, L] is never allocated.
        """
        if isinstance(module.padding, str):
            raise NotImplementedError("String Conv2d padding is not implemented.")

        inputs_float = inputs.float()
        kernel_height, kernel_width = _pair(module.kernel_size)
        stride_height, stride_width = _pair(module.stride)
        dilation_height, dilation_width = _pair(module.dilation)
        padding_height, padding_width = _pair(module.padding)

        if module.padding_mode != "zeros":
            if padding_height != 0 or padding_width != 0:
                inputs_float = F.pad(
                    inputs_float,
                    (padding_width, padding_width, padding_height, padding_height),
                    mode=module.padding_mode,
                )
        elif padding_height != 0 or padding_width != 0:
            inputs_float = F.pad(
                inputs_float,
                (padding_width, padding_width, padding_height, padding_height),
            )

        inputs_float = inputs_float.contiguous()
        batch_size, channels, padded_height, padded_width = inputs_float.shape

        expected_height = (
            padded_height
            - dilation_height * (kernel_height - 1)
            - 1
        ) // stride_height + 1
        expected_width = (
            padded_width
            - dilation_width * (kernel_width - 1)
            - 1
        ) // stride_width + 1

        if expected_height != output_height or expected_width != output_width:
            raise RuntimeError(
                "Conv2d strided patch dimensions do not match output gradients: "
                f"expected {(expected_height, expected_width)}, "
                f"received {(output_height, output_width)}."
            )

        stride_batch, stride_channel, stride_row, stride_column = inputs_float.stride()
        patch_view = inputs_float.as_strided(
            size=(
                batch_size,
                channels,
                kernel_height,
                kernel_width,
                output_height,
                output_width,
            ),
            stride=(
                stride_batch,
                stride_channel,
                dilation_height * stride_row,
                dilation_width * stride_column,
                stride_height * stride_row,
                stride_width * stride_column,
            ),
        )

        return patch_view.mean(dim=(-2, -1)).reshape(batch_size, -1)

    def _add_conv2d_exact_chunked_contribution(
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
                torch.arange(self.batch_size, device=anchors.device),
            )
        )

        if full_exact:
            all_weight_gradients = self._conv_exact_weight_gradients(
                module,
                inputs,
                output_gradients,
            )
            self.cross_matrix.add_(all_weight_gradients @ all_weight_gradients.T)
            self.diagonal.add_(all_weight_gradients.square().sum(dim=1))
            del all_weight_gradients
        else:
            anchor_weight_gradients = self._conv_exact_weight_gradients(
                module,
                inputs[anchors],
                output_gradients[anchors],
            )

            for start in range(0, self.batch_size, self.conv_sample_chunk_size):
                end = min(start + self.conv_sample_chunk_size, self.batch_size)
                chunk_weight_gradients = self._conv_exact_weight_gradients(
                    module,
                    inputs[start:end],
                    output_gradients[start:end],
                )
                self.cross_matrix[start:end].add_(
                    chunk_weight_gradients @ anchor_weight_gradients.T
                )
                self.diagonal[start:end].add_(
                    chunk_weight_gradients.square().sum(dim=1)
                )
                del chunk_weight_gradients

            del anchor_weight_gradients

    def _add_conv2d_spatial_mean_contribution(
        self,
        module: nn.Conv2d,
        inputs: torch.Tensor,
        output_gradients: torch.Tensor,
    ):
        """
        Approximate each sample's Conv weight gradient by

            sum_p(delta_p) outer mean_p(patch_p).

        No F.unfold or per-sample weight-gradient tensor is constructed.
        """
        gradients_float = output_gradients.float()
        output_height, output_width = gradients_float.shape[-2:]
        mean_patches = self._conv_mean_patches_strided(
            module,
            inputs,
            output_height,
            output_width,
        )
        summed_output_gradients = gradients_float.sum(dim=(2, 3))
        anchors = self.anchor_indices

        activation_cross = mean_patches @ mean_patches[anchors].T
        gradient_cross = (
            summed_output_gradients @ summed_output_gradients[anchors].T
        )

        self.cross_matrix.add_(activation_cross * gradient_cross)
        self.diagonal.add_(
            mean_patches.square().sum(dim=1)
            * summed_output_gradients.square().sum(dim=1)
        )

        return summed_output_gradients

    def _add_conv2d_contribution(self, module, inputs, output_gradients):
        summed_output_gradients = None

        if self.conv_geometry_mode == "exact_chunked":
            self._add_conv2d_exact_chunked_contribution(
                module,
                inputs,
                output_gradients,
            )
        else:
            summed_output_gradients = self._add_conv2d_spatial_mean_contribution(
                module,
                inputs,
                output_gradients,
            )

        if module.bias is not None:
            if summed_output_gradients is None:
                summed_output_gradients = output_gradients.float().sum(dim=(2, 3))
            anchors = self.anchor_indices
            self.cross_matrix.add_(
                summed_output_gradients @ summed_output_gradients[anchors].T
            )
            self.diagonal.add_(summed_output_gradients.square().sum(dim=1))

    @staticmethod
    def _groupnorm_local_gradients(
        module: nn.GroupNorm,
        inputs: torch.Tensor,
        output_gradients: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs_float = inputs.float()
        gradients_float = output_gradients.float()

        batch_size, channels = inputs_float.shape[:2]
        grouped = inputs_float.reshape(batch_size, module.num_groups, -1)
        mean = grouped.mean(dim=2, keepdim=True)
        variance = grouped.var(dim=2, unbiased=False, keepdim=True)
        normalized = (grouped - mean) * torch.rsqrt(variance + module.eps)
        normalized = normalized.reshape_as(inputs_float)

        spatial_dims = tuple(range(2, inputs_float.ndim))
        scale_local = (gradients_float * normalized).sum(dim=spatial_dims)
        bias_local = gradients_float.sum(dim=spatial_dims)

        if scale_local.shape != (batch_size, channels):
            raise RuntimeError("Unexpected GroupNorm scale-gradient shape.")
        if bias_local.shape != (batch_size, channels):
            raise RuntimeError("Unexpected GroupNorm bias-gradient shape.")

        return scale_local, bias_local

    def _add_groupnorm_contribution(self, module, inputs, output_gradients):
        scale_local, bias_local = self._groupnorm_local_gradients(
            module,
            inputs,
            output_gradients,
        )
        anchors = self.anchor_indices

        if module.affine and module.weight is not None:
            self.cross_matrix.add_(scale_local @ scale_local[anchors].T)
            self.diagonal.add_(scale_local.square().sum(dim=1))
        if module.affine and module.bias is not None:
            self.cross_matrix.add_(bias_local @ bias_local[anchors].T)
            self.diagonal.add_(bias_local.square().sum(dim=1))

    def finish(self):
        if not self.collecting:
            raise RuntimeError("Collector was not started.")
        if self.saved_inputs:
            raise RuntimeError(
                f"{len(self.saved_inputs)} saved module inputs were not consumed."
            )

        result = {
            "cross_matrix": self.cross_matrix,
            "diagonal": self.diagonal,
            "factors": self.factors,
        }

        self.collecting = False
        self.batch_size = 0
        self.anchor_indices = None
        self.cross_matrix = None
        self.diagonal = None
        self.factors = []
        return result

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


# ============================================================
# Fresh Nyström solve and exact weighted-gradient reconstruction
# ============================================================


def choose_anchor_indices(
    batch_size: int,
    use_nystrom: bool,
    nystrom_directions: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    if not use_nystrom:
        return torch.arange(batch_size, device=device)
    if nystrom_directions < 1:
        raise ValueError("NYSTROM_DIRECTIONS must be at least one.")
    if nystrom_directions > batch_size:
        raise ValueError("NYSTROM_DIRECTIONS cannot exceed BATCH_SIZE.")

    return torch.randperm(
        batch_size,
        device=device,
        generator=generator,
    )[:nystrom_directions]


@torch.no_grad()
def _retained_subspace_weights(
    eigenvectors: torch.Tensor,
    raw_eigenvalues: torch.Tensor,
    rhs: torch.Tensor,
    ordinary_weights: torch.Tensor,
    batch_size: int,
    ridge: torch.Tensor,
    use_mean_gradient_nullspace: bool,
):
    if raw_eigenvalues.numel() == 0:
        if use_mean_gradient_nullspace:
            return ordinary_weights, 0, 0.0, 0.0, float("inf")
        return torch.zeros_like(ordinary_weights), 0, 0.0, 0.0, float("inf")

    fisher_eigenvalues = raw_eigenvalues / batch_size
    projected_rhs = eigenvectors.T @ rhs
    whitened_weights = (
        eigenvectors
        @ (projected_rhs * (fisher_eigenvalues + ridge).rsqrt())
    ) / batch_size

    if use_mean_gradient_nullspace:
        ordinary_retained = (eigenvectors @ projected_rhs) / batch_size
        weights = whitened_weights + ordinary_weights - ordinary_retained
    else:
        weights = whitened_weights

    minimum = fisher_eigenvalues.min().item()
    maximum = fisher_eigenvalues.max().item()
    condition = maximum / max(minimum, 1.0e-30)
    return weights, raw_eigenvalues.numel(), minimum, maximum, condition


@torch.no_grad()
def compute_whitening_weights(
    cross_matrix: torch.Tensor,
    diagonal: torch.Tensor,
    anchor_indices: torch.Tensor,
    use_nystrom: bool,
    use_mean_gradient_nullspace: bool,
    damping: float,
    eigenvalue_tolerance: float,
):
    batch_size = diagonal.numel()

    # Each row of G is already a complete per-example loss gradient. Therefore the
    # ordinary mean gradient is G^T (1 / B), so the right-hand side is all ones.
    rhs = torch.ones(batch_size, device=diagonal.device, dtype=torch.float32)
    ordinary_weights = rhs / batch_size

    mean_diagonal_fisher = (diagonal.mean() / batch_size).clamp_min(1.0e-30)
    ridge = damping * mean_diagonal_fisher

    if not use_nystrom:
        gram = 0.5 * (cross_matrix + cross_matrix.T)
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        largest = eigenvalues.max().clamp_min(0.0)
        threshold = max(largest.item() * eigenvalue_tolerance, 1.0e-20)
        keep = eigenvalues > threshold
        retained_values = eigenvalues[keep]
        retained_vectors = eigenvectors[:, keep]
    else:
        anchor_matrix = cross_matrix[anchor_indices]
        anchor_matrix = 0.5 * (anchor_matrix + anchor_matrix.T)
        anchor_values, anchor_vectors = torch.linalg.eigh(anchor_matrix)
        largest_anchor = anchor_values.max().clamp_min(0.0)
        anchor_threshold = max(
            largest_anchor.item() * eigenvalue_tolerance,
            1.0e-20,
        )
        keep_anchor = anchor_values > anchor_threshold

        if not torch.any(keep_anchor):
            retained_vectors = cross_matrix.new_zeros((batch_size, 0))
            retained_values = cross_matrix.new_zeros((0,))
        else:
            anchor_values = anchor_values[keep_anchor]
            anchor_vectors = anchor_vectors[:, keep_anchor]
            anchor_inverse_square_root = (
                anchor_vectors * anchor_values.rsqrt()[None, :]
            ) @ anchor_vectors.T

            z_matrix = cross_matrix @ anchor_inverse_square_root
            small_matrix = z_matrix.T @ z_matrix
            small_matrix = 0.5 * (small_matrix + small_matrix.T)
            small_values, small_vectors = torch.linalg.eigh(small_matrix)
            largest_small = small_values.max().clamp_min(0.0)
            small_threshold = max(
                largest_small.item() * eigenvalue_tolerance,
                1.0e-20,
            )
            keep_small = small_values > small_threshold
            retained_values = small_values[keep_small]

            if retained_values.numel() == 0:
                retained_vectors = cross_matrix.new_zeros((batch_size, 0))
            else:
                retained_vectors = (
                    z_matrix @ small_vectors[:, keep_small]
                ) * retained_values.rsqrt()[None, :]

    weights, rank, minimum, maximum, condition = _retained_subspace_weights(
        retained_vectors,
        retained_values,
        rhs,
        ordinary_weights,
        batch_size,
        ridge,
        use_mean_gradient_nullspace,
    )

    anchor_diagonal = diagonal[anchor_indices]
    denominator = torch.sqrt(
        diagonal[:, None].clamp_min(1.0e-30)
        * anchor_diagonal[None, :].clamp_min(1.0e-30)
    )
    absolute_cosines = (cross_matrix / denominator).abs()
    self_mask = torch.ones_like(absolute_cosines, dtype=torch.bool)
    row_indices = torch.arange(batch_size, device=cross_matrix.device)[:, None]
    self_mask &= row_indices != anchor_indices[None, :]
    valid_cosines = absolute_cosines[self_mask]

    if valid_cosines.numel() == 0:
        mean_cosine = 0.0
        maximum_cosine = 0.0
    else:
        mean_cosine = valid_cosines.mean().item()
        maximum_cosine = valid_cosines.max().item()

    ordinary_norm = torch.linalg.vector_norm(ordinary_weights)
    whitened_norm = torch.linalg.vector_norm(weights)
    relative_change = (
        torch.linalg.vector_norm(weights - ordinary_weights)
        / ordinary_norm.clamp_min(1.0e-30)
    ).item()

    diagnostics = WhiteningDiagnostics(
        effective_rank=rank,
        mean_absolute_anchor_cosine=mean_cosine,
        maximum_absolute_anchor_cosine=maximum_cosine,
        relative_weight_change_from_mean=relative_change,
        ordinary_weight_norm=ordinary_norm.item(),
        whitened_weight_norm=whitened_norm.item(),
        minimum_retained_eigenvalue=minimum,
        maximum_retained_eigenvalue=maximum,
        retained_condition_number=condition,
    )
    return weights, diagnostics


@torch.no_grad()
def assign_reconstructed_gradients(
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
            gradients = output_gradients.reshape(batch_size, -1).float()
            weighted_gradients = gradients * sample_weights[:, None]

            module.weight.grad = (weighted_gradients.T @ activations).to(
                dtype=module.weight.dtype
            )
            if module.bias is not None:
                module.bias.grad = weighted_gradients.sum(dim=0).to(
                    dtype=module.bias.dtype
                )

        elif isinstance(module, nn.Conv2d):
            weighted_output_gradients = output_gradients.float() * sample_weights.view(
                batch_size,
                1,
                1,
                1,
            )
            module.weight.grad = torch.nn.grad.conv2d_weight(
                inputs.float(),
                module.weight.shape,
                weighted_output_gradients,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
            ).to(dtype=module.weight.dtype)

            if module.bias is not None:
                module.bias.grad = weighted_output_gradients.sum(dim=(0, 2, 3)).to(
                    dtype=module.bias.dtype
                )

        elif isinstance(module, nn.GroupNorm):
            inputs_float = inputs.float()
            gradients_float = output_gradients.float()
            grouped = inputs_float.reshape(batch_size, module.num_groups, -1)
            mean = grouped.mean(dim=2, keepdim=True)
            variance = grouped.var(dim=2, unbiased=False, keepdim=True)
            normalized = (grouped - mean) * torch.rsqrt(variance + module.eps)
            normalized = normalized.reshape_as(inputs_float)
            weight_view = sample_weights.view(batch_size, 1, 1, 1)
            spatial_dims = (0, 2, 3)

            if module.affine and module.weight is not None:
                module.weight.grad = (
                    gradients_float * normalized * weight_view
                ).sum(dim=spatial_dims).to(dtype=module.weight.dtype)

            if module.affine and module.bias is not None:
                module.bias.grad = (
                    gradients_float * weight_view
                ).sum(dim=spatial_dims).to(dtype=module.bias.dtype)
        else:
            raise TypeError(type(module))


# ============================================================
# Optimizer construction and scheduling
# ============================================================


def attach_base_learning_rates(optimizer):
    for group in optimizer.param_groups:
        group.setdefault("base_lr", group["lr"])


def build_standard_optimizer(method: str, model: nn.Module):
    if method == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=SGD_LR,
            momentum=SGD_MOMENTUM,
            weight_decay=WEIGHT_DECAY,
        )
    elif method == "adam":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=ADAM_LR,
            betas=ADAM_BETAS,
            weight_decay=WEIGHT_DECAY,
        )
    elif method == "muon":
        optimizer = HybridMuon(
            model=model,
            muon_lr=MUON_LR,
            aux_lr=MUON_AUX_LR,
            momentum=MUON_MOMENTUM,
            weight_decay=WEIGHT_DECAY,
            ns_steps=MUON_NS_STEPS,
            exclude_input_and_output=MUON_EXCLUDE_INPUT_AND_OUTPUT,
        )
    else:
        raise ValueError(f"Unknown standard method: {method}")

    attach_base_learning_rates(optimizer)
    return optimizer


def build_nystrom_outer_optimizer(model: nn.Module):
    mode = NYSTROM_OUTER_OPTIMIZER.lower()

    if mode == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=NYSTROM_OUTER_SGD_LR,
            momentum=NYSTROM_OUTER_SGD_MOMENTUM,
            weight_decay=WEIGHT_DECAY,
        )
    elif mode == "adam":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=NYSTROM_OUTER_ADAM_LR,
            betas=ADAM_BETAS,
            weight_decay=WEIGHT_DECAY,
        )
    elif mode == "muon":
        optimizer = HybridMuon(
            model=model,
            muon_lr=NYSTROM_OUTER_MUON_LR,
            aux_lr=MUON_AUX_LR,
            momentum=MUON_MOMENTUM,
            weight_decay=WEIGHT_DECAY,
            ns_steps=MUON_NS_STEPS,
            exclude_input_and_output=MUON_EXCLUDE_INPUT_AND_OUTPUT,
        )
    else:
        raise ValueError(
            "NYSTROM_OUTER_OPTIMIZER must be 'sgd', 'adam', or 'muon'."
        )

    attach_base_learning_rates(optimizer)
    return optimizer


def learning_rate_multiplier(epoch_fraction: float) -> float:
    if not COSINE_DECAY:
        return 1.0
    return 0.5 * (1.0 + math.cos(math.pi * epoch_fraction / EPOCHS))


def set_optimizer_lr_multiplier(optimizer, multiplier: float):
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"] * multiplier


# ============================================================
# Training steps
# ============================================================


def autocast_context():
    enabled = (
        USE_BFLOAT16
        and DEVICE.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type=DEVICE.type, enabled=False)


@torch.no_grad()
def gradient_l2_norm(model: nn.Module) -> torch.Tensor:
    squared = torch.zeros((), device=DEVICE, dtype=torch.float32)
    for parameter in model.parameters():
        if parameter.grad is not None:
            squared.add_(parameter.grad.detach().float().square().sum())
    return torch.sqrt(squared.clamp_min(0.0))


def set_parameters_trainable(model: nn.Module, trainable: bool):
    for parameter in model.parameters():
        parameter.requires_grad_(trainable)


def ordinary_step(model, optimizer, inputs, targets):
    optimizer.zero_grad(set_to_none=True)
    with autocast_context():
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
    loss.backward()
    grad_norm = gradient_l2_norm(model).item()
    optimizer.step()
    return loss.detach(), logits.detach(), {"gradient_norm": grad_norm}


def _collect_empirical_fisher_geometry(
    model: nn.Module,
    collector: BatchNystromGeometryCollector,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    anchor_indices: torch.Tensor,
):
    """Collect the observed-label empirical-Fisher Gram and CE factors."""
    geometry_inputs = inputs.detach().requires_grad_(True)
    collector.start(
        batch_size=targets.numel(),
        anchor_indices=anchor_indices,
        device=inputs.device,
    )

    try:
        with autocast_context():
            logits = model(geometry_inputs)
            per_sample_losses = F.cross_entropy(
                logits,
                targets,
                reduction="none",
            )

        per_sample_losses.sum().backward()
        geometry = collector.finish()
    except Exception:
        collector.abort()
        raise

    return geometry, per_sample_losses, logits, geometry_inputs


def _collect_true_categorical_fisher(
    model: nn.Module,
    collector: BatchNystromGeometryCollector,
    inputs: torch.Tensor,
    anchor_indices: torch.Tensor,
):
    """
    Build the exact categorical-model Fisher in sample space.

    For sample i and class c, define the score vector

        s_i,c = grad_theta log p_theta(c | x_i).

    The collector accumulates the Gram of

        sqrt(p_i,c) * s_i,c

    independently for every class, giving

        K_i,j = sum_c sqrt(p_i,c p_j,c) <s_i,c, s_j,c>.

    The square-root weighting is required for a symmetric PSD Gram matrix.
    Class probabilities are detached, as required by the Fisher expectation.
    """
    batch_size = inputs.shape[0]

    # Probabilities define the expectation distribution but must not themselves
    # contribute derivatives to the score vectors.
    with torch.no_grad():
        with autocast_context():
            probability_logits = model(inputs)
        probabilities = probability_logits.float().softmax(dim=1)

    number_of_classes = probabilities.shape[1]
    fisher_cross = torch.zeros(
        batch_size,
        anchor_indices.numel(),
        device=inputs.device,
        dtype=torch.float32,
    )
    fisher_diagonal = torch.zeros(
        batch_size,
        device=inputs.device,
        dtype=torch.float32,
    )

    for class_index in range(number_of_classes):
        geometry_inputs = inputs.detach().requires_grad_(True)
        collector.start(
            batch_size=batch_size,
            anchor_indices=anchor_indices,
            device=inputs.device,
        )

        try:
            with autocast_context():
                logits = model(geometry_inputs)
                log_probabilities = F.log_softmax(logits, dim=1)

            # Multiplying each score by sqrt(p_c) makes the accumulated Gram equal
            # to the probability-weighted Fisher inner product.
            class_weights = probabilities[:, class_index].clamp_min(0.0).sqrt()
            weighted_score_sum = (
                class_weights * log_probabilities[:, class_index].float()
            ).sum()
            weighted_score_sum.backward()
            class_geometry = collector.finish()
        except Exception:
            collector.abort()
            raise

        fisher_cross.add_(class_geometry["cross_matrix"])
        fisher_diagonal.add_(class_geometry["diagonal"])

        del geometry_inputs
        del logits
        del log_probabilities
        del class_geometry

    return fisher_cross, fisher_diagonal, probabilities


def _collect_cross_entropy_reconstruction_factors(
    model: nn.Module,
    collector: BatchNystromGeometryCollector,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    anchor_indices: torch.Tensor,
):
    """
    Capture the real observed-label CE gradients used for the optimizer update.

    The Gram produced in this pass is intentionally discarded when true Fisher
    geometry is selected. Only the saved module factors are retained.
    """
    geometry_inputs = inputs.detach().requires_grad_(True)
    collector.start(
        batch_size=targets.numel(),
        anchor_indices=anchor_indices,
        device=inputs.device,
    )

    try:
        with autocast_context():
            logits = model(geometry_inputs)
            per_sample_losses = F.cross_entropy(
                logits,
                targets,
                reduction="none",
            )

        per_sample_losses.sum().backward()
        reconstruction_geometry = collector.finish()
    except Exception:
        collector.abort()
        raise

    return (
        reconstruction_geometry["factors"],
        per_sample_losses,
        logits,
        geometry_inputs,
    )


def batch_nystrom_step(
    model: nn.Module,
    optimizer,
    collector: BatchNystromGeometryCollector,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    anchor_generator: torch.Generator,
):
    optimizer.zero_grad(set_to_none=True)
    batch_size = targets.numel()
    anchor_indices = choose_anchor_indices(
        batch_size=batch_size,
        use_nystrom=USE_NYSTROM,
        nystrom_directions=NYSTROM_DIRECTIONS,
        device=inputs.device,
        generator=anchor_generator,
    )

    # Parameter gradients are reconstructed explicitly after the geometry solve.
    # Freezing parameters prevents all Fisher passes from allocating .grad tensors.
    set_parameters_trainable(model, False)

    try:
        if FISHER_GEOMETRY_MODE == "true_fisher":
            cross_matrix, diagonal, probabilities = (
                _collect_true_categorical_fisher(
                    model=model,
                    collector=collector,
                    inputs=inputs,
                    anchor_indices=anchor_indices,
                )
            )

            (
                reconstruction_factors,
                per_sample_losses,
                logits,
                reconstruction_inputs,
            ) = _collect_cross_entropy_reconstruction_factors(
                model=model,
                collector=collector,
                inputs=inputs,
                targets=targets,
                anchor_indices=anchor_indices,
            )
        elif FISHER_GEOMETRY_MODE == "empirical_fisher":
            (
                empirical_geometry,
                per_sample_losses,
                logits,
                reconstruction_inputs,
            ) = _collect_empirical_fisher_geometry(
                model=model,
                collector=collector,
                inputs=inputs,
                targets=targets,
                anchor_indices=anchor_indices,
            )
            cross_matrix = empirical_geometry["cross_matrix"]
            diagonal = empirical_geometry["diagonal"]
            reconstruction_factors = empirical_geometry["factors"]
            probabilities = None
        else:
            raise ValueError(
                "FISHER_GEOMETRY_MODE must be 'true_fisher' or "
                "'empirical_fisher'."
            )
    except Exception:
        set_parameters_trainable(model, True)
        raise

    set_parameters_trainable(model, True)

    sample_weights, whitening = compute_whitening_weights(
        cross_matrix=cross_matrix,
        diagonal=diagonal,
        anchor_indices=anchor_indices,
        use_nystrom=USE_NYSTROM,
        use_mean_gradient_nullspace=USE_MEAN_GRADIENT_NULLSPACE,
        damping=WHITENING_DAMPING,
        eigenvalue_tolerance=EIGENVALUE_TOLERANCE,
    )

    # The true Fisher changes only the geometry. The parameter update is still
    # the weighted sum of the actual observed-label cross-entropy gradients.
    assign_reconstructed_gradients(
        reconstruction_factors,
        sample_weights,
    )

    grad_norm = gradient_l2_norm(model).item()
    optimizer.step()

    diagnostics = {
        "gradient_norm": grad_norm,
        "effective_rank": float(whitening.effective_rank),
        "mean_absolute_anchor_cosine": whitening.mean_absolute_anchor_cosine,
        "maximum_absolute_anchor_cosine": whitening.maximum_absolute_anchor_cosine,
        "relative_weight_change_from_mean": (
            whitening.relative_weight_change_from_mean
        ),
        "ordinary_weight_norm": whitening.ordinary_weight_norm,
        "whitened_weight_norm": whitening.whitened_weight_norm,
        "minimum_retained_eigenvalue": whitening.minimum_retained_eigenvalue,
        "maximum_retained_eigenvalue": whitening.maximum_retained_eigenvalue,
        "retained_condition_number": whitening.retained_condition_number,
    }

    if probabilities is not None:
        entropy = -(
            probabilities
            * probabilities.clamp_min(1.0e-30).log()
        ).sum(dim=1).mean()
        diagnostics["mean_predictive_entropy"] = entropy.item()

    loss = per_sample_losses.detach().mean()
    logits = logits.detach()

    del reconstruction_inputs
    del reconstruction_factors
    del cross_matrix
    del diagonal
    del sample_weights
    if probabilities is not None:
        del probabilities
    if FISHER_GEOMETRY_MODE == "empirical_fisher":
        del empirical_geometry

    return loss, logits, diagnostics


# ============================================================
# Optional spatial-mean correctness self-test
# ============================================================


def run_conv_geometry_self_test():
    print("Running strided spatial-mean Conv patch self-test...")
    test_cases = [
        dict(in_channels=3, out_channels=5, kernel_size=3, stride=1, padding=1, dilation=1),
        dict(in_channels=3, out_channels=5, kernel_size=3, stride=2, padding=1, dilation=1),
        dict(in_channels=4, out_channels=6, kernel_size=(2, 3), stride=(2, 1), padding=(1, 2), dilation=(1, 2)),
    ]

    generator = torch.Generator(device=DEVICE.type).manual_seed(1234)
    for case_index, case in enumerate(test_cases):
        module = nn.Conv2d(**case, bias=False).to(DEVICE)
        inputs = torch.randn(
            4,
            case["in_channels"],
            13,
            15,
            device=DEVICE,
            generator=generator,
        )
        output = module(inputs)

        reference = F.unfold(
            inputs.float(),
            kernel_size=module.kernel_size,
            dilation=module.dilation,
            padding=module.padding,
            stride=module.stride,
        ).mean(dim=2)
        candidate = BatchNystromGeometryCollector._conv_mean_patches_strided(
            module,
            inputs,
            output.shape[-2],
            output.shape[-1],
        )

        torch.testing.assert_close(
            candidate,
            reference,
            rtol=1.0e-5,
            atol=1.0e-6,
        )
        print(f"  case {case_index + 1}: passed")

    print("Spatial-mean Conv patch self-test passed.\n")


# ============================================================
# Evaluation and experiment loop
# ============================================================


@torch.inference_mode()
def evaluate(model: nn.Module, test_loader):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for inputs, targets in test_loader:
        inputs = inputs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        with autocast_context():
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets, reduction="sum")

        total_loss += loss.item()
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_examples += targets.numel()

    return total_loss / total_examples, 100.0 * total_correct / total_examples


def train_method(method: str, seed: int, initial_state):
    set_seed(seed)
    train_loader, test_loader = make_loaders(seed)

    model = SmallCifarCNN().to(DEVICE)
    model.load_state_dict(initial_state)
    model.train()

    collector = None
    anchor_generator = None

    if method == "batch_nystrom":
        optimizer = build_nystrom_outer_optimizer(model)
        collector = BatchNystromGeometryCollector(
            model=model,
            conv_geometry_mode=CONV_GEOMETRY_MODE,
            conv_sample_chunk_size=CONV_SAMPLE_CHUNK_SIZE,
        )
        if DEVICE.type == "cuda":
            anchor_generator = torch.Generator(device="cuda")
        else:
            anchor_generator = torch.Generator()
        anchor_generator.manual_seed(seed + 100_000)
    else:
        optimizer = build_standard_optimizer(method, model)

    history = []

    try:
        for epoch in range(EPOCHS):
            model.train()
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            epoch_start = time.perf_counter()

            total_loss = 0.0
            total_correct = 0
            total_examples = 0
            diagnostic_sums = defaultdict(float)
            diagnostic_counts = defaultdict(int)

            for batch_index, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)

                epoch_fraction = epoch + batch_index / len(train_loader)
                multiplier = learning_rate_multiplier(epoch_fraction)
                set_optimizer_lr_multiplier(optimizer, multiplier)

                if method == "batch_nystrom":
                    loss, logits, diagnostics = batch_nystrom_step(
                        model=model,
                        optimizer=optimizer,
                        collector=collector,
                        inputs=inputs,
                        targets=targets,
                        anchor_generator=anchor_generator,
                    )
                else:
                    loss, logits, diagnostics = ordinary_step(
                        model,
                        optimizer,
                        inputs,
                        targets,
                    )

                current_batch_size = targets.numel()
                total_loss += loss.item() * current_batch_size
                total_correct += (logits.argmax(dim=1) == targets).sum().item()
                total_examples += current_batch_size

                for key, value in diagnostics.items():
                    value = float(value)
                    if math.isfinite(value):
                        diagnostic_sums[key] += value
                        diagnostic_counts[key] += 1

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            epoch_seconds = time.perf_counter() - epoch_start

            train_loss = total_loss / total_examples
            train_accuracy = 100.0 * total_correct / total_examples
            test_loss, test_accuracy = evaluate(model, test_loader)

            row = {
                "method": method,
                "seed": seed,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "epoch_seconds": epoch_seconds,
                "lr_multiplier_end": learning_rate_multiplier(epoch + 1),
                "conv_geometry_mode": (
                    CONV_GEOMETRY_MODE
                    if method == "batch_nystrom"
                    else "not_applicable"
                ),
                "fisher_geometry_mode": (
                    FISHER_GEOMETRY_MODE
                    if method == "batch_nystrom"
                    else "not_applicable"
                ),
                "nystrom_outer_optimizer": (
                    NYSTROM_OUTER_OPTIMIZER
                    if method == "batch_nystrom"
                    else "not_applicable"
                ),
                "nystrom_directions": (
                    NYSTROM_DIRECTIONS
                    if method == "batch_nystrom" and USE_NYSTROM
                    else BATCH_SIZE if method == "batch_nystrom" else 0
                ),
            }

            for key in sorted(diagnostic_sums):
                row[key] = diagnostic_sums[key] / diagnostic_counts[key]

            history.append(row)

            print(
                f"[{method:16s} seed={seed} epoch={epoch + 1:02d}/{EPOCHS}] "
                f"train loss={train_loss:.4f}  "
                f"train acc={train_accuracy:6.2f}%  "
                f"test acc={test_accuracy:6.2f}%  "
                f"time={epoch_seconds:7.2f}s"
            )
    finally:
        if collector is not None:
            collector.remove()

    return history


# ============================================================
# Saving and plotting
# ============================================================


def write_history_csv(rows, path):
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_curves(rows, metric):
    grouped = defaultdict(list)
    for row in rows:
        if metric in row and isinstance(row[metric], (int, float)):
            grouped[(row["method"], row["epoch"])].append(row[metric])

    curves = defaultdict(lambda: {"epoch": [], "mean": [], "std": []})
    for (method, epoch), values in sorted(grouped.items()):
        values = np.asarray(values, dtype=np.float64)
        curves[method]["epoch"].append(epoch)
        curves[method]["mean"].append(values.mean())
        curves[method]["std"].append(values.std(ddof=0))

    return curves


def save_curve_plot(rows, metric, ylabel, filename, methods=None):
    import matplotlib.pyplot as plt

    methods = METHODS if methods is None else methods
    curves = aggregate_curves(rows, metric)

    plt.figure(figsize=(8, 5))
    for method in methods:
        if method not in curves or not curves[method]["epoch"]:
            continue
        curve = curves[method]
        epoch = np.asarray(curve["epoch"])
        mean = np.asarray(curve["mean"])
        std = np.asarray(curve["std"])
        plt.plot(epoch, mean, label=method)
        plt.fill_between(epoch, mean - std, mean + std, alpha=0.2)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=180)
    plt.close()


def main():
    if USE_NYSTROM and NYSTROM_DIRECTIONS > BATCH_SIZE:
        raise ValueError("NYSTROM_DIRECTIONS cannot exceed BATCH_SIZE.")
    if CONV_GEOMETRY_MODE not in {"exact_chunked", "spatial_mean"}:
        raise ValueError("Invalid CONV_GEOMETRY_MODE.")
    if NYSTROM_OUTER_OPTIMIZER.lower() not in {"sgd", "adam", "muon"}:
        raise ValueError("Invalid NYSTROM_OUTER_OPTIMIZER.")
    if FISHER_GEOMETRY_MODE not in {"true_fisher", "empirical_fisher"}:
        raise ValueError("Invalid FISHER_GEOMETRY_MODE.")

    torch.backends.cudnn.deterministic = TORCH_DETERMINISTIC
    torch.backends.cudnn.benchmark = not TORCH_DETERMINISTIC

    if RUN_CONV_GEOMETRY_SELF_TEST:
        run_conv_geometry_self_test()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"device={DEVICE}")
    print(f"methods={METHODS}")
    print(f"nystrom_outer_optimizer={NYSTROM_OUTER_OPTIMIZER}")
    print(
        "nystrom_directions="
        f"{NYSTROM_DIRECTIONS if USE_NYSTROM else 'FULL_GRAM'}"
    )
    print(f"conv_geometry_mode={CONV_GEOMETRY_MODE}")
    print(f"fisher_geometry_mode={FISHER_GEOMETRY_MODE}")
    print("persistent_nystrom=False")
    print("anchor_orthogonalization=False")

    all_history = []

    for seed in SEEDS:
        set_seed(seed)
        reference_model = SmallCifarCNN().to(DEVICE)
        initial_state = copy.deepcopy(reference_model.state_dict())
        del reference_model

        for method in METHODS:
            history = train_method(method, seed, initial_state)
            all_history.extend(history)
            write_history_csv(
                all_history,
                os.path.join(OUTPUT_DIR, "training_history_partial.csv"),
            )

    history_path = os.path.join(OUTPUT_DIR, "training_history.csv")
    write_history_csv(all_history, history_path)

    save_curve_plot(
        all_history,
        metric="test_accuracy",
        ylabel="Test accuracy (%)",
        filename="test_accuracy.png",
    )
    save_curve_plot(
        all_history,
        metric="train_loss",
        ylabel="Training loss",
        filename="training_loss.png",
    )
    save_curve_plot(
        all_history,
        metric="epoch_seconds",
        ylabel="Epoch time (seconds)",
        filename="epoch_time.png",
    )
    save_curve_plot(
        all_history,
        metric="effective_rank",
        ylabel="Nyström effective rank",
        filename="nystrom_effective_rank.png",
        methods=["batch_nystrom"],
    )
    save_curve_plot(
        all_history,
        metric="relative_weight_change_from_mean",
        ylabel="Relative sample-weight change",
        filename="sample_weight_change.png",
        methods=["batch_nystrom"],
    )

    print(f"\nSaved results to: {OUTPUT_DIR}")
    print(f"History CSV: {history_path}")


if __name__ == "__main__":
    main()