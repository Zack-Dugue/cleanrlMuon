from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor
import math

from typing import Iterable, List, Dict, Any, Tuple

import math
import torch
from torch import Tensor
from torch.optim import Optimizer

# ----- your existing zeropower_via_newtonschulz5 -----
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


# ==============================
# AdaMuonWithAuxAdam (with aux AdamW)
# ==============================
class AdaMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Mixed optimizer:
      - AdaMuon path for groups with use_muon=True
      - AdamW-style path for groups with use_muon=False

    Param group examples:
        dict(params=[...2D+...], use_muon=True,  lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, eps=1e-8, weight_decay=0.01)
        dict(params=[...bias/embeds...], use_muon=False, lr=3e-4, betas=(0.9,0.95), eps=1e-10, weight_decay=0.0)
    """

    def __init__(self, param_groups, *, rank: int | None = None, world_size: int | None = None):
        # ---- Change 1: auto-detect distributed, default to single-GPU ----
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank() if rank is None else int(rank)
            self.world_size = dist.get_world_size() if world_size is None else int(world_size)
            self._dist_ready = True
        else:
            self.rank = 0 if rank is None else int(rank)
            self.world_size = 1 if world_size is None else int(world_size)
            self._dist_ready = False

        expanded_groups = []
        for group in param_groups:
            assert "use_muon" in group, "Each param_group must include use_muon=True/False"
            params = list(group["params"])
            if not params:
                continue

            if group["use_muon"]:
                # AdaMuon defaults
                lr = group.get("lr", 0.02)
                momentum = group.get("momentum", 0.95)
                weight_decay = group.get("weight_decay", 0.01)
                nesterov = group.get("nesterov", True)
                ns_steps = group.get("ns_steps", 5)
                eps = group.get("eps", 1e-8)

                # Group by numel for fused buffers (only used if distributed)
                unique_sizes = {p.numel() for p in params}
                for size in unique_sizes:
                    p_list = [p for p in params if p.numel() == size]
                    device = p_list[0].device
                    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
                    buf = torch.empty(self.world_size, size, dtype=dtype, device=device)  # harmless if ws==1

                    expanded_groups.append(dict(
                        params=p_list,
                        use_muon=True,
                        lr=lr, weight_decay=weight_decay,
                        momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, eps=eps,
                        update_buffer=buf,
                        update_buffer_views=[buf[i] for i in range(self.world_size)],
                    ))
            else:
                # Aux AdamW defaults
                lr = group.get("lr", 3e-4)
                betas = group.get("betas", (0.9, 0.95))
                eps = group.get("eps", 1e-10)
                weight_decay = group.get("weight_decay", 0.0)

                expanded_groups.append(dict(
                    params=params,
                    use_muon=False,
                    lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                ))

        super().__init__(expanded_groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                self._step_adamuon_group(group)
            else:
                self._step_aux_adam_group(group)
        return loss

    @torch.no_grad()
    def _step_aux_adam_group(self, group: dict):
        # AdamW-style (bias-corrected)
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        wd = group["weight_decay"]

        for p in group["params"]:
            g = p.grad
            if g is None:
                continue

            if wd != 0:
                p.mul_(1 - lr * wd)

            st = self.state[p]
            if len(st) == 0:
                st["exp_avg"] = torch.zeros_like(p)
                st["exp_avg_sq"] = torch.zeros_like(p)
                st["step"] = 0
            st["step"] += 1
            t = st["step"]

            m = st["exp_avg"]; v = st["exp_avg_sq"]
            m.mul_(beta1).add_(g, alpha=1 - beta1)
            v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

            bc1 = 1 - beta1 ** t
            bc2 = 1 - beta2 ** t
            denom = (v.sqrt() / (bc2 ** 0.5)).add_(eps)
            step_dir = (m / bc1) / denom
            p.add_(step_dir, alpha=-lr)

    @torch.no_grad()
    def _step_adamuon_group(self, group: dict):
        """
        AdaMuon path:
          momentum (+ optional Nesterov) -> zeropower on sign(g) -> per-param variance buffer v
          -> normalize by sqrt(v)+eps -> heuristic scaling -> (decoupled WD) -> param update
          Dist path uses all_gather; single-process path bypasses collectives.
        """
        lr = group["lr"]; wd = group["weight_decay"]
        momentum = group["momentum"]; nesterov = group["nesterov"]
        ns_steps = group["ns_steps"]; eps = group["eps"]

        params = group["params"]

        # ---- Change 2: single-process fast path ----
        if (not self._dist_ready) or self.world_size == 1:
            for p in params:
                g = p.grad
                if g is None:
                    continue

                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(g)
                buf: Tensor = st["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g_mom = g.add(buf, alpha=momentum) if nesterov else buf

                g_flat = g_mom
                if g_flat.ndim >= 2:
                    g_flat.flatten(0,-1)


                z = zeropower_via_newtonschulz5(torch.sign(g_flat), steps=ns_steps)

                if "v_buffer" not in st:
                    st["v_buffer"] = torch.zeros_like(z)
                v = st["v_buffer"]
                v.mul_(momentum).addcmul_(1 - momentum, z, z)

                z = z / (v.sqrt().add(eps))

                scale = 0.2 * (min(p.shape) * max(p.shape)) ** 0.5 / (z.norm() + eps)
                z.mul_(scale)

                if wd != 0:
                    p.mul_(1 - lr * wd)
                p.add_(z.view_as(p), alpha=-lr)
            return

        # ---- Distributed path (unchanged semantics) ----
        update_buffer: Tensor = group["update_buffer"]
        update_buffer_views: list[Tensor] = group["update_buffer_views"]
        handle = None
        params_world = None

        def flush_prev():
            handle.wait()
            for p_world, g_world in zip(params_world, update_buffer_views):
                if wd != 0:
                    p_world.mul_(1 - lr * wd)
                p_world.add_(g_world.view_as(p_world), alpha=-lr)

        for base_i in range(0, len(params), self.world_size):
            if base_i + self.rank < len(params):
                p = params[base_i + self.rank]
                g = p.grad

                if g is None:
                    z = update_buffer_views[self.rank]
                else:
                    st = self.state[p]
                    if "momentum_buffer" not in st:
                        st["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = st["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    g_mom = g.add(buf, alpha=momentum) if nesterov else buf

                    g_flat = g_mom
                    if g_flat.ndim == 4:
                        g_flat = g_flat.view(len(g_flat), -1)
                    g_flat = g_flat.flatten()

                    z = zeropower_via_newtonschulz5(torch.sign(g_flat), steps=ns_steps)

                    if "v_buffer" not in st:
                        st["v_buffer"] = torch.zeros_like(z)
                    v = st["v_buffer"]
                    v.mul_(momentum).addcmul_(1 - momentum, z, z)
                    z = z / (v.sqrt().add(eps))

                    scale = 0.2 * (min(p.shape) * max(p.shape)) ** 0.5 / (z.norm() + eps)
                    z.mul_(scale)

                    z = z.to(update_buffer.dtype)
            else:
                z = update_buffer_views[self.rank]

            if base_i > 0:
                flush_prev()
            handle = dist.all_gather_into_tensor(update_buffer, z, async_op=True)
            params_world = params[base_i: base_i + self.world_size]

        if handle is not None:
            flush_prev()


# ==============================
# MuonWithAuxAdam (normalized Muon + aux AdamW)
# ==============================
class MuonWithAuxAdam(torch.optim.Optimizer):
    """
    Mixed optimizer:
      - Normalized Muon path for use_muon=True (EMA via lerp, optional Nesterov, zeropower on raw g)
      - AdamW-style path for use_muon=False
    """

    def __init__(self, param_groups, *, rank: int | None = None, world_size: int | None = None):
        # ---- Change 1: auto-detect distributed, default to single-GPU ----
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank() if rank is None else int(rank)
            self.world_size = dist.get_world_size() if world_size is None else int(world_size)
            self._dist_ready = True
        else:
            self.rank = 0 if rank is None else int(rank)
            self.world_size = 1 if world_size is None else int(world_size)
            self._dist_ready = False

        expanded = []
        for g in param_groups:
            assert "use_muon" in g, "Each param_group must include use_muon=True/False"
            params = list(g["params"])
            if not params:
                continue

            if g["use_muon"]:
                lr = g.get("lr", 0.02)
                weight_decay = g.get("weight_decay", 0.01)
                momentum = g.get("momentum", 0.95)
                nesterov = g.get("nesterov", True)
                ns_steps = g.get("ns_steps", 5)

                unique_sizes = {p.numel() for p in params}
                for size in unique_sizes:
                    p_list = [p for p in params if p.numel() == size]
                    device = p_list[0].device
                    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
                    buf = torch.empty(self.world_size, size, dtype=dtype, device=device)

                    expanded.append(dict(
                        params=p_list,
                        use_muon=True,
                        lr=lr, weight_decay=weight_decay,
                        momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        update_buffer=buf,
                        update_buffer_views=[buf[i] for i in range(self.world_size)],
                    ))
            else:
                lr = g.get("lr", 3e-4)
                betas = g.get("betas", (0.9, 0.95))
                eps = g.get("eps", 1e-10)
                weight_decay = g.get("weight_decay", 0.0)
                expanded.append(dict(
                    params=params,
                    use_muon=False,
                    lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                ))

        super().__init__(expanded, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_muon"]:
                self._step_muon_group(group)
            else:
                self._step_aux_adam_group(group)
        return loss

    @torch.no_grad()
    def _step_aux_adam_group(self, group: dict):
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        wd = group["weight_decay"]

        for p in group["params"]:
            g = p.grad
            if g is None:
                continue

            if wd != 0:
                p.mul_(1 - lr * wd)

            st = self.state[p]
            if len(st) == 0:
                st["exp_avg"] = torch.zeros_like(p)
                st["exp_avg_sq"] = torch.zeros_like(p)
                st["step"] = 0
            st["step"] += 1
            t = st["step"]

            m = st["exp_avg"]; v = st["exp_avg_sq"]
            m.mul_(beta1).add_(g, alpha=1 - beta1)
            v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

            bc1 = 1 - beta1**t
            bc2 = 1 - beta2**t
            step_dir = (m / bc1) / (v.sqrt() / (bc2 ** 0.5) + eps)
            p.add_(step_dir, alpha=-lr)

    @torch.no_grad()
    def _step_muon_group(self, group: dict):
        """
        Normalized Muon path (your variant):
          EMA via lerp, optional Nesterov, zeropower on raw g (not sign),
          decoupled WD, per-size fused gather only if distributed.
          Update scaling: -lr * 0.2 * sqrt(max(dim_last2))
        """
        lr = group["lr"]; wd = group["weight_decay"]
        momentum = group["momentum"]; nesterov = group["nesterov"]
        ns_steps = group["ns_steps"]
        params = group["params"]

        # ---- Change 2: single-process fast path ----
        if (not self._dist_ready) or self.world_size == 1:
            for p in params:
                g = p.grad
                if g is None:
                    continue

                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(g)
                buf: Tensor = st["momentum_buffer"]

                # EMA via lerp
                buf.lerp_(g, 1 - momentum)
                g_eff = g.lerp(buf, momentum) if nesterov else buf

                if g_eff.ndim == 4:
                    g_eff = g_eff.view(len(g_eff), -1)
                z = zeropower_via_newtonschulz5(g_eff, steps=ns_steps).flatten()

                if wd != 0:
                    p.mul_(1 - lr * wd)

                if p.ndim >= 2:
                    scale = (-lr) * 0.2 * (max(p.size(-2), p.size(-1)) ** 0.5)
                else:
                    scale = -lr * 0.2

                p.add_(z.view_as(p), alpha=scale)
            return

        # ---- Distributed path (unchanged semantics) ----
        update_buffer: Tensor = group["update_buffer"]
        update_buffer_views = group["update_buffer_views"]
        handle = None
        params_world = None

        def apply_prev():
            handle.wait()
            for p_world, g_world in zip(params_world, update_buffer_views):
                if wd != 0:
                    p_world.mul_(1 - lr * wd)
                if p_world.ndim >= 2:
                    scale = (-lr) * 0.2 * (max(p_world.size(-2), p_world.size(-1)) ** 0.5)
                else:
                    scale = -lr * 0.2
                p_world.add_(g_world.view_as(p_world), alpha=scale)

        for base_i in range(0, len(params), self.world_size):
            if base_i + self.rank < len(params):
                p = params[base_i + self.rank]
                g = p.grad
                assert g is not None, "Gradient is None for a Muon param; ensure backward() ran."

                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(g)
                buf: Tensor = st["momentum_buffer"]

                buf.lerp_(g, 1 - momentum)
                g_eff = g.lerp(buf, momentum) if nesterov else buf

                if g_eff.ndim == 4:
                    g_eff = g_eff.view(len(g_eff), -1)
                z = zeropower_via_newtonschulz5(g_eff, steps=ns_steps).flatten()
                z = z.to(update_buffer.dtype)
            else:
                z = update_buffer_views[self.rank]

            if base_i > 0:
                apply_prev()
            handle = dist.all_gather_into_tensor(update_buffer, z, async_op=True)
            params_world = params[base_i: base_i + self.world_size]

        if handle is not None:
            apply_prev()






class BGD(torch.optim.Optimizer):
    """Implements BGD.
    A simple usage of BGD would be:
    for samples, labels in batches:
        for mc_iter in range(mc_iters):
            optimizer.randomize_weights()
            output = model.forward(samples)
            loss = cirterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.aggregate_grads()
        optimizer.step()
    """
    def __init__(self, params, std_init, mean_eta=1, std_eta = 1, std_exp_factor = 1 , mc_iters=20, betas=(0.9,0.999, .9), warm_up_iters = 200,paranoia=1):
        """
        Initialization of BGD optimizer
        group["mean_param"] is the learned mean.
        group["std_param"] is the learned STD.
        :param params: List of model parameters
        :param std_init: Initialization value for STD parameter
        :param mean_eta: Eta value
        :param mc_iters: Number of Monte Carlo iteration. Used for correctness check.
                         Use None to disable the check.
        """
        super(BGD, self).__init__(params, defaults={})
        assert mc_iters is None or (type(mc_iters) == int and mc_iters > 0), "mc_iters should be positive int or None."
        self.std_init = std_init
        self.mean_eta = mean_eta / std_init**2
        self.std_eta = std_eta / std_init**2
        self.std_exp_factor = std_exp_factor
        self.mc_iters = mc_iters
        self.betas = betas
        self.fast_eps = 1e-10
        self.n_steps = 1
        self.warm_up_iters = warm_up_iters
        # Initialize mu (mean_param) and sigma (std_param)
        self.paranoia = paranoia
        for group in self.param_groups:

            assert len(group["params"]) == 1, "BGD optimizer does not support multiple params in a group"
            # group['params'][0] is the weights
            assert isinstance(group["params"][0], torch.Tensor), "BGD expect param to be a tensor"
            # We use the initialization of weights to initialize the mean.
            group["mean_param"] = group["params"][0].data.clone()
            group["std_param"] = torch.zeros_like(group["params"][0].data).add_(self.std_init)
            group["mom"] = torch.zeros_like(group["params"][0].data)
            group["std_mom"] = torch.zeros_like(group["params"][0].data)
            group["lr"] = self.mean_eta
            group["std_lr"] = self.std_eta
            group["std_reg"] = self.std_exp_factor

        self._init_accumulators()

    def get_mc_iters(self):
        return self.mc_iters

    def _init_accumulators(self):
        self.mc_iters_taken = 0
        for group in self.param_groups:
            group["eps"] = None
            group["grad_mul_eps_sum"] = torch.zeros_like(group["params"][0].data)
            group["grad_sum"] = torch.zeros_like(group["params"][0].data)

    @staticmethod
    def create_unique_param_groups(model):
        """
        Create a unique parameter group for each parameter in the given model.
        """
        param_groups = []
        for param in model.parameters():
            param_groups.append({'params': [param]})
        return param_groups

    def randomize_weights(self, force_std=-1):
        """
        Randomize the weights according to N(mean, std).
        :param force_std: If force_std>=0 then force_std is used for STD instead of the learned STD.
        :return: None
        """
        std_mean = 0
        for group in self.param_groups:
            mean = group["mean_param"]
            std = group["std_param"]
            std_mean += std.mean()
            if force_std >= 0:
                std = std.mul(0).add(force_std)
            group["eps"] = torch.normal(torch.zeros_like(mean), self.paranoia)
            # Reparameterization trick (here we set the weights to their randomized value):
            group["params"][0].data.copy_(mean.add(std.mul(group["eps"])))
        #print(std_mean/len(self.param_groups))

    def aggregate_grads(self, batch_size):
        """
        Aggregates a single Monte Carlo iteration gradients. Used in step() for the expectations calculations.
        optimizer.zero_grad() should be used before calling .backward() once again.
        :param batch_size: BGD is using non-normalized gradients, but PyTorch gives normalized gradients.
                            Therefore, we multiply the gradients by the batch size.
        :return: None
        """
        self.mc_iters_taken += 1
        groups_cnt = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for group in self.param_groups:
            for each in group.keys():
                if isinstance(group[each], torch.Tensor):
                    group[each] = group[each].to(device)
            if group["params"][0].grad is None:
                continue
            assert group["eps"] is not None, "Must randomize weights before using aggregate_grads"
            groups_cnt += 1
            grad = group["params"][0].grad.data.mul(batch_size)
            # group["grad_sum"] = group["grad_sum"].detach().cpu()
            # grad = grad.detach().cpu()
            group["grad_sum"].add_(grad)
            group["grad_mul_eps_sum"].add_(grad.mul(group["eps"]))
            group["eps"] = None

        assert groups_cnt > 0, "Called aggregate_grads, but all gradients were None. Make sure you called .backward()"

    def get_stats(self):
        avg_mean_sq = 0
        avg_var = 0
        param_count = 0
        avg_grad_sq = 0
        avg_grad_epsilon = 0
        with torch.no_grad():
            for group in self.param_groups:
                mean = group["mean_param"]
                std = group["std_param"]
                e_grad = group["grad_sum"].div(self.mc_iters_taken)
                e_grad_epsilon = group["grad_mul_eps_sum"].div(self.mc_iters_taken)

                avg_mean_sq += torch.sum(mean**2)
                avg_var += torch.sum(std**2)
                param_count += torch.numel(mean)
                avg_grad_sq += torch.sum(e_grad**2)
                avg_grad_epsilon += torch.sum(e_grad_epsilon)

        return float(avg_mean_sq / param_count) , float(avg_var / param_count), float(avg_grad_sq / param_count), float(avg_grad_epsilon / param_count)

    def get_std_distribution(self, bucket_range = None, num_buckets = 1000):
        if bucket_range == None:
            bucket_range = self.std_init * 2
        with torch.no_grad():
            running_hist = torch.zeros(num_buckets)
            param_count = 0
            for group in self.param_groups:
                std = group["std_param"]

                if std.data.device != running_hist.device:
                    running_hist = running_hist.to(std.data.device)


                new_hist = torch.histc(std,num_buckets, min = 0, max =bucket_range)
                if std.data.device != new_hist.device:
                    running_hist.to(std.data.device)

                running_hist = running_hist + new_hist
                param_count += torch.numel(std)

            return running_hist / param_count

    def step(self, closure=None):
        """
        Updates the learned mean and STD.
        :return:
        """
        # Makes sure that self.mc_iters had been taken.
        assert self.mc_iters is None or self.mc_iters == self.mc_iters_taken, "MC iters is set to " \
                                                                              + str(self.mc_iters) \
                                                                              + ", but took " + \
                                                                              str(self.mc_iters_taken) + " MC iters"
        max_grads = []
        for group in self.param_groups:
            mean = group["mean_param"]
            std = group["std_param"]
            mom = group["mom"]
            mean_eta = group["lr"]
            std_eta = group["std_lr"]
            std_exp_factor = group["std_reg"]
            std_mom = group["std_mom"]


            if self.n_steps < self.warm_up_iters:
                mean_eta = mean_eta * math.sin( self.n_steps * (math.pi / self.warm_up_iters))**2
            if self.n_steps < 2 * self.warm_up_iters:
                std_eta = std_eta * math.sin( self.n_steps * (math.pi / (self.warm_up_iters*2)))**2


            # Divide gradients by MC iters to get expectation
            e_grad = group["grad_sum"].div(self.mc_iters_taken)
            e_grad_eps = group["grad_mul_eps_sum"].div(self.mc_iters_taken)
            max_grads.append(e_grad.max().item())



            alpha = (math.sqrt(1-self.betas[1]**(self.n_steps))/(1-self.betas[0]**(self.n_steps)))*(mean_eta*std.pow(2))

            mom.copy_(self.betas[0]*mom + (1-self.betas[0])*(e_grad))
            mean.add_(-mom*mean_eta*std.pow(2))
            if torch.sum(torch.isnan(mean)):
                raise ValueError("Badam optimizer has caused nan mean value.")

            # tmp_mean = mean + (-alpha*std.pow(2)*(
            #     mom / (torch.sqrt(mom_var)+self.fast_eps)
            # )
            # )

            std_mom.copy_(self.betas[2]*std_mom + (1-self.betas[2])*(std_eta*e_grad_eps))
            std_val = std_mom

            sqrt_term = torch.sqrt(std_val.mul(std).div(2).pow(2).add(std_exp_factor)).mul(std)
            final_std_val = sqrt_term.add(-std_val.mul(std.pow(2)).div(2))
            std.copy_(torch.clamp(final_std_val,min=.0001*self.std_init, max = 2*self.std_init))

        self.n_steps+=1
        self.randomize_weights(force_std=0)
        self._init_accumulators()


# =======================================================
#  NorMuon row-normalized update (2D)
# =======================================================
def normuon_update(
    grad_2d: Tensor,
    momentum: Tensor,
    v_buffer: Tensor,
    beta: float = 0.95,
    beta2: float = 0.95,
    ns_steps: int = 5,
    eps: float = 1e-10,
) -> Tuple[Tensor, Tensor]:
    """
    NorMuon-style update on a *2D view* of a weight matrix.

    Shapes:
      grad_2d      : (m, n_flat)
      momentum     : (m, n_flat)  -- EMA of gradients
      v_buffer     : (m, 1)       -- row-wise second moment

    Returns:
      O_hat : (m, n_flat)  -- row-normalized orthogonal update
      O     : (m, n_flat)  -- pre-normalization orthogonal matrix
    """
    # 1) First-order momentum
    momentum.lerp_(grad_2d, 1.0 - beta)
    M = momentum  # effective direction

    # 2) Orthogonalize via Muon NS iteration
    O = zeropower_via_newtonschulz5(M, steps=ns_steps).to(dtype=grad_2d.dtype)

    # 3) Row-wise second moment: v_t in R^{m x 1}
    row_mean_sq = (O * O).mean(dim=1, keepdim=True)  # (m, 1)
    v_buffer.lerp_(row_mean_sq, 1.0 - beta2)

    # 4) Row-wise normalization
    O_hat = O / (v_buffer.sqrt() + eps)  # (m, n_flat)

    return O_hat, O


# =======================================================
#  Aux Adam update
# =======================================================
def adam_update(
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    step: int,
    betas: Tuple[float, float],
    eps: float,
) -> Tensor:
    """
    Single-step Adam update with bias correction.
    """
    beta1, beta2 = betas
    exp_avg.lerp_(grad, 1.0 - beta1)
    exp_avg_sq.lerp_(grad * grad, 1.0 - beta2)

    # Bias correction
    bias_c1 = 1.0 - beta1 ** step
    bias_c2 = 1.0 - beta2 ** step

    denom = (exp_avg_sq / bias_c2).sqrt() + eps
    return (exp_avg / bias_c1) / denom


# =======================================================
#  Single-device NorMuon + aux Adam optimizer
# =======================================================
class SingleDeviceNorMuonWithAuxAdam(Optimizer):
    """
    Non-distributed NorMuon + aux Adam optimizer.

    Usage pattern:
        hidden_params = []
        aux_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim >= 2 and not name.endswith(".bias"):
                hidden_params.append(p)   # NorMuon
            else:
                aux_params.append(p)      # Adam

        optimizer = SingleDeviceNorMuonWithAuxAdam([
            dict(params=hidden_params, use_muon=True,  lr=0.02,
                 momentum=0.95, beta2=0.95, weight_decay=0.0005),
            dict(params=aux_params,    use_muon=False, lr=3e-4,
                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0),
        ])
    """

    def __init__(self, param_groups: List[Dict[str, Any]]):
        processed_groups: List[Dict[str, Any]] = []

        for group in param_groups:
            if "use_muon" not in group:
                raise ValueError("Each param_group must include 'use_muon': True/False")

            params = list(group["params"])
            if len(params) == 0:
                continue

            if group["use_muon"]:
                lr = group.get("lr", 0.02)
                momentum = group.get("momentum", 0.95)
                beta2 = group.get("beta2", 0.95)
                weight_decay = group.get("weight_decay", 0.0)

                new_group = dict(
                    params=params,
                    use_muon=True,
                    lr=lr,
                    momentum=momentum,
                    beta2=beta2,
                    weight_decay=weight_decay,
                )
                processed_groups.append(new_group)
            else:
                lr = group.get("lr", 3e-4)
                betas = group.get("betas", (0.9, 0.95))
                eps = group.get("eps", 1e-10)
                weight_decay = group.get("weight_decay", 0.0)

                new_group = dict(
                    params=params,
                    use_muon=False,
                    lr=lr,
                    betas=betas,
                    eps=eps,
                    weight_decay=weight_decay,
                )
                processed_groups.append(new_group)

        super().__init__(processed_groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                self._step_nor_muon_group(group)
            else:
                self._step_aux_adam_group(group)

        return loss

    @torch.no_grad()
    def _step_nor_muon_group(self, group: Dict[str, Any]) -> None:
        lr: float = group["lr"]
        beta: float = group["momentum"]
        beta2: float = group["beta2"]
        wd: float = group["weight_decay"]

        for p in group["params"]:
            g = p.grad
            if g is None:
                continue

            # Use 2D view for linear/conv weights
            if p.ndim == 2:
                m, n_flat = p.shape
                grad_2d = g
                W_2d = p
            elif p.ndim == 4:
                # Conv weights: (out_channels, in_channels, kH, kW)
                m = p.size(0)
                n_flat = p[0].numel()
                grad_2d = g.view(m, n_flat)
                W_2d = p.view(m, n_flat)
            else:
                # 1D / others: skip here; should be in aux Adam group
                continue

            st = self.state[p]
            if len(st) == 0:
                st["momentum_buffer"] = torch.zeros_like(grad_2d)
                st["v_buffer"] = torch.zeros(
                    m, 1, device=p.device, dtype=p.dtype
                )

            mom: Tensor = st["momentum_buffer"]
            v_buf: Tensor = st["v_buffer"]

            O_hat, _ = normuon_update(
                grad_2d,
                mom,
                v_buf,
                beta=beta,
                beta2=beta2,
                ns_steps=5,
                eps=1e-10,
            )

            # Global RMS-matching scale
            fro_norm = O_hat.norm() + 1e-10
            eta_hat = 0.2 * lr * math.sqrt(m * n_flat) / fro_norm

            # Decoupled weight decay
            if wd != 0.0:
                W_2d.mul_(1.0 - lr * wd)

            # Apply update
            W_2d.add_(O_hat, alpha=-eta_hat)

            # Write back if conv
            if p.ndim == 4:
                p.copy_(W_2d.view_as(p))

    @torch.no_grad()
    def _step_aux_adam_group(self, group: Dict[str, Any]) -> None:
        lr: float = group["lr"]
        betas: Tuple[float, float] = group["betas"]
        eps: float = group["eps"]
        wd: float = group["weight_decay"]

        for p in group["params"]:
            g = p.grad
            if g is None:
                continue

            st = self.state[p]
            if len(st) == 0:
                st["exp_avg"] = torch.zeros_like(p)
                st["exp_avg_sq"] = torch.zeros_like(p)
                st["step"] = 0

            st["step"] += 1
            step_num = st["step"]
            exp_avg: Tensor = st["exp_avg"]
            exp_avg_sq: Tensor = st["exp_avg_sq"]

            upd = adam_update(g, exp_avg, exp_avg_sq, step_num, betas, eps)

            if wd != 0.0:
                p.mul_(1.0 - lr * wd)

            p.add_(upd, alpha=-lr)

