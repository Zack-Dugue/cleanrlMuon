import torch
import torch.distributed as dist
from torch import Tensor

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
