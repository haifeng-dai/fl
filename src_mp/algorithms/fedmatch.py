import copy
import math

import torch
from torch import nn
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, TensorDataset

from src.algorithms.utils.input import prepare_input_batch

from .core import (
    BaseClientExecutor,
    BaseServer,
    ClientResult,
    ce_loss,
    clone_state,
    kl_loss,
)
from .core.augment import strong_augment
from .core.model import build_model


def init_psi_state(
    model_state: dict[str, torch.Tensor],
    parameter_names: tuple[str, ...],
    psi_factor: float,
) -> dict[str, torch.Tensor]:
    """初始化仅包含参数的 psi 状态字典。"""
    psi_state: dict[str, torch.Tensor] = {}
    for name in parameter_names:
        if name not in model_state:
            raise KeyError(f"Parameter '{name}' not found in model_state")
        param = model_state[name]
        if not param.is_floating_point():
            raise TypeError(f"Parameter '{name}' must be floating point tensor")
        psi_state[name] = (param * psi_factor).cpu().detach().clone()
    return psi_state


def merge_state(
    sigma_state: dict[str, torch.Tensor],
    psi_state: dict[str, torch.Tensor],
    l1_thres: float = 0.0,
    sparsify: bool = False,
) -> dict[str, torch.Tensor]:
    """逐元素合并 σ + ψ 为完整模型状态（输入为 CPU 字典，不改变输入）。"""
    merged: dict[str, torch.Tensor] = {}
    for k, s_val in sigma_state.items():
        if k in psi_state:
            p_val = psi_state[k]
            if sparsify:
                p_val = p_val * (p_val.abs() > l1_thres).to(p_val.dtype)
            merged[k] = s_val + p_val
        else:
            merged[k] = s_val.clone()
    return merged


def s2c_merge_state(
    server_state: dict[str, torch.Tensor],
    client_state: dict[str, torch.Tensor],
    delta_thres: float,
    sparsify_thres: float | None = None,
) -> dict[str, torch.Tensor]:
    """服务端到客户端 (S2C) 稀疏差分合并。"""
    if set(server_state.keys()) != set(client_state.keys()):
        raise ValueError(
            f"Key mismatch between server and client states: "
            f"{set(server_state.keys()) ^ set(client_state.keys())}"
        )

    res: dict[str, torch.Tensor] = {}
    for k, s_val in server_state.items():
        c_val = client_state[k]
        s_eval = s_val
        c_eval = c_val
        if sparsify_thres is not None:
            s_eval = s_eval * (s_eval.abs() > sparsify_thres).to(s_eval.dtype)
            c_eval = c_eval * (c_eval.abs() > sparsify_thres).to(c_eval.dtype)
        diff = s_eval - c_eval
        mask = diff.abs() > delta_thres
        res[k] = torch.where(mask, s_val, c_val).clone()
    return res


def c2s_merge_state(
    client_state: dict[str, torch.Tensor],
    server_state: dict[str, torch.Tensor],
    delta_thres: float,
    sparsify_thres: float | None = None,
) -> dict[str, torch.Tensor]:
    """客户端到服务端 (C2S) 稀疏差分合并。"""
    if set(client_state.keys()) != set(server_state.keys()):
        raise ValueError(
            f"Key mismatch between client and server states: "
            f"{set(client_state.keys()) ^ set(server_state.keys())}"
        )

    res: dict[str, torch.Tensor] = {}
    for k, c_val in client_state.items():
        s_val = server_state[k]
        c_eval = c_val
        if sparsify_thres is not None:
            c_eval = c_eval * (c_eval.abs() > sparsify_thres).to(c_eval.dtype)
        diff = c_eval - s_val
        mask = diff.abs() > delta_thres
        res[k] = torch.where(mask, c_val, s_val).clone()
    return res


def aggregate_sigma_states(
    states: list[dict[str, torch.Tensor]],
    weights: list[float],
) -> dict[str, torch.Tensor]:
    """聚合完整 sigma 状态（浮点参数及 buffer 加权平均，非浮点 buffer 取首个状态值）。"""
    if not states:
        raise ValueError("states list is empty")
    if len(states) != len(weights):
        raise ValueError(
            f"Length mismatch: len(states)={len(states)}, len(weights)={len(weights)}"
        )

    total_w = sum(weights)
    norm_w = [w / total_w for w in weights]
    keys = list(states[0].keys())
    res: dict[str, torch.Tensor] = {}

    for k in keys:
        ref_val = states[0][k]
        if ref_val.is_floating_point():
            stacked = torch.stack([s[k] for s in states], dim=0)
            w_tensor = torch.tensor(norm_w, dtype=ref_val.dtype, device=ref_val.device)
            while w_tensor.ndim < stacked.ndim:
                w_tensor = w_tensor.unsqueeze(-1)
            res[k] = (stacked * w_tensor).sum(dim=0).cpu().detach().clone()
        else:
            res[k] = ref_val.cpu().detach().clone()
    return res


class DecomposedModel(nn.Module):
    """θ = σ + ψ 分解模型。

    σ 为包含所有参数及 buffer 的模型；ψ 为同构副本（仅使用参数）。
    θ 为前向实体，其每个参数在同步时重建为 σ + ψ，同时 θ 自行维护 buffer 的运行状态。
    """

    def __init__(self, model: nn.Module, l1_thres: float):
        super().__init__()
        self.theta = model
        self.theta_buffer_names = tuple(name for name, _ in model.named_buffers())
        self.sigma = copy.deepcopy(model)
        self.psi = copy.deepcopy(model)
        self.l1_thres = l1_thres

    def sync_theta(self, mode: str = "both"):
        """将 θ 的每个参数重建为 σ + ψ。

        mode='sigma': 仅 σ 保持可导（有监督阶段固定 ψ）
        mode='psi': 仅 ψ 保持可导（无监督阶段固定 σ）
        mode='both': σ 与 ψ 均可导
        """
        for name, sp in self.sigma.named_parameters():
            pp = self.psi.get_parameter(name)
            if mode == "sigma":
                merged = sp + pp.detach()
            elif mode == "psi":
                merged = sp.detach() + pp
            else:
                merged = sp + pp
            target = self.theta
            *mod_path, attr = name.split(".")
            for part in mod_path:
                target = getattr(target, part)
            target.__dict__["_parameters"].pop(attr, None)
            target.__dict__["_buffers"][attr] = merged

    def load_sigma_psi(
        self,
        sigma_state: dict[str, torch.Tensor],
        psi_state: dict[str, torch.Tensor],
    ):
        """加载下发的 σ 与 ψ 状态并初始化 θ。"""
        self.sigma.load_state_dict(sigma_state)
        psi_param_names = {n for n, _ in self.psi.named_parameters()}
        if set(psi_state.keys()) != psi_param_names:
            raise KeyError(
                f"psi_state keys mismatch: {set(psi_state.keys()) ^ psi_param_names}"
            )
        with torch.no_grad():
            for name, param in self.psi.named_parameters():
                param.copy_(psi_state[name])
        merged_init = merge_state(sigma_state, psi_state, self.l1_thres, sparsify=False)
        self.theta.load_state_dict(merged_init, strict=True)
        self.sync_theta()

    def sigma_state_dict(self) -> dict[str, torch.Tensor]:
        """导出训练后的 sigma 参数，以及由 theta 真正运行产生的 buffer。"""
        s_state = self.sigma.state_dict()
        theta_buffers = dict(self.theta.named_buffers())
        for name in self.theta_buffer_names:
            if name not in theta_buffers:
                raise KeyError(f"Real buffer '{name}' missing in theta named_buffers")
            s_state[name] = theta_buffers[name]
        return clone_state(s_state)

    def psi_state_dict(self) -> dict[str, torch.Tensor]:
        """导出仅包含 parameters 的 psi 状态字典。"""
        return {k: v.cpu().detach().clone() for k, v in self.psi.named_parameters()}


def compute_iccs_loss(
    dm: DecomposedModel,
    x_ub_raw: torch.Tensor,
    y_ub_raw: torch.Tensor,
    y_logits: torch.Tensor,
    helper_net: nn.Module | None,
    helper_states: list[dict[str, torch.Tensor]],
    conf: float,
    lambda_i: float,
    lambda_a: float,
    curr_round: int,
    num_class: int,
    dataset_name: str,
) -> tuple[torch.Tensor, int, int]:
    """计算基于置信度掩码的 inter-client consistency KL 损失与 agreement 伪标签 CE 损失。"""
    y_probs = torch.softmax(y_logits.detach(), dim=1)
    conf_mask = y_probs.max(dim=1).values >= conf

    if not conf_mask.any():
        return torch.tensor(0.0, device=x_ub_raw.device), 0, 0

    x_conf_raw = x_ub_raw[conf_mask]
    local_conf_logits = y_logits[conf_mask]  # 承接梯度，不 detach
    y_conf_logits_det = local_conf_logits.detach()

    with torch.no_grad():
        helper_conf_logits = []
        if helper_net is not None:
            x_conf_norm = prepare_input_batch(x_conf_raw, dataset_name)
            for hs in helper_states:
                helper_net.load_state_dict(hs)
                helper_conf_logits.append(helper_net(x_conf_norm))

    loss_iccs = torch.tensor(0.0, device=x_ub_raw.device)

    # 1. Inter-client consistency: KL(helper || local) * lambda_i
    if helper_conf_logits and curr_round > 0:
        phi_kl = sum(
            kl_loss(local_conf_logits, h_logits) for h_logits in helper_conf_logits
        ) / len(helper_conf_logits)
        loss_iccs = loss_iccs + lambda_i * phi_kl

    # 2. Agreement-based pseudo labeling: CE * lambda_a
    with torch.no_grad():
        votes = one_hot(y_conf_logits_det.argmax(dim=1), num_class)
        if helper_conf_logits and curr_round > 0:
            for h_logits in helper_conf_logits:
                votes += one_hot(h_logits.argmax(dim=1), num_class)
        y_pseudo = votes.argmax(dim=1)

    y_hard_logits = dm.theta(strong_augment(x_conf_raw, dataset_name))
    loss_ce = lambda_a * ce_loss(y_hard_logits, y_pseudo)
    loss_iccs = loss_iccs + loss_ce

    pseudo_cnt = int(conf_mask.sum().item())
    y_ub_device = y_ub_raw.to(y_pseudo.device)
    pseudo_corr = int((y_pseudo == y_ub_device[conf_mask]).sum().item())
    return loss_iccs, pseudo_cnt, pseudo_corr


class Client(BaseClientExecutor):
    """FedMatch 客户端，分别优化 sigma 和 psi 两个参数分支。"""

    def __init__(self, args, device, num_class):
        super().__init__(args, device, num_class)
        self.conf = args.conf
        self.lambda_s = args.lambda_s
        self.lambda_i = args.lambda_i
        self.lambda_a = args.lambda_a
        self.lambda_l2 = args.lambda_l2
        self.lambda_l1 = args.lambda_l1
        self.l1_thres = args.l1_thres
        self.delta_thres = args.delta_thres
        self.curr_round = 0

    def train(self):
        payload = self.payload
        self.curr_round = payload.get("curr_round", 0)
        # 每个任务从干净的模型实例开始，避免复用模型中被重参数化的
        # non-leaf Tensor 触发 deepcopy 限制。
        base_model = build_model(self.model_param).to(self.device)
        base_model.load_state_dict(self.model.state_dict())
        dm = DecomposedModel(base_model, self.l1_thres)
        dm.load_sigma_psi(payload["sigma_state"], payload["psi_state"])
        optimizer_s = torch.optim.SGD(
            dm.sigma.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        optimizer_u = torch.optim.SGD(
            dm.psi.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        train_set = self.current_task.train_set
        labeled = train_set.is_labeled.bool()
        x_l, y_l = train_set.x[labeled], train_set.y[labeled]
        x_u, y_u = train_set.x[~labeled], train_set.y[~labeled]
        l_loader = DataLoader(TensorDataset(x_l, y_l), self.batch_size, shuffle=True)
        steps = max(1, math.ceil(len(x_l) / self.batch_size))
        u_batch = max(1, math.ceil(len(x_u) / steps))
        u_loader = DataLoader(TensorDataset(x_u, y_u), u_batch, shuffle=True)
        helper_states = []
        for state in payload.get("helper_psi_states") or []:
            helper_states.append({k: v for k, v in state.items()})
        helper_net = (
            build_model(self.model_param).to(self.device).eval()
            if helper_states
            else None
        )
        helper_net = (
            build_model(self.model_param).to(self.device).eval()
            if helper_states
            else None
        )
        self._dm = dm
        self._optimizer_s = optimizer_s
        self._optimizer_u = optimizer_u
        self._l_loader = l_loader
        self._u_loader = u_loader
        self._helper_net = helper_net
        self._helper_states = helper_states
        total = batches = 0
        self._pseudo_count = self._pseudo_correct = 0
        for _ in range(self.epochs):
            total, batches = self.run_epoch(total, batches)
        pseudo_count, pseudo_correct = self._pseudo_count, self._pseudo_correct
        sigma = dm.sigma_state_dict()
        psi = dm.psi_state_dict()
        return ClientResult(
            self.client_id,
            total / max(1, batches),
            sigma,
            {
                "sigma": sigma,
                "psi": psi,
                "pseudo_count": pseudo_count,
                "pseudo_correct": pseudo_correct,
            },
        )

    def run_epoch(self, total: float, batches: int) -> tuple[float, int]:
        """执行一个 FedMatch 本地 epoch，并更新 sigma/psi 两个分支。"""
        for (x_lb, y_lb), (x_ub, y_ub) in zip(self._l_loader, self._u_loader):
            x_lb, y_lb = x_lb.to(self.device), y_lb.to(self.device)
            x_ub, y_ub = x_ub.to(self.device), y_ub.to(self.device)
            self._dm.sync_theta("sigma")
            self._optimizer_s.zero_grad()
            x_lb_w = prepare_input_batch(x_lb, self.dataset)
            logits_l = self._dm.theta(x_lb_w)
            loss_s = self.lambda_s * ce_loss(logits_l, y_lb)
            self.check_nan(loss_s)
            loss_s.backward()
            self._optimizer_s.step()
            self._dm.sync_theta("psi")
            self._optimizer_u.zero_grad()
            x_uw = prepare_input_batch(x_ub, self.dataset)
            reg_l1 = sum(p.abs().sum() for p in self._dm.psi.parameters())
            reg_l2 = sum(
                (s.detach() - p).square().sum()
                for s, p in zip(self._dm.sigma.parameters(), self._dm.psi.parameters())
            )
            loss_u = self.lambda_l1 * reg_l1 + self.lambda_l2 * reg_l2
            self.check_nan(loss_u)
            iccs, count, correct = compute_iccs_loss(
                self._dm,
                x_ub,
                y_ub,
                self._dm.theta(x_uw),
                self._helper_net,
                self._helper_states,
                self.conf,
                self.lambda_i,
                self.lambda_a,
                self.curr_round,
                self.num_class,
                self.dataset,
            )
            loss_u_total = loss_u + iccs
            self.check_nan(loss_u_total)
            loss_u_total.backward()
            self._optimizer_u.step()
            total += (loss_s.item() + loss_u_total.item()) / 2
            batches += 1
            self._pseudo_count += count
            self._pseudo_correct += correct
        return total, batches


class Server(BaseServer):
    """FedMatch Server，维护全局及每客户端的 sigma/psi 状态。"""

    client_cls = Client
    supports_ssl = True

    def __init__(self, args, devices):
        super().__init__(args, devices)
        self.conf = args.conf
        self.h_interval = args.h_interval
        self.num_helpers = args.num_helpers
        self.l1_thres = args.l1_thres
        self.delta_thres = args.delta_thres
        self.psi_factor = args.psi_factor
        self.parameter_names = tuple(name for name, _ in self.model.named_parameters())
        self.sigma_state = clone_state(self.model.state_dict())
        self.psi_state = init_psi_state(
            self.sigma_state, self.parameter_names, self.psi_factor
        )
        self.client_sigma_states = {
            cid: clone_state(self.sigma_state) for cid in range(self.num_clients)
        }
        self.client_psi_states = {
            cid: clone_state(self.psi_state) for cid in range(self.num_clients)
        }
        self.seen_clients = set()
        self.current_round = 0
        self.sigmas, self.psis, self.embeddings = {}, {}, {}
        self.pseudo_acc, self.pseudo_count = [], []
        self.embedding_noise = self._build_embedding_noise()

    def _build_embedding_noise(self):
        sample = next((s.x[0] for s in self.train_sets.values() if len(s.x)), None)
        if sample is None:
            raise ValueError("所有客户端数据均为空，无法构造 embedding noise")
        return torch.rand(
            (1, *sample.shape), generator=torch.Generator().manual_seed(42)
        )

    def _embed(self, cid):
        self.model.load_state_dict(merge_state(self.sigmas[cid], self.psis[cid]))
        self.model.eval()
        with torch.no_grad():
            return self.model(self.embedding_noise).flatten().cpu().clone()

    def _helpers(self, cid):
        if cid not in self.embeddings or len(self.embeddings) < 2:
            return None
        ids = [i for i in self.embeddings if i != cid]
        target = self.embeddings[cid].unsqueeze(0)
        distances = torch.cdist(
            torch.stack([self.embeddings[i] for i in ids]), target
        ).flatten()
        return [
            merge_state(self.sigma_state, self.psis[ids[i]])
            for i in torch.argsort(distances)[: self.num_helpers]
        ]

    def train_payloads(self):
        server_sigma = clone_state(self.sigma_state)
        server_psi = clone_state(self.psi_state)
        names = set(self.parameter_names)
        sigma_params = {k: v for k, v in server_sigma.items() if k in names}
        payloads = {}
        for cid in self.selected:
            if cid not in self.seen_clients:
                sigma, psi = server_sigma, server_psi
            else:
                old_sigma = self.client_sigma_states[cid]
                sigma = {
                    **s2c_merge_state(
                        sigma_params,
                        {k: v for k, v in old_sigma.items() if k in names},
                        self.delta_thres,
                    ),
                    **{k: v for k, v in server_sigma.items() if k not in names},
                }
                psi = s2c_merge_state(
                    server_psi,
                    self.client_psi_states[cid],
                    self.delta_thres,
                    self.l1_thres,
                )
            payloads[cid] = {
                "sigma_state": sigma,
                "psi_state": psi,
                "helper_psi_states": self._helpers(cid)
                if ((self.current_round + 1) % self.h_interval == 0)
                else None,
                "curr_round": self.current_round,
            }
        return payloads

    def apply_result(self, results):
        names = set(self.parameter_names)
        sigmas, psis = [], []
        for cid in self.selected:
            sigma = results[cid].state
            psi = results[cid].payload["psi"]
            self.client_sigma_states[cid] = clone_state(sigma)
            self.client_psi_states[cid] = clone_state(psi)
            self.seen_clients.add(cid)
            self.sigmas[cid], self.psis[cid] = (
                clone_state(sigma),
                clone_state(psi),
            )
            self.embeddings[cid] = self._embed(cid)
            sigmas.append(sigma)
            psis.append(psi)
        weights = [1.0 / len(sigmas)] * len(sigmas)
        self.sigma_state = aggregate_sigma_states(sigmas, weights)
        self.psi_state = {
            k: v for k, v in aggregate_sigma_states(psis, weights).items() if k in names
        }
        self.model.load_state_dict(
            merge_state(self.sigma_state, self.psi_state, self.l1_thres, True)
        )

    def record_round(self, loss, accuracy):
        super().record_round(loss, accuracy)
        self.pseudo_count.append(
            sum(r.payload["pseudo_count"] for r in self._last_results.values())
        )
        count = self.pseudo_count[-1]
        correct = sum(r.payload["pseudo_correct"] for r in self._last_results.values())
        self.pseudo_acc.append(100.0 * correct / max(1, count))

    def run_round(self):
        self._last_results = self.train_clients()
        self.apply_result(self._last_results)
        loss = sum(r.loss for r in self._last_results.values()) / len(self.selected)
        self.record_round(loss, self.evaluate())
        self.current_round += 1

    def progress_metrics(self):
        metrics = super().progress_metrics()
        metrics.update(
            pseudo_acc=f"{self.pseudo_acc[-1]:.2f}%", pseudo_count=self.pseudo_count[-1]
        )
        return metrics

    def round_log_metrics(self):
        metrics = super().round_log_metrics()
        metrics.update(
            pseudo_acc=f"{self.pseudo_acc[-1]:.2f}%", pseudo_count=self.pseudo_count[-1]
        )
        return metrics
