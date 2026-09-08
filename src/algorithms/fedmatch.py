import copy
import math
import os
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, TensorDataset

from .utils import (
    BaseParams,
    BaseServer,
    clone_cpu_state,
    fmt_num,
    get_model,
    param_aggregate,
    prepare_input_batch,
)
from .utils.augment import strong_augment
from .utils.input import DATASET_SPECS
from .utils.loss import kl_loss


def get_path(args):
    """构造实验日志文件名（含域配置与算法超参值）。"""
    args.file_name = (
        f"{args.common_name}_{fmt_num(args.confidence)}"
        f"_{fmt_num(args.h_interval)}_{fmt_num(args.num_helpers)}"
        f"_{fmt_num(args.lambda_s)}_{fmt_num(args.lambda_i)}"
        f"_{fmt_num(args.lambda_a)}_{fmt_num(args.lambda_l2)}"
        f"_{fmt_num(args.lambda_l1)}_{fmt_num(args.l1_thres)}"
        f"_{fmt_num(args.delta_thres)}_{fmt_num(args.psi_factor)}"
    )
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


@dataclass
class Params(BaseParams):
    sigma_state: dict[str, torch.Tensor]
    psi_state: dict[str, torch.Tensor]
    server_sigma_state: dict[str, torch.Tensor]
    server_psi_state: dict[str, torch.Tensor]
    curr_round: int
    helper_psi_states: list[dict[str, torch.Tensor]] | None
    confidence: float
    lambda_s: float
    lambda_i: float
    lambda_a: float
    lambda_l2: float
    lambda_l1: float
    l1_thres: float
    delta_thres: float


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

    def sync_theta(self):
        """将 θ 的每个参数重建为 σ + ψ（保留计算图，梯度可回传至 σ/ψ）。"""
        for name, sp in self.sigma.named_parameters():
            pp = self.psi.get_parameter(name)
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
        return clone_cpu_state(s_state)

    def psi_state_dict(self) -> dict[str, torch.Tensor]:
        """导出仅包含 parameters 的 psi 状态字典。"""
        return {k: v.cpu().detach().clone() for k, v in self.psi.named_parameters()}


def compute_iccs_loss(
    dm: DecomposedModel,
    x_ub_raw: torch.Tensor,
    y_logits: torch.Tensor,
    helper_net: nn.Module | None,
    helper_states: list[dict[str, torch.Tensor]],
    confidence: float,
    lambda_i: float,
    lambda_a: float,
    curr_round: int,
    num_class: int,
    dataset_name: str,
) -> torch.Tensor:
    """计算基于置信度掩码的 inter-client consistency KL 损失与 agreement 伪标签 CE 损失。"""
    y_probs = torch.softmax(y_logits.detach(), dim=1)
    conf_mask = y_probs.max(dim=1).values >= confidence

    if not conf_mask.any():
        return torch.tensor(0.0, device=x_ub_raw.device)

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
    loss_ce = lambda_a * F.cross_entropy(y_hard_logits, y_pseudo)
    loss_iccs = loss_iccs + loss_ce

    return loss_iccs


def train(p: Params):
    """Ray Worker：单客户端本地训练。"""
    device = torch.device(p.client_gpu)

    # 1. 重建分解模型：θ 前向实体 + σ/ψ 可训练副本
    model = get_model(p.model_name, p.dataset, p.num_class, p.feature_dim).to(device)
    dm = DecomposedModel(model, p.l1_thres)
    dm.load_sigma_psi(p.sigma_state, p.psi_state)

    # 2. 两个独立优化器实现 disjoint learning：σ ← 监督损失，ψ ← 无监督损失
    optimizer_s = torch.optim.SGD(
        dm.sigma.parameters(),
        lr=p.lr,
        momentum=p.momentum,
        weight_decay=p.weight_decay,
    )
    optimizer_u = torch.optim.SGD(
        dm.psi.parameters(),
        lr=p.lr,
        momentum=p.momentum,
        weight_decay=p.weight_decay,
    )

    # 3. helper 模型权重准备：使用全局基准 server_sigma_state 合并 helper_psi
    helper_states = []
    if p.helper_psi_states is not None:
        for hps in p.helper_psi_states:
            merged = merge_state(p.server_sigma_state, hps, p.l1_thres, sparsify=False)
            helper_states.append(merged)

    # 准备一个单例的 GPU 模型供 helper 推理复用
    helper_net = None
    if helper_states:
        helper_net = get_model(p.model_name, p.dataset, p.num_class, p.feature_dim).to(
            device
        )
        helper_net.eval()

    # 4. 按 is_labeled 掩码切出有标签/无标签数据
    x_l = p.train_set.x[p.train_set.is_labeled]
    y_l = p.train_set.y[p.train_set.is_labeled]
    x_u = p.train_set.x[~p.train_set.is_labeled]

    # 5. 无标签批大小按步数反推
    num_steps = round(len(x_l) / p.batch_size)
    bsize_u = math.ceil(len(x_u) / max(1, num_steps))

    l_loader = DataLoader(
        TensorDataset(x_l, y_l), batch_size=p.batch_size, shuffle=True
    )
    u_loader = DataLoader(TensorDataset(x_u), batch_size=bsize_u, shuffle=True)

    # 6. 本地训练：每步先监督（仅 σ）再无监督（仅 ψ），步后重建 θ
    total_loss = 0.0
    num_batches = 0
    for _ in range(p.epochs):
        for (x_lb, y_lb), (x_ub,) in zip(l_loader, u_loader):
            x_lb, y_lb = x_lb.to(device), y_lb.to(device)
            x_ub = x_ub.to(device)

            x_lb_norm = prepare_input_batch(x_lb, p.dataset)
            x_ub_norm = prepare_input_batch(x_ub, p.dataset)

            # ── 监督分支（仅更新 σ）──
            optimizer_s.zero_grad()
            loss_s = p.lambda_s * F.cross_entropy(dm.theta(x_lb_norm), y_lb)
            loss_s.backward()
            optimizer_s.step()
            dm.sync_theta()

            # ── 无监督分支（仅更新 ψ）：单次 backward ──
            optimizer_u.zero_grad()
            y_logits = dm.theta(x_ub_norm)

            # L1(ψ) 与 L2(σ − ψ) 正则化项
            reg_l1 = sum(pp.abs().sum() for pp in dm.psi.parameters())
            reg_l2 = sum(
                (sp - pp).square().sum()
                for sp, pp in zip(dm.sigma.parameters(), dm.psi.parameters())
            )
            loss_u = p.lambda_l1 * reg_l1 + p.lambda_l2 * reg_l2

            # 计算置信度掩码下的 ICCS KL 与 CE 损失
            loss_iccs = compute_iccs_loss(
                dm=dm,
                x_ub_raw=x_ub,
                y_logits=y_logits,
                helper_net=helper_net,
                helper_states=helper_states,
                confidence=p.confidence,
                lambda_i=p.lambda_i,
                lambda_a=p.lambda_a,
                curr_round=p.curr_round,
                num_class=p.num_class,
                dataset_name=p.dataset,
            )
            loss_u = loss_u + loss_iccs

            loss_u.backward()
            optimizer_u.step()
            dm.sync_theta()

            total_loss += (loss_s.item() + loss_u.item()) / 2
            num_batches += 1

    # 7. C2S 差分处理
    trained_sigma = dm.sigma_state_dict()
    trained_psi = dm.psi_state_dict()

    # 区分参数与 buffer
    psi_param_names = set(trained_psi.keys())
    sigma_params_trained = {
        k: v for k, v in trained_sigma.items() if k in psi_param_names
    }
    sigma_buffers_trained = {
        k: v for k, v in trained_sigma.items() if k not in psi_param_names
    }
    server_sigma_params = {
        k: v for k, v in p.server_sigma_state.items() if k in psi_param_names
    }

    # 参数执行 C2S
    c2s_sigma_params = c2s_merge_state(
        sigma_params_trained,
        server_sigma_params,
        p.delta_thres,
        sparsify_thres=None,
    )
    c2s_psi = c2s_merge_state(
        trained_psi,
        p.server_psi_state,
        p.delta_thres,
        sparsify_thres=p.l1_thres,
    )

    # 组装完整 sigma（保留 theta 训练产生的 buffer）
    c2s_sigma = {**c2s_sigma_params, **sigma_buffers_trained}

    res = {
        "loss": total_loss / max(1, num_batches),
        "sigma": clone_cpu_state(c2s_sigma),
        "psi": clone_cpu_state(c2s_psi),
    }

    del dm, model, optimizer_s, optimizer_u, helper_net, helper_states
    del l_loader, u_loader, x_l, y_l, x_u
    torch.cuda.empty_cache()

    return res


class Server(BaseServer):
    def __init__(self, args):
        super().__init__(args, is_ssl=True)

        if args.ssl == "client":
            raise ValueError("fedmatch does not support ssl='client'")

        # 算法超参数
        self.confidence = args.confidence
        self.h_interval = args.h_interval
        self.num_helpers = args.num_helpers
        self.lambda_s = args.lambda_s
        self.lambda_i = args.lambda_i
        self.lambda_a = args.lambda_a
        self.lambda_l2 = args.lambda_l2
        self.lambda_l1 = args.lambda_l1
        self.l1_thres = args.l1_thres
        self.delta_thres = args.delta_thres
        self.psi_factor = args.psi_factor

        self.parameter_names = tuple(name for name, _ in self.model.named_parameters())

        # 分解模型状态
        self.sigma_state = clone_cpu_state(self.model.state_dict())
        self.psi_state = init_psi_state(
            self.sigma_state, self.parameter_names, self.psi_factor
        )

        # 维护每客户端历史状态
        self.client_sigma_states: dict[int, dict[str, torch.Tensor]] = {
            cid: clone_cpu_state(self.sigma_state) for cid in range(self.num_clients)
        }
        self.client_psi_states: dict[int, dict[str, torch.Tensor]] = {
            cid: clone_cpu_state(self.psi_state) for cid in range(self.num_clients)
        }
        self.seen_clients: set[int] = set()

        # 辅助历史跨轮缓存
        self.sigmas: dict[int, dict[str, torch.Tensor]] = {}
        self.psis: dict[int, dict[str, torch.Tensor]] = {}
        self.embeddings: dict[int, torch.Tensor] = {}

        # 构造固定嵌入噪声
        self.embedding_noise = self.build_embedding_noise()

    def build_embedding_noise(self) -> torch.Tensor:
        """根据实际数据集样本形状构建确定的固定噪声，并标准化。"""
        sample_x = None
        for cid in range(self.num_clients):
            t_set = self.train_sets[cid]
            if len(t_set.x) > 0:
                sample_x = t_set.x[0]
                break
        if sample_x is None:
            raise ValueError(
                "Cannot build embedding noise: all client datasets are empty"
            )

        gen = torch.Generator().manual_seed(42)
        shape = (1, *sample_x.shape)
        noise = (torch.randn(shape, generator=gen) * 0.5 + 0.5).clamp_(0.0, 1.0)
        noise = noise.to(torch.float32)

        spec = DATASET_SPECS.get(self.dataset)
        if spec is not None and spec["kind"] == "image":
            mean = torch.tensor(spec["mean"], dtype=torch.float32).view(-1, 1, 1)
            std = torch.tensor(spec["std"], dtype=torch.float32).view(-1, 1, 1)
            noise = (noise - mean) / std

        return noise

    def embed_client(self, cid: int) -> torch.Tensor:
        """将客户端模型映射为嵌入向量：使用该 cid 最新完整状态，在 eval 模式下前向。"""
        merged = merge_state(
            self.sigmas[cid], self.psis[cid], self.l1_thres, sparsify=False
        )
        self.model.load_state_dict(merged)
        self.model.eval()
        with torch.no_grad():
            vec = self.model(self.embedding_noise).squeeze(0)
        return vec.cpu().detach().clone()

    def get_helpers(self, cid: int) -> list[dict[str, torch.Tensor]] | None:
        """基于嵌入空间距离为客户端 cid 选取最多 num_helpers 个最相似 helper 的 ψ。"""
        if cid not in self.embeddings or len(self.embeddings) <= 1:
            return None
        candidate_ids = [i for i in self.embeddings if i != cid]
        if not candidate_ids:
            return None

        vectors = torch.stack([self.embeddings[i] for i in candidate_ids])
        target_vec = self.embeddings[cid].unsqueeze(0)
        dists = torch.cdist(vectors, target_vec).squeeze(1)
        order = torch.argsort(dists).tolist()
        k = min(self.num_helpers, len(candidate_ids))
        selected_ids = [candidate_ids[idx] for idx in order[:k]]
        return [self.psis[h] for h in selected_ids]

    def fit(self):
        num_join = max(1, int(self.num_clients * self.join_ratio))

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedMatch Round {r + 1}/{self.rounds} ---")

            # 1. 随机选择参与客户端
            selected = sorted(torch.randperm(self.num_clients)[:num_join].tolist())
            print(f"Selected clients: {selected}")

            # 2. helper 选取：每 h_interval 轮基于上一轮历史嵌入选取
            use_helpers = (r + 1) % self.h_interval == 0 and len(self.embeddings) > 1
            helper_map = (
                {cid: self.get_helpers(cid) for cid in selected} if use_helpers else {}
            )

            # 3. 为每个选中的客户端构造参数（包含 S2C 差分状态）
            psi_names = set(self.parameter_names)
            curr_server_sigma = clone_cpu_state(self.sigma_state)
            curr_server_psi = clone_cpu_state(self.psi_state)

            server_sigma_params = {
                k: v for k, v in curr_server_sigma.items() if k in psi_names
            }
            server_sigma_buffers = {
                k: v for k, v in curr_server_sigma.items() if k not in psi_names
            }

            parameters = []
            for base in self.build_base_params(selected):
                cid = base.client_id
                if cid not in self.seen_clients:
                    c_sigma_input = clone_cpu_state(curr_server_sigma)
                    c_psi_input = clone_cpu_state(curr_server_psi)
                else:
                    old_c_sigma = self.client_sigma_states[cid]
                    old_c_sigma_params = {
                        k: v for k, v in old_c_sigma.items() if k in psi_names
                    }
                    s2c_sig_params = s2c_merge_state(
                        server_sigma_params,
                        old_c_sigma_params,
                        self.delta_thres,
                        sparsify_thres=None,
                    )
                    c_sigma_input = {
                        **s2c_sig_params,
                        **server_sigma_buffers,
                    }
                    c_psi_input = s2c_merge_state(
                        curr_server_psi,
                        self.client_psi_states[cid],
                        self.delta_thres,
                        sparsify_thres=self.l1_thres,
                    )

                parameters.append(
                    Params(
                        **asdict(base),
                        sigma_state=c_sigma_input,
                        psi_state=c_psi_input,
                        server_sigma_state=curr_server_sigma,
                        server_psi_state=curr_server_psi,
                        curr_round=r,
                        helper_psi_states=helper_map.get(cid),
                        confidence=self.confidence,
                        lambda_s=self.lambda_s,
                        lambda_i=self.lambda_i,
                        lambda_a=self.lambda_a,
                        lambda_l2=self.lambda_l2,
                        lambda_l1=self.lambda_l1,
                        l1_thres=self.l1_thres,
                        delta_thres=self.delta_thres,
                    )
                )

            # 4. Ray 调度客户端任务
            results = self.run_clients(train, parameters)
            if not results:
                raise RuntimeError("No results returned from clients")

            # 5. 汇总：更新客户端历史与嵌入缓存
            sigma_list = []
            psi_list = []
            total_loss = 0.0
            for cid, res in results.items():
                total_loss += res["loss"]
                sigma_list.append(res["sigma"])
                psi_list.append(res["psi"])

                # 更新客户端历史
                self.client_sigma_states[cid] = clone_cpu_state(res["sigma"])
                self.client_psi_states[cid] = clone_cpu_state(res["psi"])
                self.seen_clients.add(cid)

                # 更新 helper 缓存
                self.sigmas[cid] = clone_cpu_state(res["sigma"])
                self.psis[cid] = clone_cpu_state(res["psi"])
                self.embeddings[cid] = self.embed_client(cid)

            # 6. 等权平均聚合全局 σ 与 ψ
            self.loss.append(total_loss / num_join)
            uniform_weights = [1.0 / len(sigma_list)] * len(sigma_list)
            self.sigma_state = aggregate_sigma_states(sigma_list, uniform_weights)
            self.psi_state = param_aggregate(psi_list, uniform_weights)

            # 7. 评估：ψ 稀疏化合并后评估
            eval_state = merge_state(
                self.sigma_state,
                self.psi_state,
                self.l1_thres,
                sparsify=True,
            )
            self.model.load_state_dict(eval_state)
            self.evaluate()

            print(f"Global Acc: {self.acc[-1]:.2f}%, Avg Loss: {self.loss[-1]:.4f}")
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def save(self):
        """保存指标与最终参数（global 与评估模型保持一致，使用 sparsify=True）。"""
        metrics = {"acc": self.acc, "loss": self.loss}
        params = {
            "global": merge_state(
                self.sigma_state,
                self.psi_state,
                self.l1_thres,
                sparsify=True,
            ),
            "sigma": self.sigma_state,
            "psi": self.psi_state,
        }
        self.deal_save(metrics, params)
