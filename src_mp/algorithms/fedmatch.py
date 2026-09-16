import copy
from typing import Any, cast

import torch
from torch import nn
from torch.nn.functional import one_hot

from src.algorithms.utils.input import (
    normalize_image_tensor,
    prepare_input_batch,
)

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


def decomposed_parameter_names(model: nn.Module) -> tuple[str, ...]:
    """返回 FedMatch 中需要拆分的 Conv2d/Linear kernel 名称。"""
    names = []
    for module_name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if module.weight is None:
                raise ValueError(f"Decomposed module '{module_name}' has no weight")
            names.append(f"{module_name}.weight")
    if not names:
        raise ValueError("FedMatch model must contain Conv2d or Linear layers")
    return tuple(names)


class DecomposedModel(nn.Module):
    """θ = σ + ψ 分解模型。

    【核心设计】
    - 仅拆分 Conv2d/Linear 的权重 kernel（因为这是特征提取的核心，也是参数量主体）。
    - bias 与 BatchNorm 等 buffer 始终保留在 σ 中，不拆分，避免破坏网络基本结构。
    - 前向传播通过 torch.func.functional_call 临时组装权重 θ = σ + ψ，
      不修改原 PyTorch 模块结构，并通过梯度阻断（detach）实现双分支解耦训练。
    """

    def __init__(self, model: nn.Module, l1_thres: float):
        super().__init__()
        self.theta = copy.deepcopy(model)
        self.sigma = copy.deepcopy(model)
        self.l1_thres = l1_thres
        self.parameter_names = decomposed_parameter_names(model)
        self._psi_keys = {
            name: name.replace(".", "__") for name in self.parameter_names
        }
        sigma_parameters = dict(self.sigma.named_parameters())
        self.psi = nn.ParameterDict(
            {
                key: nn.Parameter(torch.empty_like(sigma_parameters[name]))
                for name, key in self._psi_keys.items()
            }
        )

    def forward(self, inputs: torch.Tensor, mode: str, sparse: bool = False):
        """以指定的可导分支执行 θ 前向。

        模式说明：
        - mode='sigma': 冻结 psi (psi.detach())，梯度只流向全局共享分支 sigma。
        - mode='psi':   冻结 sigma (sigma.detach())，梯度只流向个性化/半监督分支 psi。
        - mode='both':  sigma 和 psi 均计算梯度。
        """
        if mode not in {"sigma", "psi", "both"}:
            raise ValueError(f"Unknown decomposition mode: {mode}")

        parameters = {}
        for name, sigma in self.sigma.named_parameters():
            if name not in self._psi_keys:
                parameters[name] = sigma if mode != "psi" else sigma.detach()
                continue

            psi = self.psi[self._psi_keys[name]]
            if sparse:
                # 稀疏截断：小于 l1_thres 的绝对值直接归零
                psi = psi * (psi.abs() > self.l1_thres).to(psi.dtype)
            if mode == "sigma":
                parameters[name] = sigma + psi.detach()
            elif mode == "psi":
                parameters[name] = sigma.detach() + psi
            else:
                parameters[name] = sigma + psi

        return torch.func.functional_call(
            self.theta,
            (parameters, dict(self.sigma.named_buffers())),
            (inputs,),
        )

    def load_sigma_psi(
        self,
        sigma_state: dict[str, torch.Tensor],
        psi_state: dict[str, torch.Tensor],
    ):
        """加载下发的 σ 与 ψ 状态并初始化 θ。"""
        self.sigma.load_state_dict(sigma_state)
        psi_param_names = set(self.parameter_names)
        if set(psi_state.keys()) != psi_param_names:
            raise KeyError(
                f"psi_state keys mismatch: {set(psi_state.keys()) ^ psi_param_names}"
            )
        with torch.no_grad():
            for name, key in self._psi_keys.items():
                self.psi[key].copy_(psi_state[name])

    def sigma_state_dict(self) -> dict[str, torch.Tensor]:
        """导出训练后的完整 σ 状态（包含 bias 与所有 buffer）。"""
        return clone_state(self.sigma.state_dict())

    def psi_state_dict(self) -> dict[str, torch.Tensor]:
        """导出仅包含 parameters 的 psi 状态字典。"""
        return {
            name: self.psi[key].cpu().detach().clone()
            for name, key in self._psi_keys.items()
        }

    def l2_regularization(self) -> torch.Tensor:
        """计算 L2 对齐正则项: ||sigma_detach - psi||^2，约束 psi 不要偏离全局 sigma 太远。"""
        terms = [
            (self.sigma.get_parameter(name).detach() - self.psi[key]).square().sum()
            for name, key in self._psi_keys.items()
        ]
        return torch.stack(terms).sum()


class Client(BaseClientExecutor):
    """FedMatch 客户端，分别优化 sigma 和 psi 两个参数分支。"""

    def __init__(self, args, device, num_class, **kwargs):
        super().__init__(args, device, num_class, **kwargs)
        if args.model != "cnn":
            raise ValueError(
                "FedMatch currently supports only model='cnn'; "
                "the official ResNet-9 must be ported separately."
            )

        self.conf = args.conf  # 置信度阈值
        self.lambda_s = args.lambda_s  # 有监督损失权重
        self.lambda_i = args.lambda_i  # 跨客户端一致性权重 (KL 散度)
        self.lambda_a = args.lambda_a  # 伪标签分类损失权重
        self.lambda_l2 = args.lambda_l2  # 参数对齐正则权重 ||sigma - psi||^2
        self.lambda_l1 = args.lambda_l1  # 稀疏正则权重 sum(|psi|)
        self.l1_thres = args.l1_thres  # psi 绝对值硬截断阈值
        self.delta_thres = args.delta_thres  # 差分传输更新阈值
        self.unlabeled_ratio = args.unlabeled_ratio

        self.dm = DecomposedModel(self.model, self.l1_thres).to(self.device)
        self.optimizer_sigma = torch.optim.SGD(
            self.dm.sigma.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        self.optimizer_psi = torch.optim.SGD(
            self.dm.psi.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        self.helper_nets: list[nn.Module] = []

        self.curr_round = 0
        self.embedding_noise = self._build_embedding_noise(args)

    def _build_embedding_noise(self, args):
        """根据数据集规格在 GPU 上构建确定的固定噪声探针，全流程保持不变。"""
        channels = (
            3
            if (
                "cifar" in self.dataset
                or self.dataset
                in {
                    "tiny_imagenet",
                    "flowers102",
                    "cars",
                    "gtsrb",
                    "cinic10",
                    "svhn",
                    "pacs",
                    "officehome",
                    "vlcs",
                    "domainnet",
                }
            )
            else 1
        )
        if self.dataset in ["mnist", "fashionmnist", "femnist", "emnist"]:
            size = 28
        elif self.dataset in ["cars", "flowers102"]:
            size = 224
        elif self.dataset == "tiny_imagenet":
            size = 64
        else:
            size = 32
        gen = torch.Generator().manual_seed(args.seed)
        noise = (
            torch.randn((1, channels, size, size), generator=gen) * 0.49 + 0.49
        ).clamp_(0.0, 1.0)
        return normalize_image_tensor(noise, self.dataset).to(self.device)

    def c2s_merge_state(
        self, client_state, server_state, sparsify_thres: float | None = None
    ):
        """在当前设备完成 C2S 稀疏差分合并，调用方负责最终 CPU 传输。"""
        res: dict[str, torch.Tensor] = {}
        for k, c_val in client_state.items():
            s_val = server_state[k]
            c_eval = c_val
            if sparsify_thres is not None:
                c_eval = c_eval * (c_eval.abs() > sparsify_thres).to(c_eval.dtype)
            # 差分掩码：仅保留更新幅度显著的参数
            mask = (c_eval - s_val).abs() > self.delta_thres
            res[k] = torch.where(mask, c_val, s_val).clone()
        return res

    def _device_state(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """把通信边界上的 CPU 状态一次性移入当前 GPU。"""
        return {name: value.to(self.device) for name, value in state.items()}

    def _load_helper_nets(self, states: list[dict[str, torch.Tensor]]):
        """每个客户端任务仅加载一次 Helper，之后 batch 内只执行 GPU 前向。"""
        while len(self.helper_nets) < len(states):
            helper = build_model(self.model_param).to(self.device).eval()
            for parameter in helper.parameters():
                parameter.requires_grad_(False)
            self.helper_nets.append(helper)
        for helper, state in zip(self.helper_nets, states, strict=False):
            helper.load_state_dict(state)
            helper.eval()

    def iccs_loss(self, x_ub_raw, y_ub_raw, y_logits):
        """计算跨客户端无监督损失（ICCS）：
        1. 过滤最大预测概率 >= conf 的高置信度样本；
        2. Consistency KL 损失：约束本地 logits 与 Helper 辅助模型的预测分布对齐；
        3. Agreement 伪标签 CE 损失：本地模型与 Helper 模型共同投票确定伪标签，并在强数据增强下监督。
        """
        y_probs = torch.softmax(y_logits.detach(), dim=1)
        conf_mask = y_probs.max(dim=1).values >= self.conf

        # 若批次中没有超过置信度的样本，则不计算 ICCS 损失
        if not conf_mask.any():
            return torch.tensor(0.0, device=x_ub_raw.device), 0, 0

        x_conf_raw = x_ub_raw[conf_mask]
        local_conf_logits = y_logits[conf_mask]  # 保留计算图以回传梯度
        y_conf_logits_det = local_conf_logits.detach()

        # 获取 Helper 辅助模型对高置信度样本的预测 logits
        with torch.no_grad():
            helper_conf_logits = []
            if self._active_helper_nets:
                x_conf_norm = prepare_input_batch(x_conf_raw, self.dataset)
                helper_conf_logits = [
                    helper(x_conf_norm) for helper in self._active_helper_nets
                ]

        loss_iccs = torch.tensor(0.0, device=x_ub_raw.device)

        # -----------------------------------------------------------------
        # ICCS 目标 1: 跨客户端一致性 KL 散度约束 (Inter-client consistency)
        # -----------------------------------------------------------------
        if helper_conf_logits and self.curr_round > 0:
            phi_kl = sum(
                kl_loss(local_conf_logits, h_logits) for h_logits in helper_conf_logits
            ) / len(helper_conf_logits)
            loss_iccs = loss_iccs + self.lambda_i * phi_kl

        # -----------------------------------------------------------------
        # ICCS 目标 2: 基于共识的伪标签学习 (Agreement-based pseudo labeling)
        # 本地预测与 Helper 预测通过独热编码投票累计
        # -----------------------------------------------------------------
        with torch.no_grad():
            votes = one_hot(y_conf_logits_det.argmax(dim=1), self.num_class)
            if helper_conf_logits and self.curr_round > 0:
                for h_logits in helper_conf_logits:
                    votes += one_hot(h_logits.argmax(dim=1), self.num_class)
            y_pseudo = votes.argmax(dim=1)

        # 对高置信度图像进行强数据增强后输入 psi 分支进行分类
        y_hard_logits = self.dm(strong_augment(x_conf_raw, self.dataset), mode="psi")
        loss_ce = self.lambda_a * ce_loss(y_hard_logits, y_pseudo)
        loss_iccs = loss_iccs + loss_ce

        pseudo_cnt = int(conf_mask.sum().item())
        y_ub_device = y_ub_raw.to(y_pseudo.device)
        pseudo_corr = int((y_pseudo == y_ub_device[conf_mask]).sum().item())
        return loss_iccs, pseudo_cnt, pseudo_corr

    def reset_optimizer(self):
        for opt in (self.optimizer_sigma, self.optimizer_psi):
            for group in opt.param_groups:
                group.update(
                    lr=self.lr,
                    momentum=self.momentum,
                    weight_decay=self.weight_decay,
                )
            opt.state.clear()

    def train(self):
        payload = cast(dict[str, Any], self.payload)
        self.curr_round = payload["curr_round"]
        self.dm.load_sigma_psi(payload["sigma_state"], payload["psi_state"])
        self.reset_optimizer()

        helper_states = payload.get("helper_psi_states") or []
        self._load_helper_nets(helper_states)
        self._active_helper_nets = self.helper_nets[: len(helper_states)]
        total = batches = 0
        self._pseudo_count = self._pseudo_correct = 0
        for _ in range(self.epochs):
            total, batches = self.run_epoch(total, batches)
        pseudo_count, pseudo_correct = self._pseudo_count, self._pseudo_correct
        # 保持训练状态在 GPU 上完成 C2S；只有最终回传状态才落到 CPU。
        trained_sigma = {
            name: value.detach() for name, value in self.dm.sigma.state_dict().items()
        }
        trained_psi = {
            name: self.dm.psi[key].detach()
            for name, key in self.dm._psi_keys.items()
        }
        sigma_parameter_names = {name for name, _ in self.dm.sigma.named_parameters()}
        server_sigma = self._device_state(payload["server_sigma_state"])
        c2s_sigma_device = {
            **self.c2s_merge_state(
                {
                    name: value
                    for name, value in trained_sigma.items()
                    if name in sigma_parameter_names
                },
                {
                    name: value
                    for name, value in server_sigma.items()
                    if name in sigma_parameter_names
                },
            ),
            **{
                name: value
                for name, value in trained_sigma.items()
                if name not in sigma_parameter_names
            },
        }
        c2s_psi_device = self.c2s_merge_state(
            trained_psi,
            self._device_state(payload["server_psi_state"]),
            self.l1_thres,
        )
        c2s_sigma = clone_state(c2s_sigma_device)
        c2s_psi = clone_state(c2s_psi_device)
        with torch.no_grad():
            embedding = (
                self.dm(self.embedding_noise, mode="both")
                .flatten()
                .cpu()
                .detach()
                .clone()
            )
        return ClientResult(
            self.client_id,
            total / max(1, batches),
            c2s_sigma,
            {
                "sigma": c2s_sigma,
                "psi": c2s_psi,
                "embedding": embedding,
                "pseudo_count": pseudo_count,
                "pseudo_correct": pseudo_correct,
            },
        )

    def run_epoch(
        self,
        total: float,
        batches: int,
    ) -> tuple[float, int]:
        """执行一个 FedMatch 本地 epoch：交替优化 sigma 分支与 psi 分支。"""
        for (x_lb, y_lb), (x_ub, y_ub) in self.get_ssl_loaders():
            x_lb, y_lb = x_lb.to(self.device), y_lb.to(self.device)
            x_ub, y_ub = x_ub.to(self.device), y_ub.to(self.device)
            # =================================================================
            # 【阶段 1：有监督学习】仅更新全局共享分支 sigma
            # =================================================================
            self.optimizer_sigma.zero_grad()
            x_lb_w = prepare_input_batch(x_lb, self.dataset)
            logits_l = self.dm(x_lb_w, mode="sigma")  # mode='sigma' 冻结 psi
            loss_s = self.lambda_s * ce_loss(logits_l, y_lb)
            self.check_nan(loss_s)
            loss_s.backward()
            self.optimizer_sigma.step()

            # =================================================================
            # 【阶段 2：无监督学习与跨客户端协助】仅更新个性化分支 psi
            # 损失包含: L1 稀疏正则 + L2 对齐正则 + ICCS 跨端一致性/伪标签损失
            # =================================================================
            self.optimizer_psi.zero_grad()
            x_uw = prepare_input_batch(x_ub, self.dataset)
            # 1) L1 稀疏正则: 惩罚 psi 幅度
            reg_l1 = sum(p.abs().sum() for p in self.dm.psi.parameters())
            # 2) L2 对齐正则: 防止 psi 偏离 sigma 太远
            reg_l2 = self.dm.l2_regularization()
            loss_u = self.lambda_l1 * reg_l1 + self.lambda_l2 * reg_l2
            self.check_nan(loss_u)

            # 3) ICCS 跨端一致性与共识伪标签损失
            iccs, count, correct = self.iccs_loss(
                x_ub,
                y_ub,
                self.dm(x_uw, mode="psi"),  # mode='psi' 冻结 sigma
            )
            loss_u_total = loss_u + iccs
            self.check_nan(loss_u_total)
            loss_u_total.backward()
            self.optimizer_psi.step()
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
        if args.model != "cnn":
            raise ValueError(
                "FedMatch currently supports only model='cnn'; "
                "the official ResNet-9 must be ported separately."
            )
        if args.ssl in ("none", "client"):
            raise ValueError(
                "FedMatch 仅支持 sample、double、sfd 等每客户端含 L/U 的 SSL 模式。"
            )
        super().__init__(args, devices)
        self.conf = args.conf
        self.h_interval = args.h_interval
        self.num_helpers = args.num_helpers
        self.l1_thres = args.l1_thres
        self.delta_thres = args.delta_thres
        self.psi_factor = args.psi_factor
        self.dataset = args.dataset

        # 被拆分的层的名称
        self.parameter_names = decomposed_parameter_names(self.model)
        # 监督训练部分参数
        self.sigma_state = clone_state(self.model.state_dict())
        # 无监督训练部分参数
        self.psi_state = self._init_psi_state()

        self.client_sigma_states = {
            cid: clone_state(self.sigma_state) for cid in range(self.num_clients)
        }
        self.client_psi_states = {
            cid: clone_state(self.psi_state) for cid in range(self.num_clients)
        }

        # 训练过的客户端
        self.seen_clients = set()
        self.current_round = 0
        self.psis, self.embeddings = {}, {}
        self.pseudo_acc, self.pseudo_count = [], []

    def _init_psi_state(self) -> dict[str, torch.Tensor]:
        """初始化 psi 状态字典：对分解的卷积/全连接层权重缩放 psi_factor。"""
        return {
            name: (self.sigma_state[name] * self.psi_factor).cpu().clone()
            for name in self.parameter_names
        }

    def train_payloads(self):
        """为本轮选中的客户端打包下发数据"""
        server_sigma = clone_state(self.sigma_state)
        server_psi = clone_state(self.psi_state)
        names = set(self.parameter_names)
        sigma_params = {k: v for k, v in server_sigma.items() if k in names}
        payloads = {}
        for cid in self.selected:
            if cid not in self.seen_clients:
                sigma, psi = server_sigma, server_psi
            else:
                # 客户端模型 sigma
                client_sigma = self.client_sigma_states[cid]
                # 可拆分部分参数，卷积权重
                client_sigma_t = {k: v for k, v in client_sigma.items() if k in names}
                # 不可拆分部分参数，bias、BatchNorm
                client_sigma_u = {
                    k: v for k, v in server_sigma.items() if k not in names
                }
                # 把“按变化量差分压缩后的卷积权重”与“没做差分、全量同步的偏置和BN”拼成一个完整的下发模型字典。
                sigma = {
                    **self.s2c_merge_state(sigma_params, client_sigma_t),
                    **client_sigma_u,
                }
                # 服务器 psi 稀疏化
                psi = self.s2c_merge_state(
                    server_psi,
                    self.client_psi_states[cid],
                    sparsify=True,
                )
            # 每隔 h_interval 轮下发一次匹配的 helper 状态
            if (self.current_round + 1) % self.h_interval == 0:
                helper_client = self.helpers(cid)
            else:
                helper_client = None
            payloads[cid] = {
                "sigma_state": sigma,
                "psi_state": psi,
                "server_sigma_state": server_sigma,
                "server_psi_state": server_psi,
                "helper_psi_states": helper_client,
                "curr_round": self.current_round,
            }
        return payloads

    def apply_result(self, results):
        """接收客户端训练结果：记录各端状态、更新模型 Embedding 指纹、服务端聚合 sigma/psi。"""
        names = set(self.parameter_names)
        sigmas, psis = [], []
        for cid in self.selected:
            sigma = results[cid].state
            psi = results[cid].payload["psi"]
            self.client_sigma_states[cid] = clone_state(sigma)
            self.client_psi_states[cid] = clone_state(psi)
            self.seen_clients.add(cid)
            self.psis[cid] = clone_state(psi)
            # 更新该客户端最新的表征指纹
            self.embeddings[cid] = results[cid].payload["embedding"]
            sigmas.append(sigma)
            psis.append(psi)

        # 全局加权聚合
        weights = [1.0 / self.num_selected] * self.num_selected
        self.sigma_state = self.aggregate_states(sigmas, weights)
        self.psi_state = {
            k: v for k, v in self.aggregate_states(psis, weights).items() if k in names
        }
        # 更新服务端用于评估的全局模型 θ = σ + ψ
        self.model.load_state_dict(self.merge_state(self.psi_state, True))

    def run_round(self):
        """一轮联邦学习核心循环：下发 -> 并行训练 -> 聚合更新 -> 评估。"""
        results = self.train_clients()
        self.apply_result(results)
        loss = sum(r.loss for r in results.values()) / self.num_selected
        acc = self.evaluate()
        count = sum(r.payload["pseudo_count"] for r in results.values())
        correct = sum(r.payload["pseudo_correct"] for r in results.values())
        pseudo_acc = 100.0 * correct / max(1, count)
        self.record_round(loss, acc, count, pseudo_acc)
        self.current_round += 1

    def progress_metrics(self):
        metrics = super().progress_metrics()
        metrics.update(
            pseudo_acc=f"{self.pseudo_acc[-1]:.2f}%", pseudo_count=self.pseudo_count[-1]
        )
        return metrics

    def record_round(self, loss, accuracy, count, pseudo_acc):
        super().record_round(loss, accuracy)
        self.pseudo_count.append(count)
        self.pseudo_acc.append(pseudo_acc)

    def round_log_metrics(self):
        metrics = super().round_log_metrics()
        metrics.update(
            pseudo_acc=f"{self.pseudo_acc[-1]:.2f}%", pseudo_count=self.pseudo_count[-1]
        )
        return metrics

    def helpers(self, cid):
        """KNN 搜索：计算当前客户端与其他客户端模型 Embedding 的欧氏距离，推荐最近的 num_helpers 个邻居。"""
        if cid not in self.embeddings:
            return None
        ids = [i for i in self.embeddings if i != cid]
        if not ids:
            return None
        target = self.embeddings[cid].unsqueeze(0)
        distances = torch.cdist(
            torch.stack([self.embeddings[i] for i in ids]), target
        ).flatten()
        return [
            self.merge_state(self.psis[ids[i]])
            for i in torch.argsort(distances)[: self.num_helpers]
        ]

    def s2c_merge_state(
        self,
        server_state: dict[str, torch.Tensor],
        client_state: dict[str, torch.Tensor],
        sparsify: bool = False,
    ) -> dict[str, torch.Tensor]:
        """在服务器 GPU 上完成 S2C 稀疏差分，下发前再转为 CPU 状态。"""
        res: dict[str, torch.Tensor] = {}
        for k, s_val in server_state.items():
            s_val = s_val.to(self.device)
            c_val = client_state[k].to(self.device)
            s_eval, c_eval = s_val, c_val
            if sparsify:
                s_eval = s_eval * (s_eval.abs() > self.l1_thres).to(s_eval.dtype)
                c_eval = c_eval * (c_eval.abs() > self.l1_thres).to(c_eval.dtype)
            mask = (s_eval - c_eval).abs() > self.delta_thres
            res[k] = torch.where(mask, s_val, c_val).cpu().clone()
        return res

    def aggregate_states(self, states, weights):
        """在服务端 GPU 逐项加权聚合，最终仅保存可通信的 CPU 状态。"""
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
                aggregate = torch.zeros_like(ref_val, device=self.device)
                for state, weight in zip(states, norm_w, strict=True):
                    aggregate.add_(state[k].to(self.device), alpha=weight)
                res[k] = aggregate.cpu().clone()
            else:
                res[k] = ref_val.cpu().detach().clone()
        return res

    def merge_state(self, psi_state: dict[str, torch.Tensor], sparsify=False):
        """在服务器 GPU 逐元素合并 σ + ψ，返回可下发/评估的 CPU 状态。"""
        merged: dict[str, torch.Tensor] = {}
        for k, s_val in self.sigma_state.items():
            s_val = s_val.to(self.device)
            if k in psi_state:
                p_val = psi_state[k].to(self.device)
                if sparsify:
                    p_val = p_val * (p_val.abs() > self.l1_thres).to(p_val.dtype)
                merged[k] = (s_val + p_val).cpu().clone()
            else:
                merged[k] = s_val.cpu().clone()
        return merged
