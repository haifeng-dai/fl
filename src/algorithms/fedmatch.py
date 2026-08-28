import copy
import math
import os
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, TensorDataset

from .utils import (
    BaseParams,
    BaseServer,
    fmt_num,
    get_model,
    kl_loss,
    param_aggregate,
    strong_augment,
)


def get_path(args):
    """构造实验日志文件名（含域配置与算法超参值）。"""
    args.file_name = (
        f"{args.common_name}_{fmt_num(args.confidence)}"
        f"_{fmt_num(args.h_interval)}_{fmt_num(args.num_helpers)}"
        f"_{fmt_num(args.lambda_s)}_{fmt_num(args.lambda_iccs)}"
        f"_{fmt_num(args.lambda_l2)}"
        f"_{fmt_num(args.lambda_l1)}_{fmt_num(args.l1_thres)}"
    )
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


@dataclass
class Params(BaseParams):
    sigma_state: dict[str, torch.Tensor]
    psi_state: dict[str, torch.Tensor]
    curr_round: int
    helper_psi_states: list[dict[str, torch.Tensor]] | None
    confidence: float
    lambda_s: float
    lambda_iccs: float
    lambda_l2: float
    lambda_l1: float
    l1_thres: float


class DecomposedModel(nn.Module):
    """θ = σ + ψ 分解模型。

    σ 与 ψ 为两个同构副本；θ 为前向实体，其每个参数在同步时重建为 σ + ψ
    （保留计算图，梯度可回传至 σ/ψ），评估时 ψ 可按 l1_thres 稀疏化。
    """

    def __init__(self, model, l1_thres):
        super().__init__()
        self.theta = model
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

    def load_sigma_psi(self, sigma_state, psi_state):
        """加载服务端下发的全局 σ/ψ 并重建 θ（worker 侧每轮执行一次）。"""
        self.sigma.load_state_dict(sigma_state)
        self.psi.load_state_dict(psi_state)
        self.sync_theta()

    def sigma_state_dict(self):
        return {k: v.cpu().detach().clone() for k, v in self.sigma.state_dict().items()}

    def psi_state_dict(self):
        return {k: v.cpu().detach().clone() for k, v in self.psi.state_dict().items()}


def merge_state(sigma_state, psi_state, l1_thres, sparsify=False):
    """逐元素合并 σ + ψ 为完整模型参数（服务端侧，输入均为 CPU state_dict）。"""
    merged = {}
    for k in sigma_state:
        p = psi_state[k]
        if sparsify:
            p = p * (p.abs() > l1_thres).to(p.dtype)
        merged[k] = sigma_state[k] + p
    return merged


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

    # 3. helper 模型权重准备：在 CPU 上计算好合并的权重，避免占用显存
    helper_states = []
    if p.helper_psi_states is not None:
        for hps in p.helper_psi_states:
            # 在 CPU 上完成合并
            merged = {k: p.sigma_state[k] + hps[k] for k in p.sigma_state}
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

    # 5. 无标签批大小按步数反推：两个 loader 长度一致 → zip 严格 1:1 配对，保证无标签数据在 num_steps 步内被完整遍历一遍
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
            optimizer_s.zero_grad()
            loss_s = p.lambda_s * F.cross_entropy(dm.theta(x_lb), y_lb)
            loss_s.backward()
            optimizer_s.step()
            dm.sync_theta()

            # 无监督损失（只更新 ψ），四项组成：
            # 1. inter-client KL：KL(helper ‖ local)，拉近本地分布与 helper 分布（helper 侧 detach，仅本地模型承接梯度）；仅在有 helper 且非首轮时启用；
            # 2. agreement 伪标签 CE：仅保留置信度 >= confidence 的样本，用投票伪标签监督强增强输出；
            #    第 1、2 项合并为 phi = mean(KL) + CE，统一 × lambda_iccs；
            # 3. L1(ψ) × lambda_l1：稀疏化正则（配合评估时的 l1_thres 硬阈值）；
            # 4. L2(σ − ψ) × lambda_l2：限制 ψ 偏离 σ 过大（disjoint 约束）。
            optimizer_u.zero_grad()
            y_logits = dm.theta(x_ub)

            with torch.no_grad():
                helper_logits = []
                if helper_net is not None:
                    for hs in helper_states:
                        helper_net.load_state_dict(hs)
                        helper_logits.append(helper_net(x_ub))

            if helper_logits:
                phi_kl = sum(
                    kl_loss(y_logits, h_logits) for h_logits in helper_logits
                ) / len(helper_logits)
                loss_u_kl = p.lambda_iccs * phi_kl
                loss_u_kl.backward()  # 第一次 backward：仅 KL 损失（存在 helper 时）
                total_u_loss_val = loss_u_kl.item()
            else:
                total_u_loss_val = 0.0

            # 第二次 backward：CE 伪标签损失 + L1/L2 正则合并执行
            y_probs = torch.softmax(y_logits.detach(), dim=1)
            conf_mask = y_probs.max(dim=1).values >= p.confidence
            loss_u_ce = None
            if int(conf_mask.sum().item()) > 0:
                x_conf = x_ub[conf_mask]
                y_conf_logits = y_logits.detach()[conf_mask]
                with torch.no_grad():
                    helper_conf_logits = []
                    if helper_net is not None:
                        for hs in helper_states:
                            helper_net.load_state_dict(hs)
                            helper_conf_logits.append(helper_net(x_conf))

                    # 取多数票。无 helper（首轮 / helper 未启用）时退化为本地 argmax。
                    votes = one_hot(y_conf_logits.argmax(dim=1), p.num_class)
                    if helper_conf_logits and p.curr_round > 0:
                        for h_logits in helper_conf_logits:
                            votes += one_hot(h_logits.argmax(dim=1), p.num_class)
                    y_pseudo = votes.argmax(dim=1)

                y_hard_logits = dm.theta(strong_augment(x_conf))
                loss_u_ce = p.lambda_iccs * F.cross_entropy(y_hard_logits, y_pseudo)

            # L1(ψ) 与 L2(σ − ψ) 正则化项（与 CE 损失一起完成第二次 backward）
            reg_l1 = [pp.abs().sum() for pp in dm.psi.parameters()]
            reg_l2 = [
                (sp - pp).square().sum()
                for sp, pp in zip(dm.sigma.parameters(), dm.psi.parameters())
            ]
            loss_reg = p.lambda_l1 * sum(reg_l1) + p.lambda_l2 * sum(reg_l2)

            loss_u_second = loss_reg if loss_u_ce is None else (loss_u_ce + loss_reg)
            loss_u_second.backward()  # 第二次 backward：CE + L1 + L2 正则
            total_u_loss_val += loss_u_second.item()

            optimizer_u.step()
            dm.sync_theta()
            total_loss += (loss_s.item() + total_u_loss_val) / 2
            num_batches += 1

    # 7. 回传前，必须显式清理大对象并清空显存，防止 Ray worker 持续泄漏
    res = {
        "loss": total_loss / max(1, num_batches),
        "sigma": dm.sigma_state_dict(),
        "psi": dm.psi_state_dict(),
    }

    del dm, model, optimizer_s, optimizer_u, helper_net, helper_states
    del l_loader, u_loader, x_l, y_l, x_u
    torch.cuda.empty_cache()

    return res


class Server(BaseServer):
    def __init__(self, args):
        super().__init__(args, is_ssl=True)

        if not self.is_sfd:
            raise ValueError("fedmatch requires SFD data (ssl must be 'sfd')")
        if self.unlabel_domain is None:
            raise ValueError("fedmatch requires unlabel_domain to be set")

        # 算法专属超参与置信度阈值（configs/algorithms.yaml 及 default.yaml 中配置）
        self.confidence = args.confidence
        self.h_interval = args.h_interval  # helper 重建周期（每 h_interval 轮）
        self.num_helpers = args.num_helpers  # 每客户端 helper 数量
        self.lambda_s = args.lambda_s
        self.lambda_iccs = args.lambda_iccs
        self.lambda_l2 = args.lambda_l2
        self.lambda_l1 = args.lambda_l1
        self.l1_thres = args.l1_thres

        # 分解模型状态（σ 初始化为全局模型权重，ψ 初始化为全零稀疏个性化参数）
        self.sigma_state = {
            k: v.cpu().detach().clone() for k, v in self.model.state_dict().items()
        }
        self.psi_state = {
            k: torch.zeros_like(v).cpu() for k, v in self.model.state_dict().items()
        }

        # 聚合权重与辅助历史缓存（按客户端 ID 索引）
        self.psis = {i: copy.deepcopy(self.psi_state) for i in range(self.num_clients)}
        self.embeddings = {}

        # 固定噪声输入，用于把客户端模型映射为嵌入向量
        gen = torch.Generator().manual_seed(42)
        self.embedding_noise = torch.randn(1, 3, 32, 32, generator=gen)
        # 本轮各客户端缓存（helper 选取用，每轮刷新）
        self.sigmas = {}

    def embed_client(self, cid):
        """将客户端模型映射为嵌入向量：合并 σ+ψ 后对固定噪声输入前向取输出。"""
        merged = merge_state(self.sigmas[cid], self.psis[cid], self.l1_thres)
        self.model.load_state_dict(merged)
        with torch.no_grad():
            vec = self.model(self.embedding_noise).squeeze(0)
        return vec.cpu().detach().clone()

    def get_helpers(self, cid):
        """基于嵌入空间距离，为客户端 cid 选取 num_helpers 个最相似 helper 的 ψ。"""
        if cid not in self.embeddings or len(self.embeddings) <= 1:
            return None
        ids = list(self.embeddings.keys())
        vectors = torch.stack([self.embeddings[i] for i in ids])
        dists = torch.cdist(vectors, self.embeddings[cid].unsqueeze(0)).squeeze(1)
        order = torch.argsort(dists).tolist()
        hids = []
        for idx in order:
            if ids[idx] == cid:
                continue
            hids.append(ids[idx])
            if len(hids) == self.num_helpers:
                break
        return [self.psis[h] for h in hids]

    def fit(self):
        num_join = max(1, int(self.num_clients * self.join_ratio))

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedMatch Round {r + 1}/{self.rounds} ---")

            # 1. 随机选择参与客户端
            selected = sorted(torch.randperm(self.num_clients)[:num_join].tolist())
            print(f"Selected clients: {selected}")

            # 2. helper 选取：每 h_interval 轮重建一次（基于上一轮嵌入，首轮无）
            use_helpers = (r + 1) % self.h_interval == 0 and self.embeddings
            helper_map = (
                {cid: self.get_helpers(cid) for cid in selected} if use_helpers else {}
            )

            # 3. 构造 worker 参数：通用 base + (σ, ψ) + 专属参数
            p = [
                Params(
                    **asdict(base),
                    sigma_state=self.sigma_state,
                    psi_state=self.psi_state,
                    curr_round=r,
                    helper_psi_states=helper_map.get(base.client_id),
                    confidence=self.confidence,
                    lambda_s=self.lambda_s,
                    lambda_iccs=self.lambda_iccs,
                    lambda_l2=self.lambda_l2,
                    lambda_l1=self.lambda_l1,
                    l1_thres=self.l1_thres,
                )
                for base in self.build_base_params(selected)
            ]
            results = self.run_clients(train, p)

            # 4. 汇总：缓存客户端 σ/ψ 与嵌入（供下一轮 helper 选取），累计损失
            sigma_list = []
            psi_list = []
            total_loss = 0.0
            self.embeddings = {}
            self.sigmas = {}
            self.psis = {}
            for cid, res in results.items():
                total_loss += res["loss"]
                sigma_list.append(res["sigma"])
                psi_list.append(res["psi"])
                self.sigmas[cid] = res["sigma"]
                self.psis[cid] = res["psi"]
                self.embeddings[cid] = self.embed_client(cid)

            # 5. 等权平均聚合全局 σ 与 ψ
            self.loss.append(total_loss / num_join)
            uniform_weights = [1 / len(sigma_list)] * len(sigma_list)
            self.sigma_state = param_aggregate(sigma_list, uniform_weights)
            self.psi_state = param_aggregate(psi_list, uniform_weights)

            # 6. 评估：ψ 硬阈值稀疏化后合并 σ+ψ 重建 θ
            merged = merge_state(
                self.sigma_state,
                self.psi_state,
                self.l1_thres,
                sparsify=True,
            )
            self.model.load_state_dict(merged)
            self.evaluate()

            print(
                f"Global Acc: {self.acc[-1]:.2f}%, "
                f"Source Acc: {self.acc_source[-1]:.2f}%, "
                f"Target Acc: {self.acc_target[-1]:.2f}%, "
                f"Avg Loss: {self.loss[-1]:.4f}"
            )
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def save(self):
        """保存指标与最终参数（global = σ+ψ 合并、sigma、psi 分开存档）。"""
        metrics = {"acc": self.acc, "loss": self.loss}
        metrics["acc_target"] = self.acc_target
        metrics["acc_source"] = self.acc_source
        params = {
            "global": merge_state(self.sigma_state, self.psi_state, self.l1_thres),
            "sigma": self.sigma_state,
            "psi": self.psi_state,
        }
        self.deal_save(metrics, params)
