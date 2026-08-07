import copy
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .utils import (
    BaseServer,
    ce_loss,
    fmt_num,
    get_model,
    kl_loss,
    param_aggregate,
    strong_augment,
)


def get_path(args):
    """构造实验日志文件名（区分数据划分、学习率、batch size、置信度阈值等关键设定）。"""
    dp = "sfd" if args.sfd else ("fdg" if args.fdg else None)
    sd = args.selected_domains
    ud = args.unlabel_domain

    parts = [args.common_name]
    if dp is not None:
        parts.append(f"dp{dp}")
    if sd:
        sd_str = sd.replace(",", "_") if isinstance(sd, str) else "_".join(map(str, sd))
        parts.append(f"sd{sd_str}")
    if ud is not None:
        parts.append(f"ud{ud}")
    parts.append(f"lr{fmt_num(args.lr)}")
    parts.append(f"bs{args.batch_size}")
    parts.append(f"ci{fmt_num(args.confidence)}")

    args.file_name = "_".join(parts)
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


class DecomposedModel(nn.Module):
    """θ = σ + ψ 分解模型。

    σ 与 ψ 为两个同构副本；θ 为前向实体，其每个参数在同步时重建为 σ + ψ
    （保留计算图，梯度可回传至 σ/ψ），评估时 ψ 可按 l1_thres 稀疏化。
    """

    def __init__(self, model, psi_factor, l1_thres):
        super().__init__()
        self.theta = model
        self.sigma = copy.deepcopy(model)
        self.psi = copy.deepcopy(model)
        self.l1_thres = l1_thres
        # ψ 初始化为 σ 的 psi_factor 倍（官方 0.2），此后 σ 与 ψ 分别被
        # 两个独立优化器驱动，实现 disjoint learning（监督只动 σ、无监督只动 ψ）。
        with torch.no_grad():
            for p, s in zip(self.psi.parameters(), self.sigma.parameters()):
                p.copy_(s * psi_factor)
        self.sync_theta()

    def sync_theta(self, sparsify=False):
        """将 θ 的每个参数重建为 σ + ψ（此时合并结果仍是张量表达式，保留计算图）。

        sparsify=True 时先对 ψ 做硬阈值：|ψ| <= l1_thres 的权重归零（评估阶段用）。
        θ 的参数改为 Buffer 挂载：
          - 前向 θ(x) 可正常执行；
          - 由于 σ/ψ 仍是可求导叶子，θ 前向产生的梯度能经 merged 表达式回传。
        """
        for name, sp in self.sigma.named_parameters():
            pp = self.psi.get_parameter(name)
            merged = sp + (
                pp * (pp.abs() > self.l1_thres).to(pp.dtype) if sparsify else pp
            )
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

    def sigma_state_dict_cpu(self):
        # 回传前搬回 CPU 并克隆，防止 Ray 对象存储出现悬空引用
        return {k: v.cpu().detach().clone() for k, v in self.sigma.state_dict().items()}

    def psi_state_dict_cpu(self):
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


def pseudo_labeling(local_logits, helper_logits_list, num_classes):
    """agreement 伪标签：本地模型与各 helper 模型对每个样本的 argmax 投票，取多数票。

    - 无 helper（首轮 / helper 未启用）：退化为本地 argmax；
    - 返回与 batch 等长的伪标签张量。
    """
    votes = F.one_hot(local_logits.argmax(dim=1), num_classes).float()
    for h_logits in helper_logits_list:
        votes = votes + F.one_hot(h_logits.argmax(dim=1), num_classes).float()
    return votes.argmax(dim=1)


def unsupervised_loss(
    theta,
    sigma,
    psi,
    x,
    helper_models,
    curr_round,
    num_classes,
    confidence,
    lambda_i,
    lambda_a,
    lambda_l1,
    lambda_l2,
):
    """无监督损失（只更新 ψ），四项组成：

    1. inter-client KL：KL(helper ‖ local) × lambda_i，拉近本地分布与 helper 分布
       （helper 侧 detach，仅本地模型承接梯度）；仅在有 helper 且非首轮时启用；
    2. agreement 伪标签 CE：仅保留置信度 >= confidence 的样本，用投票伪标签
       监督强增强输出，× lambda_a；
    3. L1(ψ) × lambda_l1：稀疏化正则（配合评估时的 l1_thres 硬阈值）；
    4. L2(σ − ψ) × lambda_l2：限制 ψ 偏离 σ 过大（disjoint 约束）。
    返回 (loss, num_conf)，num_conf 供调用方观测高置信样本规模。
    """
    loss = torch.tensor(0.0, device=x.device)
    # 置信度过滤：仅 max softmax 概率达标的样本进入无监督训练
    y_probs = torch.softmax(theta(x), dim=1)
    conf_mask = y_probs.max(dim=1).values >= confidence
    num_conf = int(conf_mask.sum().item())
    if num_conf > 0:
        x_conf = x[conf_mask]
        y_conf_logits = theta(x_conf)
        helper_logits = [h(x_conf).detach() for h in helper_models]
        if helper_logits and curr_round > 0:
            for h_logits in helper_logits:
                loss = loss + lambda_i * kl_loss(y_conf_logits, h_logits) / len(
                    helper_logits
                )
        y_hard_logits = theta(strong_augment(x_conf))
        y_pseudo = pseudo_labeling(y_conf_logits.detach(), helper_logits, num_classes)
        loss = loss + lambda_a * ce_loss(y_hard_logits, y_pseudo)
    for sp, pp in zip(sigma.parameters(), psi.parameters()):
        loss = loss + lambda_l1 * pp.abs().sum()
        loss = loss + lambda_l2 * (sp - pp).square().sum()
    return loss, num_conf


def train(params):
    """Ray Worker：单客户端本地训练。

    参数与 Server.fit 中 build_base_params + append 的追加顺序严格对应：
    base 11 项（cid/gpu/states/train_set/model/dataset/lr/batch_size/epochs/
    feature_dim/num_class）+ 轮次动态 2 项（curr_round/helper_psi_states）
    + 算法专属超参 8 项（confidence/psi_factor/lambda_s/lambda_i/lambda_a/
    lambda_l2/lambda_l1/l1_thres）。
    """
    (
        _,
        device,
        states,
        train_set,
        model_name,
        dataset_name,
        lr,
        batch_size,
        epochs,
        feature_dim,
        num_class,
        curr_round,
        helper_psi_states,
        confidence,
        psi_factor,
        lambda_s,
        lambda_i,
        lambda_a,
        lambda_l2,
        lambda_l1,
        l1_thres,
    ) = params
    sigma_state, psi_state = states

    # 1. 重建分解模型：θ 前向实体 + σ/ψ 可训练副本
    model = get_model(model_name, dataset_name, num_class, feature_dim).to(device)
    dm = DecomposedModel(model, psi_factor, l1_thres)
    dm.load_sigma_psi(sigma_state, psi_state)

    # 2. 两个独立优化器实现 disjoint learning：σ ← 监督损失，ψ ← 无监督损失
    optimizer_s = torch.optim.SGD(dm.sigma.parameters(), lr=lr)
    optimizer_u = torch.optim.SGD(dm.psi.parameters(), lr=lr)

    # 3. helper 模型：全局 σ + 各 helper 的 ψ 恢复完整模型，仅参与前向（eval）
    helper_models = []
    if helper_psi_states is not None:
        for hps in helper_psi_states:
            hm = get_model(model_name, dataset_name, num_class, feature_dim).to(device)
            merged = {
                k: sigma_state[k].to(device) + hps[k].to(device) for k in sigma_state
            }
            hm.load_state_dict(merged)
            hm.eval()
            helper_models.append(hm)

    # 4. 按 is_labeled 掩码切出有标签/无标签数据（SFD：label_domain 有标签、unlabel_domain 无标签）
    labeled_mask = (
        train_set.is_labeled
        if train_set.is_labeled is not None
        else torch.ones(len(train_set), dtype=torch.bool)
    )
    x_l = train_set.x[labeled_mask]
    y_l = train_set.y[labeled_mask]
    x_u = train_set.x[~labeled_mask]

    # 5. 无标签批大小按步数反推：两个 loader 长度一致 → zip 严格 1:1 配对，
    #    保证无标签数据在 num_steps 步内被完整遍历一遍（官方遍历语义）
    num_steps = round(len(x_l) / batch_size)
    bsize_u = math.ceil(len(x_u) / max(1, num_steps))

    l_loader = DataLoader(TensorDataset(x_l, y_l), batch_size=batch_size, shuffle=True)
    u_loader = DataLoader(TensorDataset(x_u), batch_size=bsize_u, shuffle=True)

    # 6. 本地训练：每步先监督（仅 σ）再无监督（仅 ψ），步后重建 θ
    total_loss = 0.0
    num_batches = 0
    for _ in range(epochs):
        for (x_lb, y_lb), (x_ub,) in zip(l_loader, u_loader):
            x_lb, y_lb = x_lb.to(device), y_lb.to(device)
            x_ub = x_ub.to(device)
            optimizer_s.zero_grad()
            loss_s = lambda_s * ce_loss(dm.theta(x_lb), y_lb)
            loss_s.backward()
            optimizer_s.step()
            dm.sync_theta()

            optimizer_u.zero_grad()
            loss_u, _ = unsupervised_loss(
                dm.theta,
                dm.sigma,
                dm.psi,
                x_ub,
                helper_models,
                curr_round,
                num_class,
                confidence,
                lambda_i,
                lambda_a,
                lambda_l1,
                lambda_l2,
            )
            loss_u.backward()
            optimizer_u.step()
            dm.sync_theta()
            total_loss += (loss_s.item() + loss_u.item()) / 2
            num_batches += 1

    # 7. 回传：σ/ψ 独立上传（服务端分别聚合），均为 CPU 克隆
    return {
        "loss": total_loss / max(1, num_batches),
        "sigma": dm.sigma_state_dict_cpu(),
        "psi": dm.psi_state_dict_cpu(),
    }


class Server(BaseServer):
    def __init__(self, args):
        super().__init__(False, args)

        self.unlabel_domain = args.unlabel_domain
        if not self.args.sfd:
            raise ValueError("fedmatch requires SFD data (sfd must be enabled)")
        if self.unlabel_domain is None:
            raise ValueError("fedmatch requires unlabel_domain to be set")

        # 算法专属超参（configs/algorithms.yaml 中配置）
        self.h_interval = args.h_interval  # helper 重建周期（每 h_interval 轮）
        self.num_helpers = args.num_helpers  # 每客户端 helper 数量
        self.l1_thres = args.l1_thres  # ψ 稀疏化硬阈值

        # 全局 σ/ψ（等权平均聚合维护）；ψ 初始 = σ × psi_factor
        self.sigma_state = {
            k: v.cpu().detach().clone() for k, v in self.model.state_dict().items()
        }
        self.psi_state = {k: v * args.psi_factor for k, v in self.sigma_state.items()}
        # 固定噪声输入，用于把客户端模型映射为嵌入向量
        gen = torch.Generator().manual_seed(42)
        self.embedding_noise = torch.randn(1, 3, 32, 32, generator=gen)
        # 每轮客户端状态缓存（用于嵌入计算与 helper 选取，仅跨一轮有效）
        self.cid_to_vector = {}
        self.cid_to_sigma_state = {}
        self.cid_to_psi_state = {}

    def embed_client(self, cid):
        """将客户端模型映射为嵌入向量：合并 σ+ψ 后对固定噪声输入前向取输出。"""
        merged = merge_state(
            self.cid_to_sigma_state[cid], self.cid_to_psi_state[cid], self.l1_thres
        )
        self.model.load_state_dict(merged)
        with torch.no_grad():
            vec = self.model(self.embedding_noise).squeeze(0)
        return vec.cpu().detach().clone()

    def get_helpers(self, cid):
        """按嵌入欧氏距离取该客户端的最近邻 num_helpers 个客户端，返回其 ψ。"""
        if cid not in self.cid_to_vector:
            return None
        ids = list(self.cid_to_vector.keys())
        vectors = torch.stack([self.cid_to_vector[i] for i in ids])
        dists = torch.cdist(vectors, self.cid_to_vector[cid].unsqueeze(0)).squeeze(1)
        order = torch.argsort(dists).tolist()
        hids = []
        for idx in order:
            if ids[idx] == cid:
                continue
            hids.append(ids[idx])
            if len(hids) == self.num_helpers:
                break
        return [self.cid_to_psi_state[h] for h in hids]

    def fit(self):
        num_join = max(1, int(self.num_clients * self.args.join_ratio))

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedMatch Round {r + 1}/{self.rounds} ---")

            # 1. 随机选择参与客户端
            selected = sorted(torch.randperm(self.num_clients)[:num_join].tolist())
            print(f"Selected clients: {selected}")

            # 2. helper 选取：每 h_interval 轮重建一次（基于上一轮嵌入，首轮无）
            use_helpers = (r + 1) % self.h_interval == 0 and self.cid_to_vector
            helper_map = (
                {cid: self.get_helpers(cid) for cid in selected} if use_helpers else {}
            )

            # 3. 构造 worker 参数：通用 base + (σ, ψ) 覆盖模型位 + 专属参数追加
            p = self.build_base_params(selected)
            for params in p:
                params[2] = (self.sigma_state, self.psi_state)
                for extra in (
                    r,
                    helper_map.get(params[0]),
                    self.args.confidence,
                    self.args.psi_factor,
                    self.args.lambda_s,
                    self.args.lambda_i,
                    self.args.lambda_a,
                    self.args.lambda_l2,
                    self.args.lambda_l1,
                    self.args.l1_thres,
                ):
                    params.append(extra)
            results = self.run_clients(train, p)

            # 4. 汇总：缓存客户端 σ/ψ 与嵌入（供下一轮 helper 选取），累计损失
            sigma_list = []
            psi_list = []
            total_loss = 0.0
            self.cid_to_vector = {}
            self.cid_to_sigma_state = {}
            self.cid_to_psi_state = {}
            for cid, res in results.items():
                total_loss += res["loss"]
                sigma_list.append(res["sigma"])
                psi_list.append(res["psi"])
                self.cid_to_sigma_state[cid] = res["sigma"]
                self.cid_to_psi_state[cid] = res["psi"]
                self.cid_to_vector[cid] = self.embed_client(cid)

            # 5. 等权平均聚合全局 σ 与 ψ
            self.loss.append(total_loss / num_join)
            uniform_weights = [1 / len(sigma_list)] * len(sigma_list)
            self.sigma_state = param_aggregate(sigma_list, uniform_weights)
            self.psi_state = param_aggregate(psi_list, uniform_weights)

            # 6. 评估：ψ 硬阈值稀疏化后合并 σ+ψ 重建 θ
            merged = merge_state(
                self.sigma_state, self.psi_state, self.l1_thres, sparsify=True
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
        if self.acc_target:
            metrics["acc_target"] = self.acc_target
        if self.acc_source:
            metrics["acc_source"] = self.acc_source
        params = {
            "global": merge_state(self.sigma_state, self.psi_state, self.l1_thres),
            "sigma": self.sigma_state,
            "psi": self.psi_state,
        }
        self.deal_save(metrics, params)
