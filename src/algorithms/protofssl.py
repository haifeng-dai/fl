import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .utils import BaseServer, fmt_num, get_model, proto_aggregate


def get_path(args):
    args.file_name = (
        f"{args.common_name}_lr{fmt_num(args.label_ratio)}"
        f"_lam{fmt_num(args.lambda_)}_T{fmt_num(args.sharpen_T)}"
    )
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


# --------------------------------------------------------------------------- #
# 论文公式对应的核心函数
# --------------------------------------------------------------------------- #
def proto_logits(emb, protos, temperature=1.0):
    """
    Eq.(6): 基于欧氏距离的负指数得到类概率 logits。
    emb:   [B, D]   嵌入向量
    protos:[K, D]   原型
    返回:  [B, K]   即 -dist(emb, protos) / temperature
    PyTorch 的 F.cross_entropy 直接接受该 logits（支持 hard/soft 目标）。
    """
    dist = torch.cdist(emb, protos.to(emb.device), p=2.0)  # [B, K]
    return -dist / temperature


def sharpen(probs, T):
    """
    Eq.(5): 锐化概率分布以降低熵。
    probs: [B, K]  →  p̄_k = p_k^(1/T) / Σ_{k'} p_{k'}^(1/T)
    """
    probs = probs ** (1.0 / T)
    return probs / probs.sum(dim=1, keepdim=True)


def pseudolabel(emb_u, helper_protos, T):
    """
    Eq.(6)+Eq.(7)+Eq.(5): 用辅助客户端原型给无标签样本生成 soft 伪标签。
    helper_protos: [H, K, D] 或 None
    emb_u:         [B, D]
    返回:          [B, K] soft 伪标签 p̄_i(u)；helper_protos 为 None 时返回 None
    """
    if helper_protos is None:
        return None
    hp = helper_protos.to(emb_u.device)  # [H, K, D]
    H = hp.shape[0]
    per_helper = torch.stack(
        [F.softmax(proto_logits(emb_u, hp[j], 1.0), dim=1) for j in range(H)],
        dim=0,
    )  # [H, B, K]
    avg = per_helper.mean(dim=0)  # Eq.(7) 跨 helper 平均 → [B, K]
    return sharpen(avg, T)  # Eq.(5) 锐化 → p̄_i(u)


def _group_prototypes(
    model, x, idx, y, num_class, feature_dim, device, batch_size=None
):
    """
    Eq.(2): 按类求嵌入特征的均值构造原型 → [K, D]。
    某类无样本时返回零向量（后续聚合会被 mask 忽略）。
    batch_size: 非空时按 mini-batch 前向，避免大全集一次性进显存（OOM 防护）。
    """
    protos = torch.zeros(num_class, feature_dim, device=device)
    counts = torch.zeros(num_class, dtype=torch.long, device=device)
    if len(idx) == 0:
        return protos
    with torch.no_grad():
        if not batch_size or batch_size <= 0:
            feats = model.extractor(x[idx].to(device))  # [N, D]
            for k in range(num_class):
                mask_k = y == k
                if mask_k.any():
                    protos[k] = feats[mask_k].mean(0)
        else:
            n_total = len(idx)
            for start in range(0, n_total, batch_size):
                end = min(start + batch_size, n_total)
                chunk_idx = idx[start:end]
                feats = model.extractor(x[chunk_idx].to(device))
                chunk_y = y[start:end]
                for k in range(num_class):
                    mask_k = chunk_y == k
                    if mask_k.any():
                        protos[k] += feats[mask_k].sum(0)
                        counts[k] += int(mask_k.sum())
            for k in range(num_class):
                if counts[k] > 0:
                    protos[k] /= counts[k]
    return protos


def sample_per_class(labeled_idx_by_class, support_size):
    """
    每类随机采样支持集 S，查询集 Q 取该类全部剩余有标签样本（论文 D_k \\ S_k，不封顶）。
    S 大小受保护：s = min(support_size, max(1, n // 2))，保证每类至少留一半给 Q。
    返回: sup_idx, sup_y, que_idx, que_y（均为一维张量；空时返回空张量）
    """
    sup_idx, sup_y, que_idx, que_y = [], [], [], []
    for k, idx in labeled_idx_by_class.items():
        n = len(idx)
        if n == 0:
            continue
        perm = torch.randperm(n)
        s = min(support_size, max(1, n // 2))
        q = n - s
        sup_idx.append(idx[perm[:s]])
        sup_y.append(torch.full((s,), k, dtype=torch.long))
        que_idx.append(idx[perm[s : s + q]])
        que_y.append(torch.full((q,), k, dtype=torch.long))
    if not sup_idx:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty.clone(), empty.clone(), empty.clone()
    return (
        torch.cat(sup_idx),
        torch.cat(sup_y),
        torch.cat(que_idx),
        torch.cat(que_y),
    )


# --------------------------------------------------------------------------- #
# 客户端本地训练（对应论文 RunClient）
# --------------------------------------------------------------------------- #
def train(params):
    (
        cid,
        device,
        model_state,
        train_set,
        model_name,
        dataset_name,
        lr,
        batch_size,
        epochs,
        feature_dim,
        num_class,
        label_ratio,
        support_size,
        unlabeled_query_size,
        lambda_,
        sharpen_T,
        helper_protos,
    ) = params

    # 1. 初始化模型（仅使用 extractor 作为特征提取器 f_θ，不使用 classifier）
    model = get_model(model_name, dataset_name, num_class, feature_dim).to(device)
    model.load_state_dict(model_state)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # 2. 半监督分割：优先读取数据层已给定的 is_labeled；
    #    若数据未提供（标准划分默认全有标签），则回退到全局 label_ratio 生成。
    if train_set.is_labeled is None:
        g = torch.Generator().manual_seed(1000 + cid)
        rand = torch.rand(len(train_set.y), generator=g)
        train_set.is_labeled = rand < label_ratio  # 逐样本伯努利划分
    # 否则直接使用数据集自带的 is_labeled（自定义/数据层已落好的分割）

    # 3. 构造按类索引：有标签样本索引（按类分组）与无标签样本索引
    y = train_set.y
    labeled_mask = train_set.is_labeled
    labeled_idx = torch.where(labeled_mask)[0]
    unlabeled_idx = torch.where(~labeled_mask)[0]
    labeled_idx_by_class = {
        k: labeled_idx[y[labeled_idx] == k] for k in range(num_class)
    }

    total_loss = 0.0
    num_batches = 0
    model.train()
    for _ in range(epochs):
        # 每类采样支持集 S；查询集 Q 取全部剩余有标签样本（论文 D_k \ S_k）
        sup_idx, sup_y, que_idx, que_y = sample_per_class(
            labeled_idx_by_class, support_size
        )

        # 本地原型（基于支持集, Eq.2）；分批前向以防大全集 OOM
        C_local = _group_prototypes(
            model,
            train_set.x,
            sup_idx,
            sup_y,
            num_class,
            feature_dim,
            device,
            batch_size,
        )

        # 监督项（Eq.8 第一项）：全部剩余查询集按 batch_size 分批
        # → 本地原型距离概率 → CE(真实标签)，每 batch 一次更新
        if len(que_idx) > 0:
            q_loader = DataLoader(
                TensorDataset(train_set.x[que_idx], que_y),
                batch_size=batch_size,
                shuffle=True,
            )
            for xq_b, yq_b in q_loader:
                f_q = model.extractor(xq_b.to(device))  # [B_q, D]
                p_ii_x = proto_logits(f_q, C_local, 1.0)  # Eq.(6) j=i
                loss = F.cross_entropy(p_ii_x, yq_b.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

        # 无监督项（Eq.8 第二项）：无标签子集分批 → 伪标签 → 本地原型距离概率 → CE
        if helper_protos is not None and len(unlabeled_idx) > 0:
            u_idx = unlabeled_idx[
                torch.randperm(len(unlabeled_idx))[:unlabeled_query_size]
            ]
            u_loader = DataLoader(
                TensorDataset(train_set.x[u_idx]),
                batch_size=batch_size,
                shuffle=True,
            )
            for (xu_b,) in u_loader:
                f_u = model.extractor(xu_b.to(device))  # [B_u, D]
                p_bar_u = pseudolabel(f_u, helper_protos, sharpen_T)  # [B_u, K] soft
                p_ii_u = proto_logits(f_u, C_local, 1.0)  # Eq.(6) j=i
                loss_unsup = lambda_ * F.cross_entropy(p_ii_u, p_bar_u)
                optimizer.zero_grad()
                loss_unsup.backward()
                optimizer.step()
                total_loss += loss_unsup.item()
                num_batches += 1

    avg_loss = total_loss / max(1, num_batches)

    # 4. 最终原型：用全量有标签数据 D_{i,k}^L（RunClient 步骤3, Eq.2 全量版）
    final_protos = _group_prototypes(
        model,
        train_set.x,
        labeled_idx,
        y[labeled_idx],
        num_class,
        feature_dim,
        device,
        batch_size,
    )
    final_counts = torch.tensor(
        [len(labeled_idx_by_class.get(k, [])) for k in range(num_class)],
        dtype=torch.float32,
    )

    # 5. 整理返回（CPU 化以防 Ray 对象存储悬空引用）
    model_state = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
    return {
        "loss": avg_loss,
        "state": model_state,
        "protos": final_protos.cpu(),
        "counts": final_counts.cpu(),
    }


# --------------------------------------------------------------------------- #
# 服务端（对应论文 RunServer）
# --------------------------------------------------------------------------- #
class Server(BaseServer):
    def __init__(self, args):
        # ProtoFSSL 为全局联邦方法，pfl=False
        super().__init__(False, args)

        self.client_protos = {}  # cid → 本地原型 [K, D]，供下一轮构造辅助集
        self.global_protos = None  # 聚合后的全局原型

    def fit(self):
        num_join = max(1, int(self.num_clients * self.args.join_ratio))

        prev_selected = []  # M_{r-1}
        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- ProtoFSSL Round {r + 1}/{self.rounds} ---")

            selected = torch.randperm(self.num_clients)[:num_join].tolist()
            print(f"Selected clients: {selected}")

            # 辅助集 H_r = M_{r-1}（首轮为空）
            H_r = prev_selected

            p = self.build_base_params(selected)
            for params in p:
                for extra in (
                    self.args.label_ratio,
                    self.args.support_size,
                    self.args.unlabeled_query_size,
                    self.args.lambda_,
                    self.args.sharpen_T,
                ):
                    params.append(extra)
                # 构造辅助原型 [H, K, D]（首轮 H_r 为空 → None）
                aux_list = [
                    self.client_protos[j].cpu() for j in H_r if j in self.client_protos
                ]
                params.append(torch.stack(aux_list) if aux_list else None)

            results = self.run_clients(train, p)

            # 汇总客户端回传结果
            total_loss = 0.0
            states, weights, protos, counts = [], [], [], []
            for cid, res in results.items():
                total_loss += res["loss"]
                self.clients_state[cid] = res["state"]
                self.client_protos[cid] = res["protos"]  # 供下一轮 H_{r+1}
                states.append(res["state"])
                weights.append(self.weights[cid])
                protos.append(res["protos"])
                counts.append(res["counts"])
            self.loss.append(total_loss / num_join)

            # 模型参数加权平均聚合（Eq.1）
            sum_weights = sum(weights)
            norm_weights = [w / sum_weights for w in weights]
            self.aggregate(states, weights=norm_weights)

            # 原型按样本计数聚合（缺失类保留旧全局原型）
            self.global_protos = proto_aggregate(
                protos,
                local_counts_list=counts,
                old_global_protos=self.global_protos,
            )

            self.evaluate(protos=self.global_protos)
            if self.args.sfd:
                src = f"{self.acc_source[-1]:.2f}%" if self.acc_source else "N/A"
                tgt = f"{self.acc_target[-1]:.2f}%" if self.acc_target else "N/A"
                sp = f"{self.acc_source_p[-1]:.2f}%" if self.acc_source_p else "N/A"
                tp = f"{self.acc_target_p[-1]:.2f}%" if self.acc_target_p else "N/A"
                print(
                    f"Source Acc: {src}, Target Acc: {tgt}, "
                    f"Source Proto: {sp}, Target Proto: {tp}, "
                    f"Avg Loss: {self.loss[-1]:.4f}"
                )
            else:
                acc = f"{self.acc[-1]:.2f}%" if self.acc else "N/A"
                pa = f"{self.acc_proto[-1]:.2f}%" if self.acc_proto else "N/A"
                print(f"Acc: {acc}, Proto Acc: {pa}, Avg Loss: {self.loss[-1]:.4f}")
            print(f"Round finished in {time.time() - t0:.2f} seconds")

            prev_selected = selected

    def save(self):
        """保存全局模型与全局原型"""
        metrics = {
            "acc": self.acc,
            "acc_proto": self.acc_proto,
            "loss": self.loss,
        }
        if self.args.sfd:
            metrics["acc_source"] = self.acc_source
            metrics["acc_target"] = self.acc_target
            metrics["acc_source_p"] = self.acc_source_p
            metrics["acc_target_p"] = self.acc_target_p
        params = {"global": self.model.state_dict(), "proto": self.global_protos}
        self.deal_save(metrics, params)
