import os
import time
from dataclasses import asdict, dataclass

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from .utils import (
    BaseParams,
    BaseServer,
    clone_cpu_state,
    dist_contrastive_loss,
    extract_prototypes,
    fmt_num,
    get_model,
    proto_aggregate,
)


def get_path(args):
    args.file_name = (
        f"{args.common_name}_{fmt_num(args.label_ratio)}"
        f"_{fmt_num(args.lambda_)}_{fmt_num(args.sharpen_T)}"
    )
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


@dataclass
class Params(BaseParams):
    label_ratio: float
    support_size: int
    unlabeled_query_size: int
    lambda_: float
    sharpen_T: float
    helper_protos: torch.Tensor | None


# --------------------------------------------------------------------------- #
# 论文公式对应的核心函数
# --------------------------------------------------------------------------- #
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
        [
            torch.softmax(-torch.cdist(emb_u, hp[j], p=2.0), dim=1)  # Eq.(6) j=i
            for j in range(H)
        ],
        dim=0,
    )  # [H, B, K]
    avg = per_helper.mean(dim=0)  # Eq.(7) 跨 helper 平均 → [B, K]
    probs = avg ** (1.0 / T)  # Eq.(5) 锐化降低熵
    return probs / probs.sum(dim=1, keepdim=True)


def sample_per_class(labeled_idx_by_class, support_size):
    """
    每类随机采样支持集 S，查询集 Q 取该类全部剩余有标签样本（论文 D_k \\ S_k，不封顶）。
    S 大小受保护：s = min(support_size, max(1, n // 2))，保证每类至少留一半给 Q。
    返回: sup_idx, que_idx, que_y（均为一维张量；空时返回空张量）
    """
    sup_idx, que_idx, que_y = [], [], []
    for k, idx in labeled_idx_by_class.items():
        n = len(idx)
        if n == 0:
            continue
        perm = torch.randperm(n)
        s = min(support_size, max(1, n // 2))
        q = n - s
        sup_idx.append(idx[perm[:s]])
        que_idx.append(idx[perm[s : s + q]])
        que_y.append(torch.full((q,), k, dtype=torch.long))
    if not sup_idx:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty.clone(), empty.clone()
    return (
        torch.cat(sup_idx),
        torch.cat(que_idx),
        torch.cat(que_y),
    )


# --------------------------------------------------------------------------- #
# 客户端本地训练（对应论文 RunClient）
# --------------------------------------------------------------------------- #
def train(p: Params):
    device = torch.device(p.client_gpu)

    # 1. 初始化模型（仅使用 extractor 作为特征提取器 f_θ，不使用 classifier）
    model = get_model(p.model_name, p.dataset, p.num_class, p.feature_dim).to(device)
    model.load_state_dict(p.model_state)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=p.lr,
        momentum=p.momentum,
        weight_decay=p.weight_decay,
    )

    # 2. 构造按类索引：有标签样本索引（按类分组）与无标签样本索引
    y = p.train_set.y
    labeled_mask = p.train_set.is_labeled
    labeled_idx = torch.where(labeled_mask)[0]
    unlabeled_idx = torch.where(~labeled_mask)[0]
    labeled_idx_by_class = {
        k: labeled_idx[y[labeled_idx] == k] for k in range(p.num_class)
    }

    # 全量 Dataset（零拷贝引用），供各子集 loader 复用
    full_ds = TensorDataset(p.train_set.x, p.train_set.y)

    total_loss = 0.0
    num_batches = 0
    for _ in range(p.epochs):
        # 每类采样支持集 S；查询集 Q 取全部剩余有标签样本（论文 D_k \ S_k）
        sup_idx, que_idx, que_y = sample_per_class(labeled_idx_by_class, p.support_size)

        # 本地原型（基于支持集, Eq.2）；extract_prototypes 内部 index_add_ 向量化累加
        C_local = extract_prototypes(
            model,
            DataLoader(Subset(full_ds, sup_idx), p.batch_size),
            p.num_class,
            p.feature_dim,
            device,
        ).to(device)  # extract_prototypes 返回 CPU，这里一次性搬回
        model.train()  # 其内部置 eval，须恢复训练态（否则破坏后续 BN/随机失活）

        # 监督项（Eq.8 第一项）：全部剩余查询集按 batch_size 分批
        # → 本地原型距离概率 → CE(真实标签)，每 batch 一次更新
        q_loader = DataLoader(
            TensorDataset(p.train_set.x[que_idx], que_y),
            batch_size=p.batch_size,
            shuffle=True,
        )
        for xq_b, yq_b in q_loader:
            f_q = model.extractor(xq_b.to(device))  # [B_q, D]
            # 本地原型距离概率 → CE(真实标签)（dist_contrastive_loss = Eq.(6)+CE）
            loss = dist_contrastive_loss(f_q, C_local, yq_b.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        # 无监督项（Eq.8 第二项）：无标签子集分批 → 伪标签 → 本地原型距离概率 → CE
        u_idx = unlabeled_idx[
            torch.randperm(len(unlabeled_idx))[: p.unlabeled_query_size]
        ]
        u_loader = DataLoader(
            TensorDataset(p.train_set.x[u_idx]),
            batch_size=p.batch_size,
            shuffle=True,
        )
        for (xu_b,) in u_loader:
            f_u = model.extractor(xu_b.to(device))  # [B_u, D]
            p_bar_u = pseudolabel(f_u, p.helper_protos, p.sharpen_T)  # [B_u, K] soft
            if p_bar_u is None:
                # 首轮 H_r 为空（无辅助客户端原型），无监督项跳过
                continue
            # 本地原型距离概率 → CE(soft 伪标签)（dist_contrastive_loss = Eq.(6)+CE）
            loss_unsup = p.lambda_ * dist_contrastive_loss(f_u, C_local, p_bar_u)
            optimizer.zero_grad()
            loss_unsup.backward()
            optimizer.step()
            total_loss += loss_unsup.item()
            num_batches += 1

    avg_loss = total_loss / max(1, num_batches)

    # 3. 最终原型：用全量有标签数据 D_{i,k}^L（RunClient 步骤3, Eq.2 全量版）
    #    counts = 每类有标签样本数，由 extract_prototypes 的 bincount 直接给出
    final_protos, final_counts = extract_prototypes(
        model,
        DataLoader(Subset(full_ds, labeled_idx), batch_size=p.batch_size),
        p.num_class,
        p.feature_dim,
        device,
        return_counts=True,
    )

    # 4. 整理返回
    model_state = clone_cpu_state(model.state_dict())
    return {
        "loss": avg_loss,
        "state": model_state,
        "protos": final_protos,
        "counts": final_counts,
    }


# --------------------------------------------------------------------------- #
# 服务端（对应论文 RunServer）
# --------------------------------------------------------------------------- #
class Server(BaseServer):
    def __init__(self, args):
        # ProtoFSSL 为全局联邦方法，pfl=False
        super().__init__(args, is_ssl=True)
        self.label_ratio = args.label_ratio
        self.support_size = args.support_size
        self.unlabeled_query_size = args.unlabeled_query_size
        self.lambda_ = args.lambda_
        self.sharpen_T = args.sharpen_T

        self.client_protos = {}  # cid → 本地原型 [K, D]，供下一轮构造辅助集
        self.global_protos = None  # 聚合后的全局原型

    def fit(self):
        num_join = max(1, int(self.num_clients * self.join_ratio))

        prev_selected = []  # M_{r-1}
        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- ProtoFSSL Round {r + 1}/{self.rounds} ---")

            selected = torch.randperm(self.num_clients)[:num_join].tolist()
            print(f"Selected clients: {selected}")

            # 辅助集 H_r = M_{r-1}（首轮为空）
            H_r = prev_selected

            aux_list = [
                self.client_protos[j].cpu() for j in H_r if j in self.client_protos
            ]
            helper_protos = torch.stack(aux_list) if aux_list else None

            p = [
                Params(
                    **asdict(base),
                    label_ratio=self.label_ratio,
                    support_size=self.support_size,
                    unlabeled_query_size=self.unlabeled_query_size,
                    lambda_=self.lambda_,
                    sharpen_T=self.sharpen_T,
                    helper_protos=helper_protos,
                )
                for base in self.build_base_params(selected)
            ]

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
            if self.is_sfd:
                src = f"{self.acc_source[-1]:.2f}%" if self.acc_source else "N/A"
                tgt = f"{self.acc_target[-1]:.2f}%" if self.acc_target else "N/A"
                acc = f"{self.acc[-1]:.2f}%" if self.acc else "N/A"
                sp = f"{self.acc_source_p[-1]:.2f}%" if self.acc_source_p else "N/A"
                tp = f"{self.acc_target_p[-1]:.2f}%" if self.acc_target_p else "N/A"
                print(
                    f"Global Acc: {acc}, Source Acc: {src}, Target Acc: {tgt}, "
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
        if self.is_sfd:
            metrics["acc_source"] = self.acc_source
            metrics["acc_target"] = self.acc_target
            metrics["acc_source_p"] = self.acc_source_p
            metrics["acc_target_p"] = self.acc_target_p
        params = {"global": self.model.state_dict(), "proto": self.global_protos}
        self.deal_save(metrics, params)
