import math
import os
import time

import torch
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, Subset, TensorDataset

from .utils import (
    BaseServer,
    ce_loss,
    extract_protos_ss,
    fmt_num,
    get_model,
    mixup,
    mse_loss,
    proto_aggregate,
)


def get_path(args):
    """构造实验日志文件名（含域配置与算法超参值）。"""
    args.file_name = (
        f"{args.common_name}_{fmt_num(args.confidence_threshold)}"
        f"_{fmt_num(args.beta)}_{fmt_num(args.lambda_pl)}"
        f"_{fmt_num(args.lambda_mixup)}_{fmt_num(args.mixup_alpha)}"
        f"_{fmt_num(args.lambda_pa)}"
    )
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


def extend_classifier(model):
    old = model.classifier
    if not isinstance(old, torch.nn.Linear):
        raise TypeError(f"extend_classifier requires nn.Linear, got {type(old)}")
    n = old.out_features
    new = torch.nn.Linear(old.in_features, n * 2)
    with torch.no_grad():
        new.weight[:n] = old.weight
        new.bias[:n] = old.bias
    model.classifier = new
    return model


def make_2n(block, num_class, label=True):
    """构造 2n 分类器软目标：label=True -> [block, zeros]，否则 -> [zeros, block]。

    block: [B, num_class] 张量（one-hot 标签 / ema 软概率）。
    """
    n = block.shape[0]
    zeros = torch.zeros((n, num_class), dtype=block.dtype, device=block.device)
    if label:
        return torch.cat([block, zeros], dim=1)
    return torch.cat([zeros, block], dim=1)


def train(params):
    (
        _,
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
        confidence_threshold,
        global_proto_sup,
        global_proto_unsup,
        ema_probs,
        beta,
        lambda_pl,
        lambda_mixup,
        mixup_alpha,
        lambda_pa,
    ) = params

    model = get_model(model_name, dataset_name, num_class, feature_dim).to(device)
    extend_classifier(model)
    model.to(device)
    model.load_state_dict(model_state, strict=False)

    g_proto_sup = global_proto_sup.to(device) if global_proto_sup is not None else None
    g_proto_unsup = (
        global_proto_unsup.to(device) if global_proto_unsup is not None else None
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # is_labeled 已由数据层（partition/sfd.py 掩码 / ssl 掩码）正确写入，无需在此运行时重算。
    # 拆成 labeled / unlabeled 两个 DataLoader，zip 同步训练。
    x_l = train_set.x[train_set.is_labeled]
    y_l = train_set.y[train_set.is_labeled]
    x_u = train_set.x[~train_set.is_labeled]

    num_steps = round(len(x_l) / batch_size)
    bsize_u = math.ceil(len(x_u) / max(1, num_steps))

    l_loader = DataLoader(TensorDataset(x_l, y_l), batch_size=batch_size, shuffle=True)
    # 携带样本索引：EMA 按伪标签样本逐个维护，索引对应 x_u 中的行。
    u_loader = DataLoader(
        TensorDataset(torch.arange(len(x_u)), x_u),
        batch_size=bsize_u,
        shuffle=True,
    )

    # EMA 伪标签概率缓冲 [n_u, n_class]，跨轮持久；首轮 lazy 初始化。
    if ema_probs is None:
        ema_probs = torch.zeros(len(x_u), num_class, device=device)
    else:
        ema_probs = ema_probs.to(device)

    full_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    num_batches = 0
    model.train()
    for _ in range(epochs):
        # 每 epoch 重建 mixup 数据集：数量取有标签/无标签二者较大值，
        # 随机配对后一次性完成混合，内部循环直接取用。
        # 全部可能 mixup 组合为 n_l × n_u 种，编号为 0..n_l*n_u-1；
        # 不放回采样 n_mix = max(n_l, n_u) 个组合，解码得到 (perm_l, perm_u)。
        n_mix = max(len(x_l), len(x_u))
        pair_ids = torch.randperm(len(x_l) * len(x_u))[:n_mix]
        perm_l = pair_ids // len(x_u)
        perm_u = pair_ids % len(x_u)
        y_2n_l_all = make_2n(one_hot(y_l[perm_l], num_class).float(), num_class)
        y_2n_u_all = make_2n(ema_probs[perm_u].cpu(), num_class, label=False)
        x_mix_all, y1, y2, lam_m = mixup(
            x_u[perm_u], y_2n_u_all, x_l[perm_l], y_2n_l_all, alpha=mixup_alpha
        )
        y_mix_all = lam_m * y1 + (1 - lam_m) * y2
        m_loader = DataLoader(
            TensorDataset(x_mix_all, y_mix_all),
            batch_size=batch_size,
            shuffle=True,
        )

        for (x_lb, y_lb), (idx, x_ub), (x_mb, y_mb) in zip(
            l_loader, u_loader, m_loader
        ):
            x_lb, y_lb = x_lb.to(device), y_lb.to(device)
            idx, x_ub = idx.to(device), x_ub.to(device)
            x_mb, y_mb = x_mb.to(device), y_mb.to(device)

            # 2n 分类器：前 n 位为真实标签 one-hot，后 n 位补 0 向量
            y_2n = make_2n(one_hot(y_lb, num_class).float(), num_class)
            loss_cl = ce_loss(model(x_lb), y_2n)

            # 无监督：后 n 个 logits softmax → EMA 更新 → 阈值过滤
            logits_u = model(x_ub)
            probs_u = torch.softmax(logits_u[:, num_class:], dim=1).detach()

            with torch.no_grad():
                ema_probs[idx] = beta * ema_probs[idx] + (1 - beta) * probs_u
                max_p, pseudo = ema_probs[idx].max(dim=1)
                confident = max_p >= confidence_threshold

            loss_pl = torch.tensor(0.0, device=x_ub.device)
            if confident.any():
                soft_u = ema_probs[idx][confident]  # [B', C] 软概率
                soft_2n = make_2n(soft_u, num_class, label=False)
                logp_2n = torch.log_softmax(logits_u[confident], dim=1)
                loss_pl = -(soft_2n * logp_2n).sum(dim=1).mean()

            # 原型校准：同域 + 跨域。有标签样本特征对齐监督/无监督原型，
            # 高置信无标签样本特征对齐无监督/监督原型（跨域项以全局原型为常数锚点）。
            loss_pa = torch.tensor(0.0, device=x_ub.device)
            if g_proto_sup is not None:
                feat_l = model.extractor(x_lb)
                loss_pa = loss_pa + mse_loss(feat_l, g_proto_sup[y_lb])
                if g_proto_unsup is not None:
                    loss_pa = loss_pa + mse_loss(feat_l, g_proto_unsup[y_lb])
            if g_proto_unsup is not None and confident.any():
                feat_u = model.extractor(x_ub[confident])
                loss_pa = loss_pa + mse_loss(
                    feat_u, g_proto_unsup[pseudo[confident]]
                )
                if g_proto_sup is not None:
                    loss_pa = loss_pa + mse_loss(
                        feat_u, g_proto_sup[pseudo[confident]]
                    )

            # mixup 数据集已在 epoch 开头预生成，此处直接使用。
            prob_m = torch.softmax(model(x_mb), dim=1)
            loss_mixup = ((prob_m - y_mb) ** 2).mean()

            loss = (
                loss_cl
                + lambda_pl * loss_pl
                + lambda_mixup * loss_mixup
                + lambda_pa * loss_pa
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / max(1, num_batches)

    # 本地原型提取：监督（真实标签）与无监督（高置信伪标签）分开计算。
    proto_sup, count_sup = extract_protos_ss(
        model,
        full_loader,
        num_class,
        feature_dim,
        device,
        confidence_threshold,
        labeled=True,
    )
    proto_unsup, count_unsup = extract_protos_ss(
        model,
        full_loader,
        num_class,
        feature_dim,
        device,
        confidence_threshold,
        labeled=False,
    )

    model_state = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
    return {
        "loss": avg_loss,
        "state": model_state,
        "proto_sup": proto_sup,
        "count_sup": count_sup,
        "proto_unsup": proto_unsup,
        "count_unsup": count_unsup,
        "ema_probs": ema_probs.cpu().detach().clone(),
    }


class Server(BaseServer):
    def __init__(self, args):
        super().__init__(False, args)
        self.unlabel_domain = args.unlabel_domain
        self.confidence_threshold = args.confidence_threshold
        self.beta = args.beta

        # 两组全局原型：分别由监督样本与无监督样本聚合得到，首轮为 None。
        self.global_proto_sup = None
        self.global_proto_unsup = None

        # 逐客户端的 EMA 伪标签概率缓冲（[n_u, n_class]），跨轮持久维护，
        # 每轮传入 worker 更新后回收。
        self.ema_probs = {i: None for i in range(self.num_clients)}

        self.model = extend_classifier(self.model.cpu())

        if self.unlabel_domain is None:
            raise ValueError("fedtest requires unlabel_domain to be set")
        if not self.args.sfd:
            raise ValueError("fedtest requires SFD data (sfd must be enabled)")

    def fit(self):
        num_join = max(1, int(self.num_clients * self.args.join_ratio))

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedTest Round {r + 1}/{self.rounds} ---")

            selected = torch.randperm(self.num_clients)[:num_join].tolist()
            print(f"Selected clients: {selected}")

            params_list = self.build_base_params(selected)
            for params in params_list:
                params.append(self.confidence_threshold)
                params.append(
                    self.global_proto_sup.cpu()
                    if self.global_proto_sup is not None
                    else None
                )
                params.append(
                    self.global_proto_unsup.cpu()
                    if self.global_proto_unsup is not None
                    else None
                )
                params.append(self.ema_probs[params[0]])
                params.append(self.beta)
                params.append(self.args.lambda_pl)
                params.append(self.args.lambda_mixup)
                params.append(self.args.mixup_alpha)
                params.append(self.args.lambda_pa)
            results = self.run_clients(train, params_list)

            total_loss = 0.0
            selected_states = []
            current_weights = []
            selected_proto_sup = []
            selected_count_sup = []
            selected_proto_unsup = []
            selected_count_unsup = []
            for cid, res in results.items():
                total_loss += res["loss"]
                selected_states.append(res["state"])
                current_weights.append(self.weights[cid])
                selected_proto_sup.append(res["proto_sup"])
                selected_count_sup.append(res["count_sup"])
                selected_proto_unsup.append(res["proto_unsup"])
                selected_count_unsup.append(res["count_unsup"])
                self.ema_probs[cid] = res["ema_probs"]

            self.loss.append(total_loss / num_join)
            sum_weights = sum(current_weights)
            norm_weights = [w / sum_weights for w in current_weights]

            self.aggregate(selected_states, weights=norm_weights)

            # 分别聚合两组原型（按样本计数加权，缺失类保留旧全局原型）。
            self.global_proto_sup = proto_aggregate(
                selected_proto_sup,
                local_counts_list=selected_count_sup,
                old_global_protos=self.global_proto_sup,
            )
            self.global_proto_unsup = proto_aggregate(
                selected_proto_unsup,
                local_counts_list=selected_count_unsup,
                old_global_protos=self.global_proto_unsup,
            )

            self.evaluate()

            print(
                f"Global Accuracy: {self.acc[-1]:.2f}%, Avg Loss: {self.loss[-1]:.4f}"
            )
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def evaluate_model(self, test_set):
        # fedtest 全局模型为 2n 维：前 n_class 类为主分类头，后 n_class 类为扩展头。
        # 评估时将两部分 logits 对位相加融合，再 argmax 得到预测标签。
        loader = DataLoader(test_set, batch_size=128, shuffle=False)
        correct, total = 0.0, 0.0
        for x, y, *_ in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            logits = logits.reshape(-1, 2, self.num_class).sum(dim=1)
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
            total += y.size(0)
        return 100.0 * correct / total

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        self.model.to(self.device)

        # 双域场景：从全局测试集（合并各客户端 test）按一次域掩码切分——
        # 无标签域为 target，其余为 source。
        assert self.test_set.domains is not None
        mask_tgt = torch.tensor(
            [d == self.unlabel_domain for d in self.test_set.domains]
        )
        target_eval = Subset(self.test_set, torch.where(mask_tgt)[0].tolist())
        source_eval = Subset(self.test_set, torch.where(~mask_tgt)[0].tolist())

        acc_src = self.evaluate_model(source_eval)
        self.acc_source.append(acc_src)
        acc_tgt = self.evaluate_model(target_eval)
        self.acc_target.append(acc_tgt)
        acc_all = self.evaluate_model(self.test_set)
        self.acc.append(acc_all)

        self.model.cpu()

    def save(self):
        metrics = {"acc": self.acc, "loss": self.loss}
        if self.acc_target:
            metrics["acc_target"] = self.acc_target
        if self.acc_source:
            metrics["acc_source"] = self.acc_source
        params = {
            "global": self.model.state_dict(),
            "global_proto_sup": self.global_proto_sup,
            "global_proto_unsup": self.global_proto_unsup,
        }
        self.deal_save(metrics, params)
