import os
import time

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from .utils import (
    BaseServer,
    ce_loss,
    fmt_num,
    get_model,
    mse_loss,
    proto_aggregate,
    strong_augment,
    weak_augment,
)


def get_path(args):
    """构造实验日志文件名（含域配置与算法超参值）。"""
    args.file_name = f"{args.common_name}_{fmt_num(args.mu)}"
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


# ═══════════════════════════════════════════════════════════════
# FixMatch 一致性损失（半监督核心）
#   有标签: weak_aug → CE(y_true)
#   无标签: weak_aug → pseudo_label（置信度过滤）
#           strong_aug → CE(pseudo)
# ═══════════════════════════════════════════════════════════════
def fixmatch_loss(
    model,
    x,
    y,
    is_labeled,
    n_classes,
    threshold,
    lam,
    global_proto_sup=None,
    global_proto_unsup=None,
    mu=0.0,
):
    # 原型校准损失分别与「有标签 / 无标签」样本对齐：
    #   有标签样本 → 全局监督原型 (global_proto_sup[y])
    #   高置信无标签样本 → 全局无监督原型 (global_proto_unsup[pseudo])
    loss = torch.tensor(0.0, device=x.device, requires_grad=True)

    if is_labeled.any():
        x_l = weak_augment(x[is_labeled])
        feat = model.extractor(x_l)
        logits_l = model.classifier(feat)
        loss = loss + ce_loss(logits_l, y[is_labeled])
        if global_proto_sup is not None:
            loss = loss + mu * mse_loss(feat, global_proto_sup[y[is_labeled]])

    unlabeled_mask = ~is_labeled
    if unlabeled_mask.any() and lam > 0:
        x_u = x[unlabeled_mask]
        x_w = weak_augment(x_u)
        x_s = strong_augment(x_u)

        with torch.no_grad():
            feat_w = model.extractor(x_w)
            logits_w = model.classifier(feat_w)
            probs = torch.softmax(logits_w[:, :n_classes], dim=1)
            max_probs, pseudo = torch.max(probs, dim=1)
            confident = max_probs >= threshold

        feat_s = model.extractor(x_s[confident])
        logits_s = model.classifier(feat_s)
        loss = loss + lam * ce_loss(logits_s[:, :n_classes], pseudo[confident])
        if global_proto_unsup is not None and confident.any():
            loss = loss + mu * mse_loss(feat_s, global_proto_unsup[pseudo[confident]])

    return loss


def extract_protos(model, loader, num_class, feature_dim, device, threshold, labeled):
    """按「有标签 / 无标签」分别提取本地原型与样本计数。

    - labeled=True : 使用样本真实标签 y。
    - labeled=False: 对无标签样本做弱增强取伪标签，仅保留高置信样本。
    返回 [num_class, feature_dim] 与 [num_class] 的 CPU 张量。
    """
    proto_sum = torch.zeros(num_class, feature_dim, device=device)
    counts = torch.zeros(num_class, device=device)

    model.eval()
    with torch.no_grad():
        for x, y, _, is_labeled in loader:
            x, y, is_labeled = x.to(device), y.to(device), is_labeled.to(device)

            mask = is_labeled if labeled else ~is_labeled
            if not mask.any():
                continue
            x_sel = x[mask]
            y_sel = y[mask]

            if labeled:
                targets = y_sel
                feat = model.extractor(weak_augment(x_sel))
            else:
                feat_w = model.extractor(weak_augment(x_sel))
                logits_w = model.classifier(feat_w)
                probs = torch.softmax(logits_w[:, :num_class], dim=1)
                max_probs, pseudo = torch.max(probs, dim=1)
                conf = max_probs >= threshold
                if not conf.any():
                    continue
                targets = pseudo[conf]
                feat = feat_w[conf]

            proto_sum.index_add_(0, targets, feat)
            counts += torch.bincount(targets, minlength=num_class)

    active = counts > 0
    safe = torch.where(active, counts, torch.ones_like(counts))
    protos = proto_sum / safe.unsqueeze(1)
    protos[~active] = 0.0
    return protos.cpu().detach().clone(), counts.cpu().detach().clone()


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
        unlabel_domain,
        confidence_threshold,
        lam,
        global_proto_sup,
        global_proto_unsup,
        mu,
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
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    total_loss = 0.0
    num_batches = 0
    model.train()
    for _ in range(epochs):
        for x, y, _, is_labeled in loader:
            x, y, is_labeled = x.to(device), y.to(device), is_labeled.to(device)
            loss = fixmatch_loss(
                model,
                x,
                y,
                is_labeled,
                num_class,
                confidence_threshold,
                lam,
                g_proto_sup,
                g_proto_unsup,
                mu,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / max(1, num_batches)

    # 本地原型提取：监督（真实标签）与无监督（高置信伪标签）分开计算。
    proto_sup, count_sup = extract_protos(
        model,
        loader,
        num_class,
        feature_dim,
        device,
        confidence_threshold,
        labeled=True,
    )
    proto_unsup, count_unsup = extract_protos(
        model,
        loader,
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
    }


class Server(BaseServer):
    def __init__(self, args):
        super().__init__(False, args)
        self.unlabel_domain = args.unlabel_domain
        self.confidence_threshold = args.confidence_threshold
        self.lam = args.lam
        self.mu = args.mu

        # 两组全局原型：分别由监督样本与无监督样本聚合得到，首轮为 None。
        self.global_proto_sup = None
        self.global_proto_unsup = None

        self.model = extend_classifier(self.model.cpu())

        if self.unlabel_domain is None:
            raise ValueError("fedtest requires unlabel_domain to be set")
        if not self.args.sfd:
            raise ValueError("fedtest requires SFD data (sfd must be enabled)")
        # SFD 下不同客户端可能只含单一域，故仅在全体客户端域集合的并集里校验，
        # 不强制每个客户端都包含 unlabel_domain。
        all_domains = set()
        for ds in self.train_sets.values():
            if ds.domain_map is not None:
                all_domains.update(ds.domain_map.keys())
        if self.unlabel_domain not in all_domains:
            raise ValueError(
                f"fedtest: unlabel_domain '{self.unlabel_domain}' not found in any "
                f"client data (domains: {sorted(all_domains)})"
            )

    def fit(self):
        num_join = max(1, int(self.num_clients * self.args.join_ratio))

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedTest Round {r + 1}/{self.rounds} ---")

            selected = torch.randperm(self.num_clients)[:num_join].tolist()
            print(f"Selected clients: {selected}")

            params_list = self.build_base_params(selected)
            for params in params_list:
                params.append(self.unlabel_domain)
                params.append(self.confidence_threshold)
                params.append(self.lam)
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
                params.append(self.mu)
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
        merged_test = ConcatDataset([source_eval, target_eval])
        acc_all = self.evaluate_model(merged_test)
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
