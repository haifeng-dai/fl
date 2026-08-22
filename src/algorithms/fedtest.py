import math
import os
import time

import torch
import torch.nn.functional as F
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, Subset, TensorDataset

from .utils import (
    BaseServer,
    ce_loss,
    evaluate_prototype,
    fmt_num,
    get_model,
    mixup,
    proto_aggregate,
)


def get_path(args):
    args.file_name = (
        f"{args.common_name}_{fmt_num(args.confidence)}"
        f"_{fmt_num(args.beta)}_{fmt_num(args.lambda_pl)}"
        f"_{fmt_num(args.lambda_pa)}_{fmt_num(args.lambda_mixup)}"
        f"_{fmt_num(args.mixup_alpha)}_{fmt_num(args.warmup_rounds)}"
    )
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


def extend_classifier(model):
    """Expand C -> 2C exactly as Uni-HSSL: first C copied, last C random."""
    old = model.classifier
    if not isinstance(old, torch.nn.Linear):
        raise TypeError(f"extend_classifier requires nn.Linear, got {type(old)}")

    n = old.out_features
    new = torch.nn.Linear(old.in_features, 2 * n)
    with torch.no_grad():
        # Labeled-domain classes: initialized from the supervised C-class model.
        new.weight[:n] = old.weight
        new.bias[:n] = old.bias
        # Unlabeled-domain classes: deliberately RANDOM; do not copy old weights.

    model.classifier = new
    return model


def make_2n(block, num_class, label=True):
    """Convert a C-dimensional target into the 2C fine-grained label space."""
    zeros = torch.zeros(
        (block.shape[0], num_class), dtype=block.dtype, device=block.device
    )
    if label:
        return torch.cat([block, zeros], dim=1)
    return torch.cat([zeros, block], dim=1)


def compute_local_prototypes(
    model,
    x_l,
    y_l,
    x_u,
    ema_probs,
    num_class,
    feature_dim,
    confidence,
    batch_size,
    device,
    detach=False,
):
    """Compute local source/target semantic prototypes.

    For training, prototypes remain in the autograd graph so L_pa can
    propagate through the encoder. For post-training monitoring, pass
    detach=True.
    """
    was_training = model.training
    model.eval()

    if detach:
        context = torch.no_grad()
    else:
        context = torch.enable_grad()

    with context:
        sum_l = None
        cnt_l = torch.zeros(num_class, device=device)

        l_loader = DataLoader(
            TensorDataset(x_l, y_l), batch_size=batch_size, shuffle=False
        )
        for xb, yb in l_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            feat = model.extractor(xb)
            batch_sum = torch.zeros(num_class, feature_dim, device=device)
            batch_sum = batch_sum.index_add(0, yb, feat)
            sum_l = batch_sum if sum_l is None else sum_l + batch_sum
            cnt_l += torch.bincount(yb, minlength=num_class).to(device)

        if sum_l is None:
            sum_l = torch.zeros(
                num_class, feature_dim, device=device, requires_grad=not detach
            )

        sum_u = None
        cnt_u = torch.zeros(num_class, device=device)

        if ema_probs is not None and len(x_u) > 0:
            max_p, pseudo_2n = ema_probs.max(dim=1)
            valid = (max_p >= confidence) & (pseudo_2n >= num_class)

            if valid.any():
                u_loader = DataLoader(
                    TensorDataset(torch.arange(len(x_u)), x_u),
                    batch_size=batch_size,
                    shuffle=False,
                )
                for idx_b, xb in u_loader:
                    idx_b = idx_b.to(device)
                    xb = xb.to(device)
                    local_valid = valid[idx_b]
                    if not local_valid.any():
                        continue

                    feat = model.extractor(xb)
                    idx_valid = idx_b[local_valid]
                    feat_valid = feat[local_valid]
                    pseudo_n = pseudo_2n[idx_valid] - num_class

                    batch_sum = torch.zeros(num_class, feature_dim, device=device)
                    batch_sum = batch_sum.index_add(0, pseudo_n, feat_valid)
                    sum_u = batch_sum if sum_u is None else sum_u + batch_sum
                    cnt_u += torch.bincount(pseudo_n, minlength=num_class).to(device)

        if sum_u is None:
            sum_u = torch.zeros(
                num_class, feature_dim, device=device, requires_grad=not detach
            )

        def normalize(s, c):
            active = c > 0
            safe = torch.where(active, c, torch.ones_like(c))
            p = s / safe.unsqueeze(1)
            p = torch.where(active.unsqueeze(1), p, torch.zeros_like(p))
            return p, active

        proto_l, active_l = normalize(sum_l, cnt_l)
        proto_u, active_u = normalize(sum_u, cnt_u)

    if was_training:
        model.train()
    return proto_l, cnt_l, active_l, proto_u, cnt_u, active_u


def prototype_alignment_loss(proto_l, active_l, proto_u, active_u, tau):
    """Eq. (9): symmetric cross-domain prototype-to-prototype InfoNCE."""
    valid = active_l & active_u
    if valid.sum().item() < 1:
        return torch.tensor(0.0, device=proto_l.device)

    pl = F.normalize(proto_l[valid], p=2, dim=1)
    pu = F.normalize(proto_u[valid], p=2, dim=1)
    sim = torch.matmul(pl, pu.T) / tau
    labels = torch.arange(sim.shape[0], device=sim.device)

    loss_l2u = F.cross_entropy(sim, labels)
    loss_u2l = F.cross_entropy(sim.T, labels)
    return 0.5 * (loss_l2u + loss_u2l)


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
        confidence,
        ema_probs,
        beta,
        lambda_pl,
        lambda_pa,
        lambda_mixup,
        mixup_alpha,
        tau,
        momentum,
        weight_decay,
        is_warmup,
        semi_progress_start,
        semi_progress_total,
    ) = params

    model = get_model(model_name, dataset_name, num_class, feature_dim)
    if not is_warmup:
        # Local model is instantiated as C-class, then expanded to match server's 2C state.
        extend_classifier(model)
    model.load_state_dict(model_state, strict=True)
    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    x_l = train_set.x[train_set.is_labeled]
    y_l = train_set.y[train_set.is_labeled]
    x_u = train_set.x[~train_set.is_labeled]

    num_steps = max(1, math.ceil(len(x_l) / batch_size))
    bsize_u = max(1, math.ceil(len(x_u) / num_steps))

    l_loader = DataLoader(TensorDataset(x_l, y_l), batch_size=batch_size, shuffle=True)
    u_loader = DataLoader(
        TensorDataset(torch.arange(len(x_u)), x_u),
        batch_size=bsize_u,
        shuffle=True,
    )

    if ema_probs is not None:
        ema_probs = ema_probs.to(device)

    total_loss = 0.0
    total_loss_cl = 0.0
    total_loss_pl = 0.0
    total_loss_pa = 0.0
    total_loss_mixup = 0.0
    num_batches = 0
    model.train()

    if is_warmup:
        # Federated supervised pre-training: C-class model on labeled data only.
        for _ in range(epochs):
            for x_lb, y_lb in l_loader:
                x_lb = x_lb.to(device)
                y_lb = y_lb.to(device)

                logits_l = model.classifier(model.extractor(x_lb))
                loss_cl = ce_loss(logits_l, y_lb)
                loss = loss_cl

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_loss_cl += loss_cl.item()
                num_batches += 1
    else:
        # ---------------------------------------------------------------
        # WMA initialization: paper Eq. (2)-(3)
        # p0_u = supervised C-class prediction
        # yhat0 = [0_C, p0_u]
        # ---------------------------------------------------------------
        if ema_probs is None:
            model.eval()
            with torch.no_grad():
                ema_probs = torch.zeros(len(x_u), 2 * num_class, device=device)
                for idx_b, xb in u_loader:
                    idx_b = idx_b.to(device)
                    xb = xb.to(device)
                    logits_u = model(xb)
                    prob_main = torch.softmax(logits_u[:, :num_class], dim=1)
                    ema_probs[idx_b] = make_2n(prob_main, num_class, label=False)
            model.train()

        for local_epoch in range(epochs):
            # Progressive mixup set for this local epoch.
            n_mix = max(len(x_l), len(x_u))
            m_loader = None
            if n_mix > 0 and len(x_l) > 0 and len(x_u) > 0:
                pair_space = len(x_l) * len(x_u)
                n_mix = min(n_mix, pair_space)
                pair_ids = torch.randperm(pair_space)[:n_mix]
                perm_l = pair_ids // len(x_u)
                perm_u = pair_ids % len(x_u)

                y_2n_l = make_2n(
                    one_hot(y_l[perm_l], num_class).float(),
                    num_class,
                    label=True,
                )
                y_2n_u = ema_probs[perm_u].detach().cpu()

                # ψ(t)=0.5+t/(2T), with t measured over the entire FL
                # semi-supervised training phase.
                frac_local = (local_epoch + 0.5) / max(1, epochs)
                progress = min(
                    1.0,
                    (semi_progress_start + frac_local) / max(1, semi_progress_total),
                )
                psi_t = 0.5 + 0.5 * progress

                x_mix, y_mix = mixup(
                    x_u[perm_u],
                    y_2n_u,
                    x_l[perm_l],
                    y_2n_l,
                    alpha=mixup_alpha,
                    psi_t=psi_t,
                )
                m_loader = DataLoader(
                    TensorDataset(x_mix, y_mix),
                    batch_size=batch_size,
                    shuffle=True,
                )

            if m_loader is None:
                iterator = zip(l_loader, u_loader)
            else:
                iterator = zip(l_loader, u_loader, m_loader)

            for batch_idx, batch in enumerate(iterator):
                # Paper uses cosine annealing during semi-supervised training.
                # In FL, map the local batch to the global semi-supervised progress.
                frac_batch = batch_idx / max(1, len(l_loader) - 1)
                progress = min(
                    1.0,
                    (semi_progress_start + (local_epoch + frac_batch) / epochs)
                    / max(1, semi_progress_total),
                )
                cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
                lr_cur = lr * cosine_factor
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_cur

                if len(batch) == 3:
                    (x_lb, y_lb), (idx, x_ub), (x_mb, y_mb) = batch
                else:
                    (x_lb, y_lb), (idx, x_ub) = batch
                    x_mb, y_mb = None, None

                x_lb = x_lb.to(device)
                y_lb = y_lb.to(device)
                idx = idx.to(device)
                x_ub = x_ub.to(device)

                # 1) Eq. (4): supervised 2C CE with [y_l, 0].
                feat_l = model.extractor(x_lb)
                logits_l = model.classifier(feat_l)
                y_2n_l_batch = make_2n(
                    one_hot(y_lb, num_class).float(),
                    num_class,
                    label=True,
                )
                loss_cl = ce_loss(logits_l, y_2n_l_batch)

                # 2) Eq. (5): WMA over FULL 2C class probabilities.
                feat_u = model.extractor(x_ub)
                logits_u = model.classifier(feat_u)
                prob_u = torch.softmax(logits_u, dim=1)
                ema_probs[idx] = beta * ema_probs[idx] + (1.0 - beta) * prob_u.detach()

                # 3) Eq. (6): full 2C pseudo-label CE on reliable instances.
                max_p, _ = ema_probs[idx].max(dim=1)
                conf_mask = max_p >= confidence
                if conf_mask.any():
                    loss_pl = ce_loss(
                        logits_u[conf_mask],
                        ema_probs[idx][conf_mask].detach(),
                    )
                else:
                    loss_pl = torch.tensor(0.0, device=device)

                # 4) Eq. (9): local cross-domain prototype alignment.
                #    Recompute local prototypes for the current model state.
                (
                    proto_l,
                    cnt_l,
                    active_l,
                    proto_u,
                    cnt_u,
                    active_u,
                ) = compute_local_prototypes(
                    model=model,
                    x_l=x_l,
                    y_l=y_l,
                    x_u=x_u,
                    ema_probs=ema_probs,
                    num_class=num_class,
                    feature_dim=feature_dim,
                    confidence=confidence,
                    batch_size=batch_size,
                    device=device,
                )
                loss_pa = prototype_alignment_loss(
                    proto_l, active_l, proto_u, active_u, tau
                )

                # 5) Eq. (12): progressive inter-domain mixup MSE.
                loss_mixup = torch.tensor(0.0, device=device)
                if x_mb is not None:
                    x_mb = x_mb.to(device)
                    y_mb = y_mb.to(device)
                    prob_m = torch.softmax(model(x_mb), dim=1)
                    loss_mixup = ((prob_m - y_mb) ** 2).mean()

                loss = (
                    loss_cl
                    + lambda_pl * loss_pl
                    + lambda_pa * loss_pa
                    + lambda_mixup * loss_mixup
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_loss_cl += loss_cl.item()
                total_loss_pl += loss_pl.item()
                total_loss_pa += loss_pa.item()
                total_loss_mixup += loss_mixup.item()
                num_batches += 1

    n_safe = max(1, num_batches)
    avg_loss = total_loss / n_safe
    avg_loss_cl = total_loss_cl / n_safe
    avg_loss_pl = total_loss_pl / n_safe
    avg_loss_pa = total_loss_pa / n_safe
    avg_loss_mixup = total_loss_mixup / n_safe

    # Local prototypes returned only for optional FL monitoring / aggregation.
    (
        proto_sup,
        count_sup,
        _active_sup,
        proto_unsup,
        count_unsup,
        _active_unsup,
    ) = compute_local_prototypes(
        model=model,
        x_l=x_l,
        y_l=y_l,
        x_u=x_u,
        ema_probs=ema_probs,
        detach=True,
        num_class=num_class,
        feature_dim=feature_dim,
        confidence=confidence,
        batch_size=batch_size,
        device=device,
    )

    model_state = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
    return {
        "loss": avg_loss,
        "loss_cl": avg_loss_cl,
        "loss_pl": avg_loss_pl,
        "loss_pa": avg_loss_pa,
        "loss_mixup": avg_loss_mixup,
        "state": model_state,
        "proto_sup": proto_sup.cpu().detach().clone(),
        "count_sup": count_sup.cpu().detach().clone(),
        "proto_unsup": proto_unsup.cpu().detach().clone(),
        "count_unsup": count_unsup.cpu().detach().clone(),
        "ema_probs": (
            ema_probs.cpu().detach().clone() if ema_probs is not None else None
        ),
    }


class Server(BaseServer):
    def __init__(self, args):
        super().__init__(False, args)
        self.unlabel_domain = args.unlabel_domain
        self.confidence = args.confidence
        self.beta = args.beta
        self.lambda_pl = args.lambda_pl
        self.lambda_pa = args.lambda_pa
        self.lambda_mixup = args.lambda_mixup
        self.mixup_alpha = args.mixup_alpha
        self.tau = getattr(args, "tau", 0.5)
        self.warmup_rounds = args.warmup_rounds
        self.momentum = getattr(args, "momentum", 0.9)
        self.weight_decay = getattr(args, "weight_decay", 1e-3)

        # Each client keeps its own WMA state because the pseudo-label memory
        # is attached to local unlabeled samples.
        self.client_pseudo_ema = {i: None for i in range(self.num_clients)}

        # Optional global prototype statistics for monitoring only.
        self.proto_sup = None
        self.proto_unsup = None
        self.acc_proto_sup = []
        self.acc_proto_sup_source = []
        self.acc_proto_sup_target = []
        self.acc_proto_unsup = []
        self.acc_proto_unsup_source = []
        self.acc_proto_unsup_target = []
        self.loss_cl = []
        self.loss_pl = []
        self.loss_pa = []
        self.loss_mixup = []

        self.domain_all = []
        self.domain_source = []
        self.domain_target = []
        assert self.test_set.domains is not None
        mask_tgt = torch.tensor(
            [d == self.unlabel_domain for d in self.test_set.domains]
        )
        self.target_eval = Subset(self.test_set, torch.where(mask_tgt)[0].tolist())
        self.source_eval = Subset(self.test_set, torch.where(~mask_tgt)[0].tolist())

        self.model = self.model.cpu()

        if self.unlabel_domain is None:
            raise ValueError("fedtest requires unlabel_domain to be set")
        if not self.is_ssl:
            raise ValueError("fedtest requires SFD data (ssl must be 'sfd')")

    def fit(self):
        num_join = max(1, int(self.num_clients * self.join_ratio))
        semi_total = max(1, self.rounds - self.warmup_rounds)

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedUniHSSL Round {r + 1}/{self.rounds} ---")

            # Federated supervised pre-training -> expand to 2C once.
            if r == self.warmup_rounds:
                print(
                    "-> Warmup finished, extending classifier to 2C "
                    "(first C copied, second C random)"
                )
                self.model = extend_classifier(self.model.cpu())

            selected = sorted(torch.randperm(self.num_clients)[:num_join].tolist())
            print(f"Selected clients: {selected}")

            params_list = self.build_base_params(selected)
            for params in params_list:
                cid = params[0]
                params.append(self.confidence)
                params.append(self.client_pseudo_ema[cid])
                params.append(self.beta)
                params.append(self.lambda_pl)
                params.append(self.lambda_pa)
                params.append(self.lambda_mixup)
                params.append(self.mixup_alpha)
                params.append(self.tau)
                params.append(self.momentum)
                params.append(self.weight_decay)
                params.append(r < self.warmup_rounds)
                params.append(max(0, r - self.warmup_rounds))
                params.append(semi_total)

            results = self.run_clients(train, params_list)

            total_loss = 0.0
            total_loss_cl = 0.0
            total_loss_pl = 0.0
            total_loss_pa = 0.0
            total_loss_mixup = 0.0
            selected_states = []
            current_weights = []
            selected_proto_sup = []
            selected_count_sup = []
            selected_proto_unsup = []
            selected_count_unsup = []

            for cid, res in results.items():
                total_loss += res["loss"]
                total_loss_cl += res.get("loss_cl", 0.0)
                total_loss_pl += res.get("loss_pl", 0.0)
                total_loss_pa += res.get("loss_pa", 0.0)
                total_loss_mixup += res.get("loss_mixup", 0.0)
                selected_states.append(res["state"])
                current_weights.append(self.weights[cid])
                selected_proto_sup.append(res["proto_sup"])
                selected_count_sup.append(res["count_sup"])
                selected_proto_unsup.append(res["proto_unsup"])
                selected_count_unsup.append(res["count_unsup"])
                self.client_pseudo_ema[cid] = res["ema_probs"]

            self.loss.append(total_loss / num_join)
            self.loss_cl.append(total_loss_cl / num_join)
            self.loss_pl.append(total_loss_pl / num_join)
            self.loss_pa.append(total_loss_pa / num_join)
            self.loss_mixup.append(total_loss_mixup / num_join)
            sum_weights = sum(current_weights)
            norm_weights = [w / sum_weights for w in current_weights]
            self.aggregate(selected_states, weights=norm_weights)

            # Aggregate prototypes for monitoring / optional later experiments.
            self.proto_sup = proto_aggregate(
                selected_proto_sup,
                local_counts_list=selected_count_sup,
                old_global_protos=self.proto_sup,
            )
            self.proto_unsup = proto_aggregate(
                selected_proto_unsup,
                local_counts_list=selected_count_unsup,
                old_global_protos=self.proto_unsup,
            )

            self.evaluate()

            print(
                f"[Losses]\n"
                f"  - Total:      {self.loss[-1]:.4f} "
                f"(Cl: {self.loss_cl[-1]:.4f}, PL: {self.loss_pl[-1]:.4f}, "
                f"PA: {self.loss_pa[-1]:.4f}, Mixup: {self.loss_mixup[-1]:.4f})"
            )
            print(
                f"[Accuracy]\n"
                f"  - Global:     {self.acc[-1]:.2f}%\n"
                f"  - Domains:    Source {self.acc_source[-1]:.2f}%, "
                f"Target {self.acc_target[-1]:.2f}%"
            )

            if self.acc_proto_sup:
                print(
                    "[Prototypes]\n"
                    f"  - ProtoSup:   {self.acc_proto_sup[-1]:.2f}% "
                    f"(Src: {self.acc_proto_sup_source[-1]:.2f}%, "
                    f"Tgt: {self.acc_proto_sup_target[-1]:.2f}%)"
                )
            if self.acc_proto_unsup:
                print(
                    f"  - ProtoUnsup: {self.acc_proto_unsup[-1]:.2f}% "
                    f"(Src: {self.acc_proto_unsup_source[-1]:.2f}%, "
                    f"Tgt: {self.acc_proto_unsup_target[-1]:.2f}%)"
                )

            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def evaluate_model(self, test_set):
        """Predict semantic class by argmax over the fine-grained 2C class.

        Domain-specific class IDs k and C+k map to the same semantic class k.
        Therefore prediction is argmax over 2C followed by modulo C.
        """
        loader = DataLoader(test_set, batch_size=128, shuffle=False)
        is_2n = getattr(self.model.classifier, "out_features", 0) == 2 * self.num_class
        correct, total = 0, 0

        for x, y, *_ in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            pred_2n = logits.argmax(dim=1)
            pred = pred_2n % self.num_class if is_2n else pred_2n
            correct += int((pred == y).sum().item())
            total += y.size(0)

        return 100.0 * correct / max(1, total)

    @torch.no_grad()
    def evaluate_domain(self, test_set, true_main):
        loader = DataLoader(test_set, batch_size=128, shuffle=False)
        correct, total = 0, 0
        for x, *_ in loader:
            x = x.to(self.device)
            pred = self.model(x).argmax(dim=1)
            if true_main:
                correct += int((pred < self.num_class).sum().item())
            else:
                correct += int((pred >= self.num_class).sum().item())
            total += x.size(0)
        return correct, total

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        self.model.to(self.device)

        acc_src = self.evaluate_model(self.source_eval)
        self.acc_source.append(acc_src)
        acc_tgt = self.evaluate_model(self.target_eval)
        self.acc_target.append(acc_tgt)
        acc_all = self.evaluate_model(self.test_set)
        self.acc.append(acc_all)

        if getattr(self.model.classifier, "out_features", 0) == 2 * self.num_class:
            src_c, src_t = self.evaluate_domain(self.source_eval, True)
            tgt_c, tgt_t = self.evaluate_domain(self.target_eval, False)
            self.domain_source.append((src_c, src_t))
            self.domain_target.append((tgt_c, tgt_t))
            self.domain_all.append((src_c + tgt_c, src_t + tgt_t))

        for tag, proto in (
            ("sup", self.proto_sup),
            ("unsup", self.proto_unsup),
        ):
            if proto is None:
                continue
            proto = proto.to(self.device)
            p_src = evaluate_prototype(self.model, proto, self.source_eval, self.device)
            p_tgt = evaluate_prototype(self.model, proto, self.target_eval, self.device)
            p_all = evaluate_prototype(self.model, proto, self.test_set, self.device)
            if tag == "sup":
                self.acc_proto_sup.append(p_all)
                self.acc_proto_sup_source.append(p_src)
                self.acc_proto_sup_target.append(p_tgt)
            else:
                self.acc_proto_unsup.append(p_all)
                self.acc_proto_unsup_source.append(p_src)
                self.acc_proto_unsup_target.append(p_tgt)

        self.model.cpu()

    def save(self):
        metrics = {"acc": self.acc, "loss": self.loss}
        if self.acc_target:
            metrics["acc_target"] = self.acc_target
        if self.acc_source:
            metrics["acc_source"] = self.acc_source

        for key, val in (
            ("loss_cl", self.loss_cl),
            ("loss_pl", self.loss_pl),
            ("loss_pa", self.loss_pa),
            ("loss_mixup", self.loss_mixup),
            ("acc_proto_sup", self.acc_proto_sup),
            ("acc_proto_sup_source", self.acc_proto_sup_source),
            ("acc_proto_sup_target", self.acc_proto_sup_target),
            ("acc_proto_unsup", self.acc_proto_unsup),
            ("acc_proto_unsup_source", self.acc_proto_unsup_source),
            ("acc_proto_unsup_target", self.acc_proto_unsup_target),
            ("domain_all", self.domain_all),
            ("domain_source", self.domain_source),
            ("domain_target", self.domain_target),
        ):
            if val:
                metrics[key] = val

        params = {
            "global": self.model.state_dict(),
            "proto_sup": self.proto_sup,
            "proto_unsup": self.proto_unsup,
        }
        self.deal_save(metrics, params)

    def deal_save(self, metrics, params):
        lines = []
        if self.acc_proto_sup:
            lines.append(
                f"  - ProtoSup:   {max(self.acc_proto_sup):.2f}% "
                f"(Src: {max(self.acc_proto_sup_source):.2f}%, "
                f"Tgt: {max(self.acc_proto_sup_target):.2f}%)"
            )
        if self.acc_proto_unsup:
            lines.append(
                f"  - ProtoUnsup: {max(self.acc_proto_unsup):.2f}% "
                f"(Src: {max(self.acc_proto_unsup_source):.2f}%, "
                f"Tgt: {max(self.acc_proto_unsup_target):.2f}%)"
            )
        if lines:
            print("\n[FedUniHSSL Summary]\n" + "\n".join(lines))
