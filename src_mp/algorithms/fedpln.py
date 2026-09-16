from typing import cast

import torch

from .core import (
    BaseClientExecutor,
    BaseServer,
    ClientResult,
    EvalResult,
    EvalTask,
    aggregate_weighted,
    ce_loss,
    clone_state,
    dist_contrastive_loss,
)


class PLN(torch.nn.Module):
    def __init__(self, num_classes, width, feature_dim, depth=1, fixed=0, init_emb=0):
        super().__init__()
        self.embedings = torch.nn.Embedding(num_classes, width)
        self._init_embedings(init_emb)
        if fixed:
            self.embedings.weight.requires_grad = False
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self.middle = torch.nn.Sequential(
            *[
                torch.nn.Sequential(torch.nn.Linear(width, width), torch.nn.ReLU())
                for _ in range(depth)
            ]
        )
        self.fc = torch.nn.Linear(width, feature_dim)

    def _init_embedings(self, init_emb):
        initializers = {
            1: lambda: torch.nn.init.uniform_(self.embedings.weight, -0.1, 0.1),
            2: lambda: torch.nn.init.normal_(self.embedings.weight, mean=0.0, std=0.1),
            3: lambda: torch.nn.init.normal_(self.embedings.weight, mean=0.0, std=0.01),
            4: lambda: torch.nn.init.xavier_uniform_(self.embedings.weight),
            5: lambda: torch.nn.init.xavier_normal_(self.embedings.weight),
            6: lambda: torch.nn.init.kaiming_uniform_(
                self.embedings.weight, nonlinearity="linear"
            ),
            7: lambda: torch.nn.init.orthogonal_(self.embedings.weight),
        }
        if init_emb not in initializers and init_emb != 0:
            raise ValueError("Unknown init_emb value")
        if init_emb in initializers:
            initializers[init_emb]()

    def forward(self, class_ids):
        return self.fc(self.middle(self.embedings(class_ids)))


class Client(BaseClientExecutor):
    def __init__(self, args, device, num_class, **kwargs):
        super().__init__(args, device, num_class, **kwargs)
        self.lam: float = args.lambda_
        self.epoch_pln: int = args.epoch_pln
        self.batch_size_pln: int = args.batch_size_pln

        self.pln = PLN(
            num_class,
            args.width_pln,
            args.feature_dim,
            args.depth_pln,
            args.fixed_proto,
            args.init_emb,
        ).to(device)
        self.pln_optimizer = torch.optim.SGD(
            self.pln.parameters(),
            lr=args.lr_pln,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        self.all_classes = torch.arange(num_class, device=device)
        self.pln_loaders = {}

    def train(self):
        # model train
        self.pln.load_state_dict(self.payload)
        self.model.train()
        self.pln.eval()
        with torch.no_grad():
            self.prototypes: torch.Tensor = self.pln(self.all_classes)
        total, batches = 0.0, 0
        for _ in range(self.epochs):
            total, batches = self.run_epoch(total, batches)
        # PLN learning
        self.model.eval()
        self.pln.train()
        self.pln_optimizer.state.clear()
        total_proto, batches_proto = 0.0, 0
        for _ in range(self.epoch_pln):
            total_proto, batches_proto = self.train_pln(total_proto, batches_proto)

        return ClientResult(
            self.client_id,
            total / max(1, batches),
            clone_state(self.model.state_dict()),
            {
                "pln_state": clone_state(self.pln.state_dict()),
                "loss_proto": total_proto / max(1, batches_proto),
            },
        )

    def run_epoch(self, total: float, batches: int):
        for x, y, *_ in self.loader:
            x, y = x.to(self.device), y.to(self.device)
            feature = self.model.extractor(x)
            loss_ce = ce_loss(self.model.classifier(feature), y)
            loss_p = dist_contrastive_loss(feature, self.prototypes, y)
            loss = loss_ce + self.lam * loss_p
            self.check_nan(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total += loss.item()
            batches += 1
        return total, batches

    def train_pln(self, total, batches):
        for x, y, *_ in self._pln_loader():
            x, y = x.to(self.device), y.to(self.device)
            with torch.no_grad():
                feature = self.model.extractor(x)
            loss = dist_contrastive_loss(feature, self.pln(self.all_classes), y)
            self.check_nan(loss)
            self.pln_optimizer.zero_grad()
            loss.backward()
            self.pln_optimizer.step()
            total += loss.item()
            batches += 1
        return total, batches

    def _pln_loader(self):
        if self.batch_size_pln == self.batch_size:
            return self.loader
        if self.client_id not in self.pln_loaders:
            self.pln_loaders[self.client_id] = torch.utils.data.DataLoader(
                self.current_task.train_set,
                batch_size=self.batch_size_pln,
                shuffle=True,
            )
        return self.pln_loaders[self.client_id]

    def evaluate(self, task: EvalTask) -> EvalResult:
        if task.eval_id not in self.eval_loaders:
            self.eval_loaders[task.eval_id] = torch.utils.data.DataLoader(
                task.test_set, batch_size=128, shuffle=False
            )
        self.model.load_state_dict(task.model_state)
        self.model.eval()
        prototypes: torch.Tensor = task.payload.to(self.device)
        correct = proto_correct = total = 0
        with torch.no_grad():
            for x, y, *_ in self.eval_loaders[task.eval_id]:
                x, y = x.to(self.device), y.to(self.device)
                features = self.model.extractor(x)
                predictions = self.model.classifier(features).argmax(dim=1)
                distances = torch.cdist(features, prototypes)
                prototype_predictions = distances.argmin(dim=1)
                correct += predictions.eq(y).sum().item()
                proto_correct += prototype_predictions.eq(y).sum().item()
                total += y.size(0)
        return EvalResult(
            task.eval_id, correct, total, payload=100.0 * proto_correct / max(1, total)
        )


class Server(BaseServer):
    client_cls = Client

    def __init__(self, args, devices):
        super().__init__(args, devices)
        self.pln = PLN(
            self.num_class,
            args.width_pln,
            args.feature_dim,
            args.depth_pln,
            args.fixed_proto,
            args.init_emb,
        )
        self.all_classes = torch.arange(self.num_class)
        self.loss_proto, self.acc_proto = [], []

    def run_round(self):
        results = self.train_clients()
        self.apply_result(results)
        loss = sum(results[c].loss for c in self.selected) / self.num_selected
        loss_pln = (
            sum(results[c].payload["loss_proto"] for c in self.selected)
            / self.num_selected
        )
        accuracy, accuracy_pln = self.evaluate()
        self.record_round(loss, accuracy, loss_pln, accuracy_pln)

    def train_payloads(self):
        state = clone_state(self.pln.state_dict())
        return {client_id: state for client_id in self.selected}

    def evaluate(self):
        with torch.no_grad():
            prototypes = self.pln(self.all_classes).cpu()
        result = super().evaluate({-1: prototypes})
        return cast(tuple[float, float], result)

    def summarize_evaluation(self, results):
        acc = results[-1].accuracy
        acc_pln = results[-1].payload
        return acc, acc_pln

    def apply_result(self, res):
        total = sum(self.train_counts[c] for c in self.selected)
        weights = [self.train_counts[c] / total for c in self.selected]
        model_state = aggregate_weighted([res[c].state for c in self.selected], weights)
        self.model.load_state_dict(model_state)
        pln_states = [res[c].payload["pln_state"] for c in self.selected]
        pln_state = aggregate_weighted(pln_states, weights)
        self.pln.load_state_dict(pln_state)

    def record_round(self, loss, accuracy, loss_pln, accuracy_pln):
        super().record_round(loss, accuracy)
        self.loss_proto.append(loss_pln)
        self.acc_proto.append(accuracy_pln)

    def progress_metrics(self):
        metrics = super().progress_metrics()
        metrics.update(
            loss_pln=f"{self.loss_proto[-1]:.4f}",
            accuracy_pln=f"{self.acc_proto[-1]:.2f}%",
        )
        return metrics

    def round_log_metrics(self):
        metrics = super().round_log_metrics()
        metrics.update(
            loss_proto=f"{self.loss_proto[-1]:.6f}",
            accuracy_proto=f"{self.acc_proto[-1]:.2f}%",
        )
        return metrics
