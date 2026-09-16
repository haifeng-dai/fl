import abc
import logging
import pprint
import time

import torch
from tqdm import tqdm

from .data import load_federated_data
from .model import build_model
from .pool import PersistentClientPool
from .protocol import BaseClientParams, EvalTask, ModelParam
from .utils import aggregate_weighted, clone_state

logger = logging.getLogger(__name__)


class BaseServer(abc.ABC):
    client_cls = None
    personalized = False
    supports_ssl = False

    def __init__(self, args, devices):
        if self.client_cls is None:
            raise TypeError("algorithm must define client_cls")
        self.args = args
        self.is_ssl = args.ssl != "none"
        if self.is_ssl and not bool(type(self).supports_ssl):
            raise ValueError(
                f"{type(self).__name__} does not support configured ssl={args.ssl!r}"
            )
        self.devices = devices
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.rounds = args.rounds
        self.algo = args.algo
        self.train_sets, self.test_set, self.train_counts, self.num_class = (
            load_federated_data(
                args,
                pfl=self.personalized,
                normalize=not self.is_ssl,
            )
        )
        self.model_param = ModelParam(
            args.model,
            args.dataset,
            args.feature_dim,
            self.num_class,
        )
        self.model = build_model(self.model_param)
        self.device = torch.device(devices[0])
        self.loss, self.acc = [], []
        if self.personalized:
            state = clone_state(self.model.state_dict())
            self.client_states = {
                client_id: clone_state(state) for client_id in range(self.num_clients)
            }
        self.pool = PersistentClientPool(
            self.devices, args, self.num_class, self.client_cls
        )

        self.num_clients_per_round = max(1, int(args.num_clients * args.join_ratio))
        self.selected: list[int] = []

    def select_clients(self):
        self.selected = sorted(
            torch.randperm(self.num_clients)[: self.num_clients_per_round].tolist()
        )

    def task(self, client_id, state, payload=None):
        return BaseClientParams(
            client_id,
            state,
            self.train_sets[client_id],
            payload,
        )

    def build_train_tasks(self, payloads=None):
        state = clone_state(self.model.state_dict())
        return [
            self.task(
                client_id,
                self.client_states[client_id] if self.personalized else state,
                None if payloads is None else payloads.get(client_id),
            )
            for client_id in self.selected
        ]

    def train_payloads(self):
        return None

    def train_clients(self):
        tasks = self.build_train_tasks(self.train_payloads())
        return self.pool.run(tasks)

    def update_client_states(self, results, selected):
        for client_id in selected:
            self.client_states[client_id] = results[client_id].state

    def aggregate_model(self, results, weights: list[float] | None = None):
        if weights is None:
            total = sum(self.train_counts[c] for c in self.selected)
            weights = [self.train_counts[c] / total for c in self.selected]
        self.model.load_state_dict(
            aggregate_weighted([results[c].state for c in self.selected], weights)
        )

    def build_eval_tasks(self, payloads=None):
        if not self.personalized:
            return [
                EvalTask(
                    -1,
                    clone_state(self.model.state_dict()),
                    self.test_set,
                    None if payloads is None else payloads.get(-1),
                )
            ]
        return [
            EvalTask(
                client_id,
                clone_state(state),
                self.test_set[client_id],
                None if payloads is None else payloads.get(client_id),
            )
            for client_id, state in self.client_states.items()
        ]

    def summarize_evaluation(self, results):
        if self.personalized:
            return sum(result.accuracy for result in results.values()) / len(results)
        return results[-1].accuracy

    def evaluate(self, payloads=None):
        tasks = self.build_eval_tasks(payloads)
        results = self.pool.evaluate(tasks)
        return self.summarize_evaluation(results)

    def fit(self):
        params_text = pprint.pformat(vars(self.args), sort_dicts=True)
        logger.info(
            "params=%s",
            params_text,
        )
        progress = tqdm(
            total=self.rounds + 1,
            desc=self.algo.upper(),
            unit="round",
            dynamic_ncols=True,
        )
        started = time.perf_counter()
        completed_rounds = 0
        try:
            for round_index in range(self.rounds + 1):
                round_started = time.perf_counter()
                self.select_clients()
                self.run_round()
                round_elapsed = time.perf_counter() - round_started
                completed_rounds += 1
                progress.set_postfix(**self.progress_metrics())
                metrics = self.round_log_metrics()
                metric_text = " | ".join(
                    f"{name}={value}" for name, value in metrics.items()
                )
                logger.info(
                    "round=%d/%d | %s | round_time=%.3fs",
                    round_index + 1,
                    self.rounds + 1,
                    metric_text,
                    round_elapsed,
                )
                progress.update()
            elapsed = time.perf_counter() - started
            logger.info(
                "training_complete | rounds=%d | total_time=%.3fs | "
                "max_accuracy=%.2f%% | avg_round_time=%.3fs",
                completed_rounds,
                elapsed,
                max(self.acc) if self.acc else 0.0,
                elapsed / completed_rounds if completed_rounds else 0.0,
            )
        finally:
            progress.close()
            self.pool.close()

    def record_round(self, loss, accuracy):
        self.loss.append(loss)
        self.acc.append(accuracy)

    def progress_metrics(self):
        return {
            "loss": f"{self.loss[-1]:.4f}",
            "accuracy": f"{self.acc[-1]:.2f}%",
        }

    def round_log_metrics(self):
        return {
            "loss": f"{self.loss[-1]:.6f}",
            "accuracy": f"{self.acc[-1]:.2f}%",
        }

    def run_round(self):
        results = self.train_clients()
        self.apply_result(results)
        loss = sum(results[c].loss for c in self.selected) / len(self.selected)
        accuracy = self.evaluate()
        self.record_round(loss, accuracy)

    @abc.abstractmethod
    def apply_result(self, results): ...
