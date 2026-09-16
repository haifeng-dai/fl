from __future__ import annotations

import gc
import queue
import traceback
from typing import Any

import torch
import torch.multiprocessing as mp

from .protocol import BaseClientParams, ClientResult, EvalResult, EvalTask


def _worker(
    worker_id: int,
    device_name: str,
    args,
    num_class: int,
    client_cls,
    inbox: Any,
    outbox: Any,
    train_sets: dict | None = None,
) -> None:
    try:
        torch.set_num_threads(2)
        device = torch.device(device_name)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        client = client_cls(args, device, num_class, train_sets=train_sets)
    except Exception:
        outbox.put((worker_id, None, None, traceback.format_exc()))
        return
    while (task := inbox.get()) is not None:
        try:
            result = _run_task(client, task)
            result_id = task.eval_id if isinstance(task, EvalTask) else task.client_id
            outbox.put((worker_id, result_id, result, None))
        except Exception:
            outbox.put((worker_id, None, None, traceback.format_exc()))
        finally:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


def _run_task(client, task):
    if isinstance(task, EvalTask):
        return client.evaluate(task)
    return client.run(task)


class PersistentClientPool:
    def __init__(
        self,
        devices: list[str],
        args,
        num_class: int,
        client_cls,
        train_sets: dict | None = None,
    ):
        self.ctx = mp.get_context("spawn")
        self.inboxes = [self.ctx.Queue() for _ in devices]
        self.outbox = self.ctx.Queue()
        self.processes = [
            self.ctx.Process(
                target=_worker,
                args=(
                    i,
                    device,
                    args,
                    num_class,
                    client_cls,
                    self.inboxes[i],
                    self.outbox,
                    train_sets,
                ),
            )
            for i, device in enumerate(devices)
        ]
        for process in self.processes:
            process.start()

    def run(self, tasks: list[BaseClientParams]) -> dict[int, ClientResult]:
        return self._run(tasks)

    def evaluate(self, tasks: list[EvalTask]) -> dict[int, EvalResult]:
        return self._run(tasks)

    def _run(self, tasks):
        results, next_task, active = {}, 0, 0
        for worker_id in range(min(len(self.processes), len(tasks))):
            self.inboxes[worker_id].put(tasks[next_task])
            next_task += 1
            active += 1
        while active:
            try:
                worker_id, result_id, result, error = self.outbox.get(timeout=5)
            except queue.Empty:
                failed = [
                    (worker_id, process.exitcode)
                    for worker_id, process in enumerate(self.processes)
                    if not process.is_alive()
                ]
                if failed:
                    raise RuntimeError(
                        f"torch_mp workers exited unexpectedly: {failed}"
                    )
                continue
            active -= 1
            if error:
                raise RuntimeError(f"torch_mp worker {worker_id} failed:\n{error}")
            results[result_id] = result
            if next_task < len(tasks):
                self.inboxes[worker_id].put(tasks[next_task])
                next_task += 1
                active += 1
        return results

    def close(self):
        for inbox in self.inboxes:
            inbox.put(None)
        for process in self.processes:
            process.join()
        for message_queue in [*self.inboxes, self.outbox]:
            message_queue.close()
