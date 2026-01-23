import copy
import argparse
import time
import torch

from .utils import (
    BaseServer,
    run_parallel_clients,
    ce_loss,
    evaluate_model,
    get_model,
    param_aggregate,
    kl_loss,
)


def add_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("FML Specific Arguments")
    group.add_argument(
        "--alpha_fml",
        type=float,
        default=1.0,
        help="Weight for KL Divergence Loss (Global to Local)",
    )
    group.add_argument(
        "--beta_fml",
        type=float,
        default=1.0,
        help="Weight for KL Divergence Loss (Local to Global)",
    )
    return parser


def client_worker(params):
    device = params[0]
    global_state = params[1]
    local_state = params[2]
    train_set = params[3]

    model_name = params[4]
    dataset_name = params[5]
    lr = params[6]
    batch_size = params[7]
    epochs = params[8]
    alpha = params[9]
    beta = params[10]

    # 1. Initialize Global Model (MEME)
    global_model = get_model(model_name, dataset_name).to(device)
    global_model.load_state_dict(global_state)

    # 2. Initialize Local Model (Personalized)
    local_model = get_model(model_name, dataset_name).to(device)
    local_model.load_state_dict(local_state)

    # Optimizers
    opt_g = torch.optim.SGD(global_model.parameters(), lr=lr)
    opt_l = torch.optim.SGD(local_model.parameters(), lr=lr)

    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    global_model.train()
    local_model.train()

    total_loss_g = 0.0
    total_loss_l = 0.0
    num_batches = 0

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # Forward pass
            out_g, _ = global_model(x)
            out_l, _ = local_model(x)

            # Cross Entropy Loss
            ce_g = ce_loss(out_g, y)
            ce_l = ce_loss(out_l, y)

            # Mutual Learning (KL Divergence)
            # KL(P || Q) -> P is target (detach), Q is input (log_softmax)

            # Loss for Global: CE + beta * KL(Local || Global)
            # We want Global to resemble Local
            loss_kl_g = kl_loss(out_g, out_l.detach())

            # Loss for Local: CE + alpha * KL(Global || Local)
            # We want Local to resemble Global
            loss_kl_l = kl_loss(out_l, out_g.detach())

            loss_g = ce_g + beta * loss_kl_g
            loss_l = ce_l + alpha * loss_kl_l

            # Update Global
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            # Update Local
            opt_l.zero_grad()
            loss_l.backward()
            opt_l.step()

            total_loss_g += loss_g.item()
            total_loss_l += loss_l.item()
            num_batches += 1

    avg_loss_g = total_loss_g / num_batches
    avg_loss_l = total_loss_l / num_batches

    # Return: client_id, [avg_loss, new_global_state, new_local_state]
    return [avg_loss_l, global_model.state_dict(), local_model.state_dict()]


class Server(BaseServer):
    def __init__(self, args: argparse.Namespace):
        model = get_model(args.model, args.dataset)
        # FML is Personalized FL (maintains local models)
        super().__init__(model, True, args)

        # Initialize local models for each client
        self.client_model_states = [
            copy.deepcopy(self.model.state_dict()) for _ in range(self.num_clients)
        ]

    def fit(self):
        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FML Round {r + 1}/{self.rounds} ---")

            # Global model state (on CPU)
            global_state = {k: v.cpu() for k, v in self.model.state_dict().items()}

            p = [
                [
                    self.client_gpu[i],
                    global_state,
                    self.client_model_states[i],  # Local state
                    self.train_sets[i],
                    self.args.model,
                    self.args.dataset,
                    self.args.lr,
                    self.args.batch_size,
                    self.args.epochs,
                    self.args.alpha_fml,
                    self.args.beta_fml,
                ]
                for i in range(self.num_clients)
            ]

            results = run_parallel_clients(
                client_worker=client_worker,
                num_clients=self.num_clients,
                parameters=p,
                gpu_pools=self.gpu_pools,
                mp=self.mp,
            )

            # results: [avg_loss_l, global_state, local_state]

            total_loss = 0.0
            global_states_to_agg = []

            for i, res in enumerate(results):
                loss_l = res[0]
                new_g_state = res[1]
                new_l_state = res[2]

                total_loss += loss_l
                global_states_to_agg.append(new_g_state)

                # Update stored local state (move to CPU)
                self.client_model_states[i] = {
                    k: v.cpu() for k, v in new_l_state.items()
                }

            self.loss.append(total_loss / self.num_clients)

            # Aggregate Global Models
            self.model.load_state_dict(
                param_aggregate(global_states_to_agg, self.weights)
            )

            self.evaluate()
            print(
                f"Global Accuracy: {self.acc[-1]:.2f}%, Avg Local Loss: {self.loss[-1]:.4f}"
            )
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def evaluate(self):
        # In FML, we usually evaluate the Personalized Local Models
        # But we can also evaluate the Global Model.
        # Standard: Evaluate Local Models on Local Test Sets

        accs = []
        for i in range(self.num_clients):
            self.model.load_state_dict(self.client_model_states[i])
            acc = evaluate_model(self.model, self.test_set[i], self.device)
            accs.append(acc)

        avg_acc = sum(accs) / len(accs)
        self.acc.append(avg_acc)

        # Restore global model for next round (though it's overwritten by aggregate anyway)
        # But good practice if evaluate used self.model
        # self.model.load_state_dict(...) -> done in fit loop next time

    def save(self, test):
        file_name: str = f"{self.args.epochs}_{self.args.batch_size}_{self.args.lr}.pt"
        # Save Global Model state
        # Optionally save local states if needed, but standard is global + metrics
        f = {"acc": self.acc, "loss": self.loss, "state_dict": self.model.state_dict()}
        super().deal_save(test, f, file_name)
