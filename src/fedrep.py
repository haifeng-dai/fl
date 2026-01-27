import argparse
import copy
import time
import numpy as np, os
import torch
from .utils import (
    BaseServer,
    ce_loss,
    run_parallel_clients,
    get_model,
    param_aggregate,
    evaluate_model,
)


def add_args(parser: argparse.ArgumentParser):
    """
    Add FedRep specific arguments to the parser.
    """
    group = parser.add_argument_group("FedRep Specific Arguments")
    group.add_argument(
        "--epochs_head",
        type=int,
        default=5,
        help="Number of local epochs for Head update",
    )
    return parser


def client_worker(params):
    """
    Worker function for FedRep client training.

    Args:
        params: List containing:
            0: device (str/torch.device)
            1: global_body_state (dict) - State dict of the global representation (body)
            2: local_head_state (dict) - State dict of the local classifier (head)
            3: train_set (Dataset)
            4: model_name (str)
            5: dataset_name (str)
            6: lr (float) - Learning rate for Body
            7: batch_size (int)
            8: epochs_body (int) - Number of epochs for Body update (usually 1)
            9: epochs_head (int) - Number of epochs for Head update (usually > 1)

    Returns:
        List containing:
        0: loss (float) - Average loss during Body update
        1: new_body_state (dict) - Updated body parameters (on CPU)
        2: new_head_state (dict) - Updated head parameters (on CPU)
    """
    (
        device,
        global_body_state,
        local_head_state,
        train_set,
        model_name,
        dataset_name,
        lr,
        batch_size,
        epochs_body,
        epochs_head,
    ) = params

    # 1. Instantiate Model
    model = get_model(model_name, dataset_name).to(device)

    # 2. Load Parameters
    # Load shared body
    model.load_state_dict(global_body_state, strict=False)
    # Load personalized head (if available)
    if local_head_state is not None:
        model.classifier.load_state_dict(local_head_state)

    # 3. Define Optimizer
    # Note: FedRep uses different optimization phases
    # Phase 1: Update Head (Body frozen)
    # Phase 2: Update Body (Head frozen)

    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # --- Phase 1: Train Head ---
    # Freeze Body
    for param in model.extractor.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    # Optimizer for Head
    opt_head = torch.optim.SGD(model.classifier.parameters(), lr=lr)

    model.train()
    for _ in range(epochs_head):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt_head.zero_grad()
            output, _ = model(x)
            loss = ce_loss(output, y)
            loss.backward()
            opt_head.step()

    # --- Phase 2: Train Body ---
    # Freeze Head
    for param in model.classifier.parameters():
        param.requires_grad = False
    for param in model.extractor.parameters():
        param.requires_grad = True

    # Optimizer for Body
    # Note: FedRep usually uses the same LR for both, or separate. Here we use global lr.
    opt_body = torch.optim.SGD(model.extractor.parameters(), lr=lr)

    total_loss = 0.0
    num_batches = 0

    for _ in range(epochs_body):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt_body.zero_grad()
            output, _ = model(x)
            loss = ce_loss(output, y)
            loss.backward()
            opt_body.step()

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # 4. Prepare Return Values (Move to CPU to save GPU memory)
    # We only need to return the body for aggregation
    body_state = {k: v.cpu() for k, v in model.extractor.state_dict().items()}
    # We return the head to save it in the server's state for next round
    head_state = {k: v.cpu() for k, v in model.classifier.state_dict().items()}

    return [avg_loss, body_state, head_state]


class Server(BaseServer):
    def __init__(self, args: argparse.Namespace):
        # Initialize BaseServer
        # Note: We pass personalized_flag=True because FedRep is a personalized method
        # (It maintains local heads)
        super().__init__(True, args)

        # Initialize client-specific heads
        # self.client_heads will store the state_dict of the classifier head for each client
        self.client_heads = [None] * self.num_clients

        # Initialize heads for all clients (optional, can be done lazily)
        # But doing it here ensures consistent initialization if needed
        dummy_model = get_model(args.model, args.dataset)
        initial_head_state = dummy_model.classifier.state_dict()
        for i in range(self.num_clients):
            self.client_heads[i] = copy.deepcopy(initial_head_state)

    def fit(self):
        # Calculate number of clients to participate in each round
        num_join_clients = int(self.num_clients * self.args.join_ratio)
        num_join_clients = max(1, num_join_clients)

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedRep Round {r + 1}/{self.rounds} ---")

            # Select clients
            selected_clients = np.random.choice(
                self.num_clients, num_join_clients, replace=False
            )
            print(f"Selected clients: {selected_clients}")

            # Prepare global body state (shared)
            # We extract only the extractor (body) part of the global model
            global_body_state = self.model.extractor.state_dict()
            # Ensure it's on CPU for pickling/transmission
            global_body_state = {k: v.cpu() for k, v in global_body_state.items()}

            # Prepare parameters for each client
            p = []
            for i in selected_clients:
                p.append(
                    [
                        self.client_gpu[i],  # device
                        global_body_state,  # shared body
                        self.client_heads[i],  # local head
                        self.train_sets[i],  # dataset
                        self.args.model,
                        self.args.dataset,
                        self.args.lr,
                        self.args.batch_size,
                        self.args.epochs,  # epochs for body (global arg)
                        self.args.epochs_head,  # epochs for head (FedRep arg)
                    ]
                )

            # Run parallel training
            results = run_parallel_clients(
                client_worker=client_worker,
                num_clients=num_join_clients,
                parameters=p,
                gpu_pools=self.gpu_pools,
                mp=self.mp,
            )

            # Process results
            total_loss = 0.0
            new_body_states = []

            for i, res in enumerate(results):
                loss, body_state, head_state = res
                total_loss += loss

                client_idx = selected_clients[i]

                # Update local head state (persist on server)
                self.client_heads[client_idx] = head_state

                # Collect body state for aggregation
                new_body_states.append(body_state)

            self.loss.append(total_loss / num_join_clients)

            # Aggregate Body (Representation)
            # Calculate weights based on dataset size
            current_weights = [self.weights[i] for i in selected_clients]
            sum_weights = sum(current_weights)
            norm_weights = [w / sum_weights for w in current_weights]

            # Aggregate and update global model's body
            aggregated_body = param_aggregate(new_body_states, weights=norm_weights)
            self.model.extractor.load_state_dict(aggregated_body)

            # Evaluation
            self.evaluate()
            print(f"Acc: {self.acc[-1]:.4f}")
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def evaluate(self):
        """
        Evaluate the model using local heads and the global body.
        """
        current_acc = []

        # For evaluation, we need to combine the global body with each client's local head

        for i in range(self.num_clients):
            # 1. Load Global Body
            self.model.extractor.load_state_dict(self.model.extractor.state_dict())

            # 2. Load Local Head
            if self.client_heads[i] is not None:
                self.model.classifier.load_state_dict(self.client_heads[i])

            # 3. Evaluate
            acc = evaluate_model(self.model, self.test_set[i], device=self.device)
            current_acc.append(acc)

        self.acc.append(sum(current_acc) / len(current_acc))

    def save(self):
        f = {
            "acc": self.acc,
            "loss": self.loss,
            "state_dict": {
                "global_body": self.model.extractor.state_dict(),
                "client_heads": self.client_heads,
            },
        }
        super().deal_save(f)

    def get_log_path(self):
        self.file_name = f"{self.save_name_pre}_{self.args.dataset}_{self.args.model}_{self.args.epochs}_{self.args.epochs_head}"
        return os.path.join(self.log_path, f"{self.file_name}.log")
