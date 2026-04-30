import flwr as fl
import logging
from typing import Dict, List, Tuple, Optional, Union
from flwr.common import Metrics, Parameters, FitIns, EvaluateIns, FitRes, EvaluateRes, NDArrays, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate
import os
from pathlib import Path
import torch
from model import UNet3D
import socket
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger("server")


class SaveModelStrategy(fl.server.strategy.FedAvg):
    """
    Custom Flower Strategy based on FedAvg that saves the aggregated global model
    after each round and collects/saves/plots training and evaluation metrics.
    """

    def __init__(
        self,
        *args,
        model_path: str = "./global_models", # Directory to save global models
        metrics_filename_base: str = "learning_curve", # Base name for metrics CSV and plot
        args_config=None, # To access command-line args like num_clients
        **kwargs
    ):
        """
        Initialises the strategy.

        Args:
            model_path (str): Directory path to save the global model checkpoints.
            metrics_filename_base (str): Base filename for saving metrics CSV and plot.
                                         Client count will be appended.
            args_config: Parsed command-line arguments object (optional).
            *args, **kwargs: Arguments passed to the parent FedAvg strategy.
        """
        self.model_save_path = Path(model_path)
        self.args_config = args_config
        # Extract num_clients for unique metric filenames if available
        self.num_clients_for_filename = self.args_config.num_clients if self.args_config else 'unknown'
        self.metrics_filename_base = f"{metrics_filename_base}_clients_{self.num_clients_for_filename}"

        super().__init__(*args, **kwargs)

        # Instantiate a model instance on CPU for loading the aggregated state_dict before saving
        # This avoids needing a GPU on the server side just for saving.
        self.model_for_saving = UNet3D(n_channels=4, n_classes=4).to(torch.device("cpu"))

        # Ensure the model saving directory exists
        self.model_save_path.mkdir(parents=True, exist_ok=True)

        # Initialise history to store metrics across rounds
        self.history = {
            "round": [],
            "train_loss_aggregated": [], # Aggregated from clients' fit results
            "train_dice_aggregated": [], # Aggregated from clients' fit results
            "val_loss_aggregated": [],   # Aggregated from clients' evaluate results
            "val_dice_aggregated": []    # Aggregated from clients' evaluate results
        }
        logger.info(f"SaveModelStrategy initialised. Global models will be saved to: {self.model_save_path}")
        logger.info(f"Metrics will be saved with base name: {self.metrics_filename_base}")


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregates model parameters and metrics from clients after local training (fit).
        Saves the aggregated model checkpoint.
        """
        if not results:
            logger.warning(f"Round {server_round}: aggregate_fit received no results.")
            return None, {}

        # Aggregate parameters using the standard Flower aggregation function
        # parameters_aggregated = aggregate([(client_proxy.get_properties(ins={}, timeout=None), fit_res) for client_proxy, fit_res in results]) # Use default aggregation
        # Correct way to call aggregate for FedAvg (expects weights, num_examples)
        weights_results = [(fit_res.parameters, fit_res.num_examples) for _, fit_res in results]
        parameters_aggregated = fl.common.ndarrays_to_parameters(aggregate(weights_results))


        # Convert aggregated parameters to NDArrays for saving (if aggregation was successful)
        if parameters_aggregated is not None:
            try:
                aggregated_ndarrays: NDArrays = fl.common.parameters_to_ndarrays(parameters_aggregated)
                self._save_model(aggregated_ndarrays, server_round)
            except Exception as e:
                logger.error(f"Round {server_round}: Failed to convert/save aggregated parameters: {e}")
                # Decide if we should proceed without saving or return None
                # return None, {} # Option: Halt if saving fails

        # Aggregate custom metrics reported by clients during fit
        metrics_aggregated = {}
        total_examples = sum(fit_res.num_examples for _, fit_res in results)

        if total_examples > 0:
            # Calculate weighted average for 'train_loss' and 'train_dice'
            weighted_train_loss = sum(fit_res.metrics.get("train_loss", 0.0) * fit_res.num_examples for _, fit_res in results)
            weighted_train_dice = sum(fit_res.metrics.get("train_dice", 0.0) * fit_res.num_examples for _, fit_res in results)

            avg_train_loss = weighted_train_loss / total_examples
            avg_train_dice = weighted_train_dice / total_examples

            metrics_aggregated["train_loss_avg"] = avg_train_loss
            metrics_aggregated["train_dice_avg"] = avg_train_dice

            # Store for history
            # Check if this round already has validation metrics (evaluate might run first)
            if server_round not in self.history["round"]:
                 self.history["round"].append(server_round)
                 # Pad other metrics if needed (though unlikely for fit)
                 for key in self.history:
                     if key != "round" and len(self.history[key]) < len(self.history["round"]):
                         self.history[key].append(None)

            round_index = self.history["round"].index(server_round)
            # Ensure lists are long enough before assignment
            while len(self.history["train_loss_aggregated"]) <= round_index:
                 self.history["train_loss_aggregated"].append(None)
            while len(self.history["train_dice_aggregated"]) <= round_index:
                 self.history["train_dice_aggregated"].append(None)

            self.history["train_loss_aggregated"][round_index] = avg_train_loss
            self.history["train_dice_aggregated"][round_index] = avg_train_dice

            logger.info(f"Round {server_round} aggregated fit metrics: Loss={avg_train_loss:.4f}, Dice={avg_train_dice:.4f}")
        else:
            logger.warning(f"Round {server_round}: aggregate_fit received 0 total examples from clients.")


        # Note: The built-in FedAvg strategy might add its own metrics (like 'accuracy')
        # We are primarily interested in our custom 'train_loss' and 'train_dice'.
        # The second element returned (metrics_aggregated) is mainly for logging purposes by Flower.
        return parameters_aggregated, metrics_aggregated

    def _save_model(self, parameters_ndarrays: NDArrays, server_round: int) -> None:
        """Loads the aggregated parameters into the CPU model instance and saves its state_dict."""
        try:
            # Create the state_dict from the received NumPy arrays
            params_dict = zip(self.model_for_saving.state_dict().keys(), parameters_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

            # Load the state_dict into the model instance
            self.model_for_saving.load_state_dict(state_dict, strict=True)

            # Construct filename and save
            save_path = self.model_save_path / f"global_model_round_{server_round}.pth"
            torch.save(self.model_for_saving.state_dict(), save_path)
            logger.info(f"Saved aggregated global model for round {server_round} to: {save_path}")

        except Exception as e:
            logger.error(f"Error saving global model for round {server_round}: {e}")

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregates evaluation loss and metrics from clients."""
        if not results:
            logger.warning(f"Round {server_round}: aggregate_evaluate received no results.")
            return None, {}

        # Use FedAvg's default aggregation for loss (weighted average of the loss returned by clients)
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)

        # Aggregate custom evaluation metrics (e.g., 'val_dice') using weighted average
        total_examples = sum(res.num_examples for _, res in results)
        if total_examples > 0:
            weighted_val_dice = sum(res.metrics.get("val_dice", 0.0) * res.num_examples for _, res in results)
            avg_val_dice = weighted_val_dice / total_examples
            metrics_aggregated["val_dice_avg"] = avg_val_dice # Add to metrics dict for logging

            # Store for history
            if server_round not in self.history["round"]:
                 self.history["round"].append(server_round)
                 # Pad other metrics if this is the first entry for the round
                 for key in self.history:
                     if key != "round" and len(self.history[key]) < len(self.history["round"]):
                         self.history[key].append(None)

            round_index = self.history["round"].index(server_round)
            # Ensure lists are long enough before assignment
            while len(self.history["val_loss_aggregated"]) <= round_index:
                 self.history["val_loss_aggregated"].append(None)
            while len(self.history["val_dice_aggregated"]) <= round_index:
                 self.history["val_dice_aggregated"].append(None)

            self.history["val_loss_aggregated"][round_index] = loss_aggregated # Store the FedAvg aggregated loss
            self.history["val_dice_aggregated"][round_index] = avg_val_dice

            logger.info(f"Round {server_round} aggregated evaluation metrics: Loss={loss_aggregated:.4f}, Dice={avg_val_dice:.4f}")
        else:
             logger.warning(f"Round {server_round}: aggregate_evaluate received 0 total examples.")


        return loss_aggregated, metrics_aggregated

    def save_metrics_and_plot(self):
        """Saves the collected metrics history to a CSV file and generates a plot."""
        if not self.history["round"]:
            logger.warning("No metrics history recorded. Skipping saving/plotting.")
            return

        # Ensure all metric lists have the same length as the 'round' list by padding with None
        max_len = len(self.history["round"])
        for key in self.history:
            if key != "round":
                while len(self.history[key]) < max_len:
                    self.history[key].append(None)

        # Create DataFrame from the history
        df = pd.DataFrame(self.history)
        df = df.sort_values(by="round").reset_index(drop=True) # Ensure rounds are sorted

        # Define filenames
        csv_filename = f"./{self.metrics_filename_base}.csv"
        plot_filename = f"./{self.metrics_filename_base}.png"

        # Save metrics to CSV
        try:
            df.to_csv(csv_filename, index=False)
            logger.info(f"Aggregated metrics saved to {csv_filename}")
        except Exception as e:
            logger.error(f"Error saving metrics to CSV ({csv_filename}): {e}")

        # Generate and save plot
        try:
            fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True) # Share x-axis (rounds)

            # Plot Loss (Training vs Validation)
            axes[0].plot(df["round"], df["train_loss_aggregated"], marker='o', linestyle='-', label="Avg. Training Loss")
            axes[0].plot(df["round"], df["val_loss_aggregated"], marker='x', linestyle='--', label="Avg. Validation Loss")
            axes[0].set_title("Aggregated Loss Curve")
            axes[0].set_ylabel("Loss")
            axes[0].legend()
            axes[0].grid(True, linestyle=':')

            # Plot Dice Score (Training vs Validation - Tumour Mean)
            axes[1].plot(df["round"], df["train_dice_aggregated"], marker='o', linestyle='-', label="Avg. Training Dice (Tumour Mean)")
            axes[1].plot(df["round"], df["val_dice_aggregated"], marker='x', linestyle='--', label="Avg. Validation Dice (Tumour Mean)")
            axes[1].set_title("Aggregated Dice Score Curve (Tumour Mean)")
            axes[1].set_xlabel("Federated Round")
            axes[1].set_ylabel("Dice Score")
            axes[1].legend()
            axes[1].grid(True, linestyle=':')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
            fig.suptitle(f"Federated Learning Metrics ({self.num_clients_for_filename} Clients)", fontsize=16)
            plt.savefig(plot_filename, dpi=150)
            logger.info(f"Learning curve plot saved to {plot_filename}")

        except Exception as e:
            logger.error(f"Error generating metrics plot ({plot_filename}): {e}")
        finally:
            plt.close(fig) # Close the plot figure


# --- Server Setup Functions ---

def get_evaluate_fn(args):
    """
    Returns a function for server-side evaluation (Optional).
    In this setup, evaluation is primarily done by clients, so this returns None.
    """
    # If server-side evaluation on a central dataset were needed, implement it here.
    # It would typically load the global model, load a server dataset, and perform evaluation.
    logger.info("Server-side evaluation function not implemented (evaluation performed by clients).")
    return None

def get_ip_address():
    """Attempts to get the primary local IP address of the machine."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't actually send data, just connects to determine outbound IP
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
    except Exception:
        logger.warning("Could not automatically determine local IP address. Using 'localhost'.")
        ip = 'localhost' # Fallback
    finally:
        s.close()
    return ip

def save_server_address(address, file_path="./server_address.txt"):
    """Saves the determined server address to a file for clients to read."""
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path: # Ensure directory exists if path includes one
            os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "w") as f:
            f.write(address)
        logger.info(f"Server address saved to {file_path}: {address}")
    except IOError as e:
        logger.error(f"Could not write server address to {file_path}: {e}")


def start_server(args):
    """Configures and starts the Flower federated learning server."""

    # Determine server address: Use specified IP/host or try to get local IP
    host = args.server_address.split(":")[0]
    port = args.server_address.split(":")[-1]
    if host in ["[::]", "0.0.0.0"]:
         # If listening on all interfaces, try to get a specific local IP for clients
         determined_ip = get_ip_address()
         server_address_for_clients = f"{determined_ip}:{port}"
         logger.info(f"Server listening on {args.server_address}, advertising address {server_address_for_clients} for clients.")
         save_server_address(server_address_for_clients) # Save the specific IP for clients
    else:
         # Use the explicitly provided address
         server_address_for_clients = args.server_address
         logger.info(f"Server using explicit address: {server_address_for_clients}")
         save_server_address(server_address_for_clients) # Save the explicit address

    # Configure the strategy
    # Minimum clients: Wait for at least num_clients * fraction_fit (or 1)
    min_fit_clients = max(1, int(args.num_clients * args.fraction_fit))
    # Evaluate on the same number of clients used for fitting
    min_evaluate_clients = min_fit_clients
    # Wait until the required number of clients are available
    min_available_clients = args.num_clients

    logger.info(f"Strategy settings: min_fit_clients={min_fit_clients}, min_evaluate_clients={min_evaluate_clients}, min_available_clients={min_available_clients}")

    strategy = SaveModelStrategy(
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_fit, # Use same fraction for evaluation
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        evaluate_fn=get_evaluate_fn(args), # Server-side evaluation (None in this case)
        model_path="./global_models",
        metrics_filename_base="learning_curve",
        args_config=args, # Pass args for use within strategy
        # Function to send configuration data to clients during fit
        on_fit_config_fn=lambda server_round: {"round": server_round, "local_epochs": args.epochs},
        # Function to send configuration data to clients during evaluate
        on_evaluate_config_fn=lambda server_round: {"round": server_round},
        # Use default FedAvg aggregation for fit metrics (loss, num_examples)
        # Custom fit metrics (train_loss, train_dice) are aggregated within SaveModelStrategy.aggregate_fit
        fit_metrics_aggregation_fn=None,
        # Use default FedAvg aggregation for evaluate metrics (loss, num_examples)
        # Custom evaluate metrics (val_dice) are aggregated within SaveModelStrategy.aggregate_evaluate
        evaluate_metrics_aggregation_fn=None # Let FedAvg handle primary loss aggregation
    )

    # Configure server settings
    server_config = fl.server.ServerConfig(num_rounds=args.rounds)

    logger.info(f"Starting Flower server at {args.server_address} for {args.rounds} rounds...")

    # Start the Flower server
    history = fl.server.start_server(
        server_address=args.server_address, # Address to bind to
        config=server_config,
        strategy=strategy
    )

    logger.info("Federated learning process finished.")

    # After the server stops, save the final metrics and generate plots
    logger.info("Saving final metrics and generating plots...")
    strategy.save_metrics_and_plot()

    return history # Return the history object containing aggregated metrics


