# server.py - Federated learning server implementation (FedProx version)
import flwr as fl
import logging
from typing import Dict, List, Tuple, Optional, Union, Callable # Added Callable
from flwr.common import Metrics, Parameters, FitIns, EvaluateIns, FitRes, EvaluateRes, NDArrays, Scalar, Config # Added Config
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedProx # Import FedProx strategy
import numpy as np
import os
from pathlib import Path
import torch
from model import UNet2D # Import the 2D model
import socket
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays # For handling parameters

logger = logging.getLogger("server")

# Helper function to aggregate custom evaluation metrics from clients (remains the same)
def aggregate_custom_metrics(results: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate custom evaluation metrics like 'val_dice'."""
    if not results: return {}
    total_examples = sum(num_examples for num_examples, _ in results)
    if total_examples == 0:
        logger.warning("Aggregate_custom_metrics received 0 examples.")
        return {}

    aggregated_metrics = {}
    # Aggregate 'val_dice'
    weighted_dice_sum = 0.0
    num_dice_contributions = 0
    for num_examples, metrics in results:
        dice = metrics.get("val_dice")
        if dice is not None:
            try:
                 weighted_dice_sum += float(dice) * num_examples
                 num_dice_contributions += num_examples
            except (ValueError, TypeError): logger.warning(f"Could not convert val_dice '{dice}' to float.")
    if num_dice_contributions > 0: aggregated_metrics["val_dice"] = weighted_dice_sum / num_dice_contributions
    else: aggregated_metrics["val_dice"] = None

    # Aggregate 'val_loss'
    weighted_loss_sum = 0.0
    num_loss_contributions = 0
    for num_examples, metrics in results:
        loss = metrics.get("val_loss")
        if loss is not None:
            try:
                weighted_loss_sum += float(loss) * num_examples
                num_loss_contributions += num_examples
            except (ValueError, TypeError): logger.warning(f"Could not convert val_loss '{loss}' to float.")
    if num_loss_contributions > 0: aggregated_metrics["custom_avg_val_loss"] = weighted_loss_sum / num_loss_contributions
    else: aggregated_metrics["custom_avg_val_loss"] = None

    logger.debug(f"Aggregated custom evaluation metrics: {aggregated_metrics}") # Use debug level
    return aggregated_metrics


class SaveModelFedProxStrategy(FedProx): # Inherit from FedProx
    """Strategy using FedProx, adding model saving and metric plotting"""

    def __init__(
        self,
        *args,
        proximal_mu: float, # Added proximal_mu
        model_path: str = "./global_models",
        metrics_filename: str = "federated_metrics_fedprox_2d.csv", # Updated filename
        plot_filename: str = "learning_curve_fedprox_2d.png",     # Updated filename
        args_namespace=None, # Pass the argparse namespace
        **kwargs
    ):
        # Initialize FedProx with its arguments, including proximal_mu
        super().__init__(*args, proximal_mu=proximal_mu, **kwargs)
        self.model_path = Path(model_path)
        self.metrics_filename = metrics_filename
        self.plot_filename = plot_filename
        self.args = args_namespace # Store args for potential use (like num_clients)
        self.num_clients = self.args.num_clients if self.args else 'unknown'

        # Create model directory
        self.model_path.mkdir(parents=True, exist_ok=True)

        # Initialize metrics history
        self.history = {
            "round": [], "train_loss": [], "train_dice": [],
            "val_loss": [], "val_dice": []
        }

        # Initialize the 2D model structure on CPU (needed for saving state_dict)
        self.model_structure = UNet2D(n_channels=4, n_classes=4).to(torch.device("cpu"))
        logger.info(f"SaveModelFedProxStrategy initialized with proximal_mu={proximal_mu}")


    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model updates using FedProx logic, aggregate metrics, and save model."""
        logger.info(f"Server (FedProx): Aggregating fit results for round {server_round}")

        # Aggregate parameters using the parent FedProx method
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )

        # Save the aggregated model if aggregation was successful
        if parameters_aggregated is not None:
            self._save_model(parameters_aggregated, server_round)
        else:
             logger.warning(f"Round {server_round}: Parameter aggregation failed (FedProx). Model not saved.")


        # --- Aggregate Custom Training Metrics ---
        if results:
            total_examples = sum(fit_res.num_examples for _, fit_res in results)
            weighted_train_loss = 0.0
            weighted_train_dice = 0.0
            num_loss_contrib = 0
            num_dice_contrib = 0

            for _, fit_res in results:
                if fit_res.metrics:
                    train_loss = fit_res.metrics.get("train_loss")
                    train_dice = fit_res.metrics.get("train_dice")
                    num_ex = fit_res.num_examples
                    if train_loss is not None:
                        try:
                            weighted_train_loss += float(train_loss) * num_ex
                            num_loss_contrib += num_ex
                        except (ValueError, TypeError): logger.warning(f"Could not convert train_loss '{train_loss}' to float.")
                    if train_dice is not None:
                         try:
                             weighted_train_dice += float(train_dice) * num_ex
                             num_dice_contrib += num_ex
                         except (ValueError, TypeError): logger.warning(f"Could not convert train_dice '{train_dice}' to float.")

            avg_train_loss = weighted_train_loss / num_loss_contrib if num_loss_contrib > 0 else None
            avg_train_dice = weighted_train_dice / num_dice_contrib if num_dice_contrib > 0 else None

            # Store aggregated training metrics
            # Ensure round exists before appending other metrics
            if server_round not in self.history["round"]: self.history["round"].append(server_round)
            # Pad lists if necessary (e.g., if evaluate runs first)
            while len(self.history["train_loss"]) < len(self.history["round"]): self.history["train_loss"].append(None)
            while len(self.history["train_dice"]) < len(self.history["round"]): self.history["train_dice"].append(None)
            # Find index and update/append
            round_idx = self.history["round"].index(server_round)
            if round_idx >= len(self.history["train_loss"]): # Append if new round
                 self.history["train_loss"].append(avg_train_loss)
                 self.history["train_dice"].append(avg_train_dice)
            else: # Update existing round's entry
                 self.history["train_loss"][round_idx] = avg_train_loss
                 self.history["train_dice"][round_idx] = avg_train_dice


            if avg_train_loss is not None: metrics_aggregated["train_loss"] = avg_train_loss
            if avg_train_dice is not None: metrics_aggregated["train_dice"] = avg_train_dice
            logger.info(f"Round {server_round} aggregated training metrics: Loss={avg_train_loss}, Dice={avg_train_dice}")

        else:
            logger.warning(f"Round {server_round}: No results received for fit aggregation.")
            if server_round not in self.history["round"]: self.history["round"].append(server_round)
            while len(self.history["train_loss"]) < len(self.history["round"]): self.history["train_loss"].append(None)
            while len(self.history["train_dice"]) < len(self.history["round"]): self.history["train_dice"].append(None)


        logger.debug(f"Round {server_round} default aggregated fit metrics: {metrics_aggregated}")
        return parameters_aggregated, metrics_aggregated

    def _save_model(self, parameters: Parameters, server_round: int) -> None:
        """Save the 2D model based on the aggregated parameters"""
        if parameters is None:
            logger.warning(f"Round {server_round}: Cannot save model, parameters are None.")
            return
        try:
            params_numpy: NDArrays = parameters_to_ndarrays(parameters)
            state_dict = {
                k: torch.tensor(v)
                for k, v in zip(self.model_structure.state_dict().keys(), params_numpy)
            }
            self.model_structure.load_state_dict(state_dict, strict=True)
            # Add _fedprox suffix
            save_path = self.model_path / f"model_round_{server_round}_fedprox_2d.pth"
            torch.save(self.model_structure.state_dict(), str(save_path))
            logger.info(f"Saved global FedProx model: {save_path}")
        except Exception as e:
            logger.error(f"Error saving FedProx model for round {server_round}: {e}", exc_info=True)

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation loss and metrics from clients using FedProx logic."""
        logger.info(f"Server (FedProx): Aggregating evaluate results for round {server_round}")

        # Use FedProx's aggregation for the primary loss metric
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)

        # Aggregate custom metrics
        custom_metrics = aggregate_custom_metrics([(res.num_examples, res.metrics) for _, res in results])
        metrics_aggregated.update(custom_metrics)

        if loss_aggregated is not None:
             logger.info(f"Round {server_round} aggregated evaluation loss: {loss_aggregated:.4f}")
        logger.info(f"Round {server_round} aggregated evaluation metrics: {metrics_aggregated}")

        # --- Store Validation Metrics for History ---
        try:
             # Ensure round exists (should have been added during fit)
             if server_round not in self.history["round"]:
                  logger.warning(f"Round {server_round} not found in history during evaluate. Appending.")
                  self.history["round"].append(server_round)
                  # Pad other metrics if this happens unexpectedly
                  while len(self.history["train_loss"]) < len(self.history["round"]): self.history["train_loss"].append(None)
                  while len(self.history["train_dice"]) < len(self.history["round"]): self.history["train_dice"].append(None)

             round_index = self.history["round"].index(server_round)
             # Pad val lists if needed
             while len(self.history["val_loss"]) <= round_index: self.history["val_loss"].append(None)
             while len(self.history["val_dice"]) <= round_index: self.history["val_dice"].append(None)

             # Store aggregated metrics
             self.history["val_loss"][round_index] = loss_aggregated
             self.history["val_dice"][round_index] = metrics_aggregated.get("val_dice")

        except ValueError:
             logger.error(f"Value error finding round {server_round} index in history during evaluate.")
        except Exception as e:
             logger.error(f"Unexpected error updating history during evaluate: {e}")

        return loss_aggregated, metrics_aggregated

    def save_metrics_and_plot(self):
        """Saves the collected metrics to a CSV and plots the learning curve."""
        if not self.history["round"]:
            logger.warning("No metrics history recorded. Skipping saving/plotting.")
            return

        # Ensure all lists have the same length before creating DataFrame
        max_len = max(len(lst) for lst in self.history.values())
        for key in self.history:
            while len(self.history[key]) < max_len:
                self.history[key].append(None) # Pad with None

        try:
            df = pd.DataFrame(self.history)
            df = df.sort_values(by="round").reset_index(drop=True) # Ensure rounds are sorted
        except Exception as e:
             logger.error(f"Error creating DataFrame from history: {e}")
             logger.error(f"History content: {self.history}")
             return

        csv_path = Path(f"./{self.metrics_filename}")
        plot_path = Path(f"./{self.plot_filename}")

        try:
            df.to_csv(csv_path, index=False)
            logger.info(f"Metrics saved to {csv_path}")
        except Exception as e: logger.error(f"Error saving metrics to CSV {csv_path}: {e}")

        try:
            fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            fig.suptitle(f'Federated Learning (FedProx) Metrics (Clients: {self.num_clients}, mu={self.proximal_mu})') # Include mu in title

            axes[0].plot(df["round"], df["train_loss"], marker='o', linestyle='-', label="Avg. Train Loss (per round)")
            axes[0].plot(df["round"], df["val_loss"], marker='x', linestyle='--', label="Agg. Validation Loss")
            axes[0].set_title("Loss Curve")
            axes[0].set_ylabel("Loss")
            axes[0].legend()
            axes[0].grid(True, linestyle=':')
            axes[0].scatter(df["round"], df["train_loss"], marker='o')
            axes[0].scatter(df["round"], df["val_loss"], marker='x')

            axes[1].plot(df["round"], df["train_dice"], marker='o', linestyle='-', label="Avg. Train Dice (Tumor Mean, per round)")
            axes[1].plot(df["round"], df["val_dice"], marker='x', linestyle='--', label="Agg. Validation Dice (Tumor Mean)")
            axes[1].set_title("Dice Score Curve (Tumor Mean)")
            axes[1].set_xlabel("Federated Round")
            axes[1].set_ylabel("Dice Score")
            axes[1].legend()
            axes[1].grid(True, linestyle=':')
            axes[1].scatter(df["round"], df["train_dice"], marker='o')
            axes[1].scatter(df["round"], df["val_dice"], marker='x')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
            plt.savefig(plot_path)
            logger.info(f"Learning curve plot saved to {plot_path}")
            plt.close(fig)
        except Exception as e: logger.error(f"Error generating plot {plot_path}: {e}", exc_info=True)


# --- Server Setup Functions --- (get_evaluate_fn, get_local_ip, save_server_address remain the same)
def get_evaluate_fn(args):
    logger.info("Server-side evaluation (evaluate_fn) is not configured.")
    return None

def get_local_ip():
    """Get the local IP address of this machine"""
    try:
        hostname = socket.gethostname()
        return socket.gethostbyname(hostname)
    except Exception as e:
        logger.error(f"Error determining IP address: {e}")
        return "localhost"

def save_server_address(address, file_path="./server_address.txt"):
    """Save the server address to a file"""
    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)
    with open(file_path, "w") as f:
        f.write(address)
    logger.info(f"Server address saved to {file_path}: {address}")


def start_server(args):
    """Start the federated learning server for 2D segmentation"""
    # Determine server address
    local_ip = get_local_ip()
    try:
        port = args.server_address.split(":")[-1]
        if not port.isdigit():
             port = "8080" # Default port if parsing fails
             logger.warning(f"Could not parse port from {args.server_address}, using default {port}.")
    except:
         port = "8080"
         logger.warning(f"Could not parse port from {args.server_address}, using default {port}.")

    # Use 0.0.0.0 to listen on all available interfaces, making it accessible externally
    server_address = f"0.0.0.0:{port}"
    # Save the *accessible* IP for clients
    accessible_address = f"{local_ip}:{port}"
    save_server_address(accessible_address)

    min_fit_clients = max(1, int(args.fraction_fit * args.num_clients))
    min_eval_clients = max(1, int(args.fraction_evaluate * args.num_clients))
    min_available_clients = args.num_clients

    logger.info(f"Server configuration (FedProx):")
    logger.info(f"  Proximal Mu (mu): {args.proximal_mu}") # Log mu
    logger.info(f"  Total clients expected: {args.num_clients}")
    logger.info(f"  Fraction fit per round: {args.fraction_fit} (min {min_fit_clients} clients)")
    logger.info(f"  Fraction evaluate per round: {args.fraction_evaluate} (min {min_eval_clients} clients)")
    logger.info(f"  Min available clients: {min_available_clients}")
    logger.info(f"  Number of rounds: {args.rounds}")

    # --- Define Fit Configuration Function ---
    # This function will be called by the strategy before each fit round
    # It passes round number, learning rate, and proximal_mu to the clients
    def fit_config(server_round: int) -> Config:
        config = {
            "round": server_round,
            "local_epochs": args.epochs, # Can pass local epochs if needed
            "learning_rate": args.learning_rate, # Can pass LR if needed
            "proximal_mu": args.proximal_mu # Pass mu to the client
        }
        return config

    # --- Initialize Strategy ---
    # Use the custom strategy inheriting from FedProx
    strategy = SaveModelFedProxStrategy(
        # FedProx parameters
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_eval_clients,
        min_available_clients=min_available_clients,
        proximal_mu=args.proximal_mu, # Pass mu here
        # Server-side evaluation (optional)
        evaluate_fn=get_evaluate_fn(args),
        # --- Parameter and Metric Handling ---
        on_fit_config_fn=fit_config, # Use the function defined above
        on_evaluate_config_fn=lambda server_round: {"round": server_round}, # Simple eval config
        evaluate_metrics_aggregation_fn=aggregate_custom_metrics,
        # initial_parameters=None, # Optional
        # --- Custom Strategy Parameters ---
        model_path="./global_models",
        metrics_filename=f"federated_metrics_clients_{args.num_clients}_fedprox_2d.csv",
        plot_filename=f"learning_curve_clients_{args.num_clients}_fedprox_2d.png",
        args_namespace=args
    )

    logger.info(f"Starting Flower server (FedProx) at {server_address}...")
    logger.info(f"Clients should connect to {accessible_address}")

    # --- Start Server ---
    try:
        history = fl.server.start_server(
            server_address=server_address,
            config=fl.server.ServerConfig(num_rounds=args.rounds),
            strategy=strategy,
        )
        logger.info("Federated learning (FedProx) finished successfully.")
    except Exception as e:
         logger.error(f"Federated learning server (FedProx) failed: {e}", exc_info=True)
         history = None

    # Save metrics and plot learning curve using the strategy instance
    logger.info("Saving final metrics and plotting learning curve...")
    strategy.save_metrics_and_plot()

    return history

