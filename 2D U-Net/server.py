import flwr as fl
import logging
from typing import Dict, List, Tuple, Optional, Union
from flwr.common import Metrics, Parameters, FitIns, EvaluateIns, FitRes, EvaluateRes, NDArrays, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
import os
from pathlib import Path
import torch
from model import UNet2D
import socket
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays

logger = logging.getLogger("server")


def aggregate_custom_metrics(results: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate custom evaluation metrics like 'val_dice'."""
    if not results:
        return {}

    # Calculate total number of examples (slices in 2D case)
    total_examples = sum(num_examples for num_examples, _ in results)

    if total_examples == 0:
        logger.warning("Aggregate_custom_metrics received 0 examples.")
        return {} # Avoid division by zero

    aggregated_metrics = {}

    # Aggregate 'val_dice' (mean tumor dice)
    weighted_dice_sum = 0.0
    num_dice_contributions = 0
    for num_examples, metrics in results:
        dice = metrics.get("val_dice") # Key used in client's evaluate return
        if dice is not None:
            try:
                 weighted_dice_sum += float(dice) * num_examples
                 num_dice_contributions += num_examples
            except (ValueError, TypeError):
                 logger.warning(f"Could not convert val_dice '{dice}' to float. Skipping.")


    # Calculate average, checking if any clients contributed the metric
    if num_dice_contributions > 0:
         aggregated_metrics["val_dice"] = weighted_dice_sum / num_dice_contributions
    else:
         aggregated_metrics["val_dice"] = None # Or 0.0, depending on desired behavior

    # Aggregate 'val_loss' (optional, FedAvg handles main loss)
    weighted_loss_sum = 0.0
    num_loss_contributions = 0
    for num_examples, metrics in results:
        loss = metrics.get("val_loss") # Key used in client's evaluate return
        if loss is not None:
            try:
                weighted_loss_sum += float(loss) * num_examples
                num_loss_contributions += num_examples
            except (ValueError, TypeError):
                 logger.warning(f"Could not convert val_loss '{loss}' to float. Skipping.")

    if num_loss_contributions > 0:
         # Add with a distinct key if you want both FedAvg's loss and this one
         aggregated_metrics["custom_avg_val_loss"] = weighted_loss_sum / num_loss_contributions
    else:
         aggregated_metrics["custom_avg_val_loss"] = None


    logger.info(f"Aggregated custom evaluation metrics: {aggregated_metrics}")
    return aggregated_metrics


class SaveModelStrategy(FedAvg): # Inherit from FedAvg
    """Strategy to save aggregated 2D model and log/plot metrics after each round"""

    def __init__(
        self,
        *args,
        model_path: str = "./global_models",
        metrics_filename: str = "federated_metrics_2d.csv",
        plot_filename: str = "learning_curve_2d.png",
        args_namespace=None, # Pass the argparse namespace
        **kwargs
    ):
        super().__init__(*args, **kwargs) # Pass args and kwargs to FedAvg constructor
        self.model_path = Path(model_path)
        self.metrics_filename = metrics_filename
        self.plot_filename = plot_filename
        self.args = args_namespace # Store args for potential use (like num_clients)
        self.num_clients = self.args.num_clients if self.args else 'unknown'

        # Create model directory
        self.model_path.mkdir(parents=True, exist_ok=True)

        # Initialize metrics history (using lists for simplicity)
        self.history = {
            "round": [],
            "train_loss": [], # Average loss over clients' local training epochs
            "train_dice": [], # Average dice over clients' local training epochs
            "val_loss": [],   # Aggregated validation loss (from FedAvg)
            "val_dice": []    # Aggregated custom validation dice
        }

        # Initialize the 2D model structure on CPU (needed for saving state_dict)
        # Ensure this matches the client-side model architecture
        self.model_structure = UNet2D(n_channels=4, n_classes=4).to(torch.device("cpu"))
        logger.info("SaveModelStrategy initialized with UNet2D structure for saving.")


    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model updates, metrics, and save the global model"""
        logger.info(f"Server: Aggregating fit results for round {server_round}")

        # Aggregate parameters using the parent FedAvg method
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )

        # Save the aggregated model if aggregation was successful
        if parameters_aggregated is not None:
            self._save_model(parameters_aggregated, server_round)
        else:
             logger.warning(f"Round {server_round}: Parameter aggregation failed. Model not saved.")


        # --- Aggregate Custom Training Metrics ---
        # Calculate weighted average of custom metrics returned by clients in fit
        # These metrics ('train_loss', 'train_dice') represent averages over local epochs per client
        if results:
            total_examples = sum(fit_res.num_examples for _, fit_res in results)
            weighted_train_loss = 0.0
            weighted_train_dice = 0.0
            num_loss_contrib = 0
            num_dice_contrib = 0

            for _, fit_res in results:
                # Access custom metrics returned by the client in fit method's results dict
                if fit_res.metrics:
                    train_loss = fit_res.metrics.get("train_loss")
                    train_dice = fit_res.metrics.get("train_dice")
                    num_ex = fit_res.num_examples

                    if train_loss is not None:
                        try:
                            weighted_train_loss += float(train_loss) * num_ex
                            num_loss_contrib += num_ex
                        except (ValueError, TypeError):
                             logger.warning(f"Could not convert train_loss '{train_loss}' to float.")
                    if train_dice is not None:
                         try:
                             weighted_train_dice += float(train_dice) * num_ex
                             num_dice_contrib += num_ex
                         except (ValueError, TypeError):
                              logger.warning(f"Could not convert train_dice '{train_dice}' to float.")


            # Calculate averages, checking for valid contributions
            avg_train_loss = weighted_train_loss / num_loss_contrib if num_loss_contrib > 0 else None
            avg_train_dice = weighted_train_dice / num_dice_contrib if num_dice_contrib > 0 else None

            # Store aggregated training metrics for history
            self.history["round"].append(server_round) # Add round number
            self.history["train_loss"].append(avg_train_loss)
            self.history["train_dice"].append(avg_train_dice)

            # Add to standard aggregated metrics dict for Flower's logging
            if avg_train_loss is not None: metrics_aggregated["train_loss"] = avg_train_loss
            if avg_train_dice is not None: metrics_aggregated["train_dice"] = avg_train_dice

            logger.info(f"Round {server_round} aggregated training metrics: Loss={avg_train_loss:.4f}, Dice={avg_train_dice:.4f}")
        else:
            logger.warning(f"Round {server_round}: No results received for fit aggregation.")
            # Append Nones if no results, keeping lists aligned
            if server_round not in self.history["round"]: self.history["round"].append(server_round)
            self.history["train_loss"].append(None)
            self.history["train_dice"].append(None)


        # Log default aggregated metrics (like 'accuracy' if FedAvg calculates it)
        logger.info(f"Round {server_round} default aggregated fit metrics: {metrics_aggregated}")

        return parameters_aggregated, metrics_aggregated

    def _save_model(self, parameters: Parameters, server_round: int) -> None:
        """Save the 2D model based on the aggregated parameters"""
        if parameters is None:
            logger.warning(f"Round {server_round}: Cannot save model, parameters are None.")
            return
        try:
            # Convert Flower Parameters to NumPy arrays
            params_numpy: NDArrays = parameters_to_ndarrays(parameters)

            # Create state_dict using the keys from the model structure
            state_dict = {
                k: torch.tensor(v)
                for k, v in zip(self.model_structure.state_dict().keys(), params_numpy)
            }

            # Load the state_dict into the model structure
            self.model_structure.load_state_dict(state_dict, strict=True)

            # Define save path with _2d suffix
            save_path = self.model_path / f"model_round_{server_round}_2d.pth"

            # Save the model state_dict
            torch.save(self.model_structure.state_dict(), str(save_path))
            logger.info(f"Saved global 2D model: {save_path}")

        except Exception as e:
            logger.error(f"Error saving model for round {server_round}: {e}", exc_info=True)

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation loss and metrics from clients."""
        logger.info(f"Server: Aggregating evaluate results for round {server_round}")

        # Use FedAvg's aggregation for the primary loss metric
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)

        # Aggregate custom metrics (like 'val_dice') using our helper function
        custom_metrics = aggregate_custom_metrics([(res.num_examples, res.metrics) for _, res in results])

        # Add custom aggregated metrics to the main metrics dictionary
        metrics_aggregated.update(custom_metrics)

        logger.info(f"Round {server_round} aggregated evaluation loss: {loss_aggregated:.4f}")
        logger.info(f"Round {server_round} aggregated evaluation metrics: {metrics_aggregated}")


        # --- Store Validation Metrics for History ---
        # Ensure history lists are aligned with the current round
        # Find index for the current round. If fit ran, it should exist.
        try:
            round_index = self.history["round"].index(server_round)
            # Pad val_loss and val_dice lists if evaluate runs before fit somehow
            while len(self.history["val_loss"]) <= round_index:
                 self.history["val_loss"].append(None)
            while len(self.history["val_dice"]) <= round_index:
                 self.history["val_dice"].append(None)

            # Store aggregated validation loss (from FedAvg)
            self.history["val_loss"][round_index] = loss_aggregated
            # Store aggregated custom validation dice
            self.history["val_dice"][round_index] = metrics_aggregated.get("val_dice") # Use .get for safety

        except ValueError:
             # If round not found (e.g., evaluate called before fit on round 0)
             logger.warning(f"Round {server_round} not found in history during evaluate aggregation. Appending.")
             if server_round not in self.history["round"]: self.history["round"].append(server_round)
             # Append Nones to train metrics if necessary
             while len(self.history["train_loss"]) < len(self.history["round"]): self.history["train_loss"].append(None)
             while len(self.history["train_dice"]) < len(self.history["round"]): self.history["train_dice"].append(None)
             # Append current val metrics
             self.history["val_loss"].append(loss_aggregated)
             self.history["val_dice"].append(metrics_aggregated.get("val_dice"))


        return loss_aggregated, metrics_aggregated

    def save_metrics_and_plot(self):
        """Saves the collected metrics to a CSV and plots the learning curve."""
        if not self.history["round"]:
            logger.warning("No metrics history recorded. Skipping saving/plotting.")
            return

        # Ensure all lists have the same length before creating DataFrame
        max_len = len(self.history["round"])
        for key in self.history:
            while len(self.history[key]) < max_len:
                self.history[key].append(None) # Pad with None

        # Create DataFrame
        try:
            df = pd.DataFrame(self.history)
            df = df.sort_values(by="round").reset_index(drop=True) # Ensure rounds are sorted
        except Exception as e:
             logger.error(f"Error creating DataFrame from history: {e}")
             logger.error(f"History content: {self.history}")
             return


        # Define unique filenames using instance variables
        csv_path = Path(f"./{self.metrics_filename}")
        plot_path = Path(f"./{self.plot_filename}")


        # Save to CSV
        try:
            df.to_csv(csv_path, index=False)
            logger.info(f"Metrics saved to {csv_path}")
        except Exception as e:
            logger.error(f"Error saving metrics to CSV {csv_path}: {e}")

        # Plotting
        try:
            fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            fig.suptitle(f'Federated Learning Metrics (Clients: {self.num_clients})')

            # Plot Loss
            axes[0].plot(df["round"], df["train_loss"], marker='o', linestyle='-', label="Avg. Train Loss (per round)")
            axes[0].plot(df["round"], df["val_loss"], marker='x', linestyle='--', label="Agg. Validation Loss")
            axes[0].set_title("Loss Curve")
            axes[0].set_ylabel("Loss")
            axes[0].legend()
            axes[0].grid(True, linestyle=':')
            # Add markers even if lines are missing points due to None
            axes[0].scatter(df["round"], df["train_loss"], marker='o')
            axes[0].scatter(df["round"], df["val_loss"], marker='x')


            # Plot Dice Score
            axes[1].plot(df["round"], df["train_dice"], marker='o', linestyle='-', label="Avg. Train Dice (Tumor Mean, per round)")
            axes[1].plot(df["round"], df["val_dice"], marker='x', linestyle='--', label="Agg. Validation Dice (Tumor Mean)")
            axes[1].set_title("Dice Score Curve (Tumor Mean)")
            axes[1].set_xlabel("Federated Round")
            axes[1].set_ylabel("Dice Score")
            axes[1].legend()
            axes[1].grid(True, linestyle=':')
            # Add markers
            axes[1].scatter(df["round"], df["train_dice"], marker='o')
            axes[1].scatter(df["round"], df["val_dice"], marker='x')


            plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title overlap
            plt.savefig(plot_path)
            logger.info(f"Learning curve plot saved to {plot_path}")
            plt.close(fig)

        except Exception as e:
            logger.error(f"Error generating plot {plot_path}: {e}", exc_info=True)


# --- Server Setup Functions ---

def get_evaluate_fn(args):
    """Create an evaluation function for server-side evaluation (Optional)."""
    # This is less common in FL but could be used if the server has a holdout dataset.
    # For now, we rely on client-side evaluation.
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

    # Determine minimum clients for fitting and evaluation
    # Ensure min_clients is at least 1 and not more than num_clients
    min_fit_clients = max(1, int(args.fraction_fit * args.num_clients))
    min_eval_clients = max(1, int(args.fraction_evaluate * args.num_clients)) # Use separate fraction for eval
    min_available_clients = args.num_clients # Wait for all clients to be available initially

    logger.info(f"Server configuration:")
    logger.info(f"  Total clients expected: {args.num_clients}")
    logger.info(f"  Fraction fit per round: {args.fraction_fit} (min {min_fit_clients} clients)")
    logger.info(f"  Fraction evaluate per round: {args.fraction_evaluate} (min {min_eval_clients} clients)")
    logger.info(f"  Min available clients: {min_available_clients}")
    logger.info(f"  Number of rounds: {args.rounds}")


    # --- Initialize Strategy ---
    # Define strategy with custom aggregation and saving logic
    strategy = SaveModelStrategy(
        # FedAvg parameters
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_eval_clients,
        min_available_clients=min_available_clients,
        # Server-side evaluation (optional)
        evaluate_fn=get_evaluate_fn(args),
        # --- Parameter and Metric Handling ---
        # Function to pass config data (like round number) to clients during fit
        on_fit_config_fn=lambda server_round: {"round": server_round, "lr": args.learning_rate}, # Pass LR too if needed
        # Function to pass config data to clients during evaluate
        on_evaluate_config_fn=lambda server_round: {"round": server_round},
        # Initial parameters (optional - can start training from scratch)
        # initial_parameters=None, # Or load from a checkpoint if desired
        # --- Custom Strategy Parameters ---
        model_path="./global_models", # Directory to save global models
        metrics_filename=f"federated_metrics_clients_{args.num_clients}_2d.csv", # Unique metrics filename
        plot_filename=f"learning_curve_clients_{args.num_clients}_2d.png",     # Unique plot filename
        args_namespace=args # Pass command line args to strategy
    )

    logger.info(f"Starting Flower server at {server_address}...")
    logger.info(f"Clients should connect to {accessible_address}")


    # --- Start Server ---
    try:
        history = fl.server.start_server(
            server_address=server_address,
            config=fl.server.ServerConfig(num_rounds=args.rounds),
            strategy=strategy,
            # Add SSL context if using secure connection
            # certificates=(
            #     Path(".cache/certificates/ca.crt").read_bytes(),
            #     Path(".cache/certificates/server.pem").read_bytes(),
            #     Path(".cache/certificates/server.key").read_bytes(),
            # )
        )
        logger.info("Federated learning finished successfully.")

    except Exception as e:
         logger.error(f"Federated learning server failed: {e}", exc_info=True)
         history = None # Indicate failure


    # Save metrics and plot learning curve using the strategy instance after server stops
    logger.info("Saving final metrics and plotting learning curve...")
    strategy.save_metrics_and_plot()

    return history # Return history for potential analysis

