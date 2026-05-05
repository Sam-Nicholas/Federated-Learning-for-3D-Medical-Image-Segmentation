# Federated Learning for Brain Tumour Segmentation

MEng Computer Science with Artificial Intelligence — Dissertation
University of Southampton | Samuel Nicholas | April 2025

---

## Overview

This project implements federated learning (FL) for glioma segmentation on the [BraTS 2024 Post-Treatment dataset](https://www.synapse.org/Synapse:syn53708249/wiki/), using the [Flower (flwr)](https://flower.ai/) framework. The goal is to simulate a decentralised healthcare network, like a scaled-down version of what you might see across NHS trusts, where hospitals can collaboratively train a shared model without ever sharing their raw patient data.

Two segmentation architectures are compared (2D U-Net and 3D U-Net), along with two federated aggregation strategies (FedAvg and FedProx). The code is split across three folders, each representing a different experimental configuration.

---

## Repository Structure

```
├── 2D U-Net/               # 2D U-Net with FedAvg
│   ├── main.py
│   ├── client.py
│   ├── server.py
│   ├── model.py
│   ├── dataset.py
│   └── utils.py
│
├── 2D U-Net FedProx/       # 2D U-Net with FedProx
│   ├── main.py
│   ├── client.py
│   ├── server.py
│   ├── model.py
│   ├── dataset.py
│   └── utils.py
│
└── 3D U-Net/               # 3D U-Net with FedAvg
    ├── main.py
    ├── client.py
    ├── server.py
    ├── model.py
    ├── dataset.py
    └── utils.py
```

Each folder is a self-contained implementation. The files are largely similar across folders, with the key differences being the model architecture (2D vs 3D convolutions) and the aggregation strategy configured on the server side (FedAvg vs FedProx).

---

## File Descriptions

| File | Description |
|---|---|
| `main.py` | Entry point. Parses all CLI arguments and launches either the server or a client process. |
| `server.py` | Configures and starts the Flower server. Handles the aggregation strategy, global model checkpointing, and round-by-round evaluation. |
| `client.py` | Defines the Flower client. Loads local data, runs local training and evaluation, and communicates model weights with the server. |
| `model.py` | Contains the U-Net architecture (2D or 3D depending on the folder). Built in PyTorch. |
| `dataset.py` | Handles loading and preprocessing of the BraTS NIfTI data. Includes slice extraction (for 2D), normalisation, one-hot encoding, and visualisation utilities. |
| `utils.py` | Training and evaluation loops, loss functions (combined Dice + Cross-Entropy), Dice metric computation, and checkpoint save/load logic. |

---

## Background

### Federated Learning

In standard centralised training, all data goes to one place, which is a non-starter in healthcare due to GDPR and general data governance. Federated learning solves this by keeping patient data local to each institution. Instead of raw data, only model weights are sent to a central server, which aggregates them and distributes the updated global model back to clients.

This project simulates that setup using Flower's client-server architecture, with each "client" representing a hospital holding its own slice of the BraTS dataset.

### Aggregation Strategies

**FedAvg** -> the standard approach. After local training, the server takes a weighted average of all client model weights. Simple and effective, but can struggle when clients have very different data distributions (non-IID).

**FedProx** -> extends FedAvg by adding a proximal term to each client's local loss function:

$$H_k(w; w_t) = F_k(w) + \frac{\mu}{2} \|w - w_t\|^2$$

This penalises local models for drifting too far from the global model, which tends to produce more stable convergence on heterogeneous data. When μ = 0, FedProx reduces to FedAvg.

### Model Architectures

**2D U-Net** -> processes individual axial slices from the MRI volumes (4 channels × H × W). Significantly faster and more memory-efficient. Initial filter count set to 64.

**3D U-Net** -> processes full volumetric patches (4 channels × H × W × D), allowing the model to learn inter-slice spatial relationships. Considerably heavier — around 20× slower per sample on CPU.

Both models use a standard encoder-decoder structure with skip connections, and output 4-class segmentation maps (background, oedema, non-enhancing core, enhancing core).

---

## Dataset

This project uses the **BraTS 2024 Adult Glioma Post-Treatment** dataset. Each patient case contains four co-registered MRI modalities:

- **T1** - T1-weighted MRI
- **T1c** - T1-weighted with contrast (highlights active tumour)
- **T2** - T2-weighted MRI
- **FLAIR** - highlights oedema/swelling

Segmentation labels:
- `0` - Background
- `1` - Non-Enhancing Tumour Core
- `2` - Oedema
- `4` - Enhancing Tumour

> Note: Label 4 is remapped to class 3 internally during preprocessing to give contiguous indices.

You'll need to apply for access to the dataset via [Synapse](https://www.synapse.org/Synapse:syn53708249/wiki/) and agree to the data use agreement before downloading.

---

## Setup

### Requirements

```bash
pip install torch torchvision
pip install flwr
pip install nibabel
pip install numpy tqdm matplotlib
```

Tested with Python 3.10+. GPU (CUDA) is strongly recommended, particularly for the 3D U-Net, which is borderline unusable on CPU for anything beyond very small experiments.

### Data Directory Structure

The expected structure for `--data_dir` is:

```
BraTS_data/
└── train/
    ├── BraTS-GLI-00001-000/
    │   ├── BraTS-GLI-00001-000-t1n.nii.gz
    │   ├── BraTS-GLI-00001-000-t1c.nii.gz
    │   ├── BraTS-GLI-00001-000-t2w.nii.gz
    │   ├── BraTS-GLI-00001-000-t2f.nii.gz
    │   └── BraTS-GLI-00001-000-seg.nii.gz
    ├── BraTS-GLI-00002-000/
    ...
```

---

## Usage

Everything is launched through `main.py` using `--mode server` or `--mode client`. You'll need to start the server first, then launch each client in a separate terminal (or process).

### Start the Server

```bash
python main.py --mode server \
    --num_clients 2 \
    --rounds 15 \
    --server_address 0.0.0.0:8080
```

### Start a Client

```bash
python main.py --mode client \
    --client_id 0 \
    --server_address localhost:8080 \
    --data_dir ./BraTS_data/train \
    --epochs 5 \
    --batch_size 5 \
    --device cuda
```

Repeat with `--client_id 1`, `--client_id 2`, etc. for each additional client. Each client should point to the same data directory, partitioning is handled automatically based on client ID.

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--mode` | required | `server` or `client` |
| `--client_id` | `"0"` | Unique ID for each client |
| `--server_address` | `localhost:8080` | Server host:port |
| `--data_dir` | `./BraTS_data/train` | Path to BraTS patient folders |
| `--val_data_dir` | `None` | Optional separate validation set |
| `--num_clients` | `2` | Total clients expected by the server |
| `--rounds` | `10` | Number of federated rounds |
| `--epochs` | `3` | Local training epochs per round |
| `--batch_size` | `32` | Slices per batch |
| `--learning_rate` | `1e-4` | Optimiser learning rate |
| `--max_slices` | `None` | Cap on total slices loaded per client |
| `--slices_per_volume` | `20` | Slices sampled per 3D volume |
| `--require_tumor_train` | `False` | Only use tumour-containing slices for training |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--load_latest_global` | `False` | Resume from latest global checkpoint |
| `--load_latest_client` | `False` | Resume from latest client checkpoint |
| `--save_vis` | `False` | Save segmentation visualisations during training |

---

## Output

Running the code will create the following directories automatically:

```
checkpoints/          # Client model checkpoints saved per epoch
global_models/        # Global model checkpoints saved by the server per round
visualisations/       # Training segmentation output images (if --save_vis)
val_visualisations/   # Validation segmentation output images (if --save_vis)
```

A log file `federated_segmentation_2d.log` is also written to the working directory.

---

## Results Summary

All experiments were run on CPU on a remote HPC cluster (no GPU access due to queue times). This heavily impacted the 3D U-Net in particular.

| Configuration | Clients | Algorithm | Peak Validation Dice |
|---|---|---|---|
| 2D U-Net | 1 | FedAvg | 0.658 |
| 2D U-Net | 1 | FedProx | 0.679 |
| 3D U-Net | 1 | FedAvg | 0.674 |
| 2D U-Net | 2 | FedAvg | 0.674 |
| 2D U-Net | 2 | FedProx | 0.662 |
| 2D U-Net | 10 | FedAvg | 0.670 |
| 2D U-Net | 10 | FedProx | 0.664 |
| 3D U-Net | 10 | FedAvg | 0.527 |

The 2D U-Net proved far more practical, roughly 20× faster per sample than the 3D model on CPU, and much more resilient to the server instability encountered during longer multi-client runs. FedAvg narrowly edged FedProx on peak Dice in the 10-client setting, though FedProx showed visibly smoother convergence curves.

---

## Known Limitations

- All training was done on CPU ude to HPC limits. GPU is essentially required for the 3D U-Net to be usable at any real scale
- The remote HPC server had a 60-hour job limit, which cut off several 3D U-Net runs before completion
- Some multi-client runs saw clients disconnect mid-training; checkpointing was used to recover where possible, but it still impacted results
- FedProx µ was fixed at 0.01 throughout. Further tuning could improve its performance
- No formal privacy guarantees (e.g. differential privacy) are implemented

---

## Acknowledgements

- BraTS 2024 dataset provided via [Synapse](https://www.synapse.org/)
- Flower federated learning framework: [flower.ai](https://flower.ai/)
- Project supervised by Dr Kate Farrahi, University of Southampton
