# NetSquid-FSO-QKD

Simulation framework for long-distance ground-to-ground free-space quantum key distribution (FSO-QKD) using the elliptic beam atmospheric channel model and a full end-to-end BBM92 entanglement-based protocol in NetSquid.

---

## Features

- Elliptic beam channel model (adapted from Liorni-Kampermann-Bruß) with Monte Carlo transmittance sampling for turbulence + weather-dependent scattering.
- Complete BBM92 protocol pipeline: entangled pair generation, coincidence detection, sifting, QBER estimation, CASCADE error correction, and Toeplitz privacy amplification.
- Background noise modeling (day/night).
- Six atmospheric conditions (calm/moderate/windy daytime + clear/slightly/moderately foggy nighttime).
- Reproducible Jupyter notebooks for loss analysis and key rate simulations.

---

## Repository Structure (Key Files)

- **`modern_gui.py`** — Main interactive GUI (recommended entry point for most users).
- **`quick.py`** — Core elliptic beam loss model implementation.
- **`loss_model_analysis.ipynb`** — Generates transmittance vs. distance plots, wavelength comparison (785 nm vs 1550 nm), and Probability Distributions of Transmittance (PDTs) shown in the paper.
- **`master_data.ipynb`** — Produces the final key rates and Table 3 results from the paper.
- **`qkd_runner.py`**, `key_distribution.py`, `coincidence.py`, `key_sifting.py`, etc. — Modular protocol components.
- **`info_reco.py`** & **`privacy_amp.py`** — Adapters for CASCADE and randextract.
- **`requirements.txt`** — All Python dependencies (excluding NetSquid — see Installation below).
- `loss_data/`, `day1_72.json`, `night1_72.json` — Precomputed Monte Carlo data.
- `cascade/` & `randextract/` — Embedded (slightly adapted) sub-repositories.

---

## Installation

> **⚠️ Windows users**: NetSquid does not run natively on Windows. You must use **Ubuntu** or **WSL (Windows Subsystem for Linux)** before proceeding with any of the steps below.

### 1. Install Python 3.12.3

This project requires **Python 3.12.3**. Using a different version may cause compatibility issues with NetSquid.

```bash
sudo apt update
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev -y
```

Verify the installation:

```bash
python3.12 --version
```

You should see `Python 3.12.3`.

### 2. Clone the repository

```bash
git clone https://github.com/anirudhas-2004/NetSquid-FSO-QKD.git
cd NetSquid-FSO-QKD
```

### 3. Create and activate a virtual environment

```bash
python3.12 -m venv venv
source venv/bin/activate
```

### 4. Install NetSquid (requires a free account)

NetSquid is hosted on a private PyPI server. Register for a free account at [netsquid.org](https://netsquid.org) to obtain your username and password, then run:

```bash
pip install "netsquid==1.1.8" --extra-index-url https://<YOUR_USERNAME>:<YOUR_PASSWORD>@pypi.netsquid.org
```

For example, if your username is `johndoe` and your password is `mypassword123`, the command would be:

```bash
pip install "netsquid==1.1.8" --extra-index-url https://johndoe:mypassword123@pypi.netsquid.org
```

### 5. Install remaining dependencies

```bash
pip install -r requirements.txt
```

> **Note**: `requirements.txt` contains all dependencies **except** NetSquid and its own dependencies (numpy, pandas, scipy, pydynaa), which are handled automatically in Step 4.

---

## Quick Start

Run the modern GUI:

```bash
python modern_gui.py
```

This provides an intuitive interface to configure parameters, run simulations, and visualize results.

---

## Reproducing Paper Results

- **Loss model & figures**: Open and run `loss_model_analysis.ipynb`
- **Key rates & full protocol**: Open and run `master_data.ipynb`

---

## Submodules / Embedded Projects

- **[cascade-python](https://github.com/brunorijsman/cascade-python)** — CASCADE error correction (used as-is).
- **[randextract](https://github.com/cryptohslu/randextract)** — Privacy amplification (minor path/formatting adjustments for compatibility).

Adapters (`info_reco.py` and `privacy_amp.py`) integrate these libraries seamlessly into the NetSquid workflow.
