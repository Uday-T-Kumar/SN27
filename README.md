# NI Compute (Subnet 27)

Welcome to the **Bittensor NI Compute Subnet** repository. This subnet powers a decentralized compute market, enabling **miners** to contribute GPU resources and earn rewards in return. **Validators** measure the performance of these miners and allocate GPU resources accordingly, ensuring an efficient, trustless, and permissionless compute market.

---

Welcome to the **Bittensor NI Compute Subnet** repository. This subnet powers a decentralized compute market, enabling **miners** to contribute GPU resources and earn rewards in return. **Validators** measure the performance of these miners and allocate GPU resources accordingly, ensuring an efficient, trustless, and permissionless compute market.

**Bittensor:** [Discord](https://discord.gg/bittensor) • [Network](https://taostats.io/) • [Research](https://bittensor.com/whitepaper)

---

## Table of Contents

1. [Introduction](#introduction)
   - [Decentralizing Compute](#decentralizing-compute)
   - [Subnet 27 and Bittensor](#subnet-27-and-bittensor)
2. [Key Resources](#key-resources)
3. [Miner vs. Validator Overview](#miner-vs-validator-overview)
4. [Installation and Prerequisites](#installation-and-prerequisites)
   - [System Requirements](#system-requirements)
   - [Installation Methods](#installation-methods)
   - [Install Docker](#install-docker)
   - [Prepare Project Repository](#prepare-project-repository)
   - [CUDA Toolkit and GPU Drivers](#cuda-toolkit-and-gpu-drivers)
   - [NVIDIA Docker Support](#nvidia-docker-support)
   - [Install Python Dependencies (includes Bittensor)](#install-python-dependencies-includes-bittensor)
   - [Create/Regenerate Keys](#create-or-regenerate-keys)
   - [WandB Setup](#wandb-setup)
   - [PM2 Setup](#pm2-setup)
5. [Networking and Firewall](#networking-and-firewall)
6. [Registering Your Hotkey](#registering-your-hotkey)
7. [Running a Miner](#running-a-miner)
   - [Miner Options](#miner-options)
   - [Checking Miner Logs](#checking-miner-logs)
8. [Running a Validator](#running-a-validator)
   - [Validator Options](#validator-options)
9. [GPU Scoring](#gpu-scoring)
10. [Resource Allocation Mechanism](#resource-allocation-mechanism)
11. [Network Overview Diagram](#network-overview-diagram)
12. [Troubleshooting](#troubleshooting)
13. [Actions to Update](#actions-to-update)
14. [Reward Program for Contributions](#reward-program-for-contributions)
15. [License](#license)

---

## Introduction

### Decentralizing Compute
**NI Compute** decentralizes GPU resources by weaving siloed pools of compute into a **trustless** market. This allows for scalable, on-demand GPU power, free from the constraints of centralized providers. Miners add GPU instances to the network, and validators ensure the reliability and performance of these resources.

### Subnet 27 and Bittensor
**NI Compute** is powered by the **Bittensor** network. Bittensor is a blockchain that incentivizes resource providers (miners) who contribute machine intelligence services. In SN27, the resource is **GPU-based compute**, which is crucial for training and running machine learning workloads.

---

## Key Resources

- **NI Compute App (Rent GPUs)**
  [Cloud Platform](https://app.neuralinternet.ai/)

- **Subnet 27 (This Repo)**
  [GitHub: neuralinternet/SN27](https://github.com/neuralinternet/SN27)

- **Bittensor**
  [Bittensor Documentation](https://docs.bittensor.com/)

- **Compute Subnet Discord Channel**
  [Join Discord](https://discord.gg/ZpaGVXfaCF)

- **Real-Time Compute Subnet Metrics**
  [OpenCompute Dashboard](https://opencompute.streamlit.app/)

- **Reward Program for Valuable Contributions**
  [CONTRIBUTING.md](https://github.com/neuralinternet/compute-subnet/blob/main/CONTRIBUTING.md)

> **Note**: We do **not** support container-based (dockerized) GPU platforms such as Runpod, VastAI, or Lambda. We strongly encourage providing your own hardware. If you cannot supply hardware in-house, recommended GPU providers include:
> - Oracle
> - Coreweave
> - FluidStack
> - [Latitude.sh](https://latitude.sh/) (referral code: `BITTENSOR27`)
> - [Oblivus](https://app.oblivus.com/) (referral code: `BITTENSOR27` for 2% cashback)

---

## Miner vs. Validator Overview
![Miner Overview Diagram](docs/sn27_miner_overview1.png)

### Miner
- **Role**: Contribute resources (GPU instances) to the network.
- **Rewards**: Performance-based. Higher-performance devices with more GPUs receive higher rewards.
- **Key Requirements**:
  - GPU(s) with up-to-date drivers.
  - Properly opened ports. E.g. 4444 for validator allocations, 8091 for serving the axon, and 27015-27018 for external client access.
  - A registered wallet hotkey on the correct Bittensor subnet (netuid 27 for main, or netuid 15 for test).

### Validator
- **Role**: Verifies the miners' computational integrity, assigns performance scores, and dynamically allocates resources to clients.
- **Actions**:
  - Requests performance data (e.g., GPU type, memory) from miners.
  - Benchmarks or runs tasks to confirm advertised hardware.
  - Updates scores that determine miners’ reward weights.
- **Key Requirements**:
  - A registered wallet hotkey on the correct Bittensor subnet (netuid 27 for main, or netuid 15 for test).
  - Up-to-date code to ensure accurate scoring.

---

## Installation and Prerequisites

### System Requirements
- **Operating System**: Ubuntu 22.04 (recommended) or higher.
- **Python**: 3.10 or higher.
- **GPU**: NVIDIA GPU (recommended).

> **Important**: Each UID is limited to **one external IP**. **Port 4444 is to be opened** for your miner to be allocated properly. Automatic blacklisting occurs for anomalous behavior.

### Installation Methods

#### Option 1: Automated Installation
We provide an automated installation script that handles the setup process, including:
- Docker installation and configuration
- NVIDIA drivers and CUDA setup
- Bittensor installation
- ni-compute configuration
- Miner setup and startup

Note: You will need to create and register your wallet separately using the Bittensor CLI.

To use the automated installation, visit our [installation script repository](https://github.com/neuralinternet/ni-compute/tree/dev/scripts/installation_script) and follow the instructions.

#### Option 2: Manual Installation
If you prefer to install components manually, follow the steps below:

### Install Docker
A Docker environment is required for miner resource allocation:

1. **Install Docker** on Ubuntu:
   [Official Docs](https://docs.docker.com/engine/install/ubuntu)

2. **Verify Docker**:
   ```bash
   docker run hello-world
   ```
   This should display a confirmation message with no errors.

### Prepare Project Repository
1. **Clone the repository**:
   ```bash
   git clone https://github.com/neuralinternet/SN27.git
   cd SN27
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

> **Important**: We'll install the Python dependencies later, after setting up CUDA and NVIDIA Docker support, as some packages require GPU drivers to be installed first.

### CUDA Toolkit and GPU Drivers
> **Tip**: If Nvidia toolkit and drivers are already installed on your machine, scroll down to step 5 to verify then move on to the docker CUDA support.
1. **Download** the latest CUDA from [NVIDIA's official page](https://developer.nvidia.com/cuda-downloads).
2. **Install** (example for Ubuntu 22.04 (Feb. 2024)):
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
   sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
   wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb
   sudo dpkg -i cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb
   sudo cp /var/cuda-repo-ubuntu2204-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
   sudo apt-get update
   sudo apt-get -y install cuda-toolkit-12-8

   sudo apt-get install -y nvidia-open
   ```
3. **Set environment variables**:
   ```bash
   echo "" >> ~/.bashrc
   echo 'export CUDA_VERSION=cuda-12.8' >> ~/.bashrc
   echo 'export PATH="$PATH:/usr/local/$CUDA_VERSION/bin"' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH="/usr/local/$CUDA_VERSION/lib64:$LD_LIBRARY_PATH"' >> ~/.bashrc
   source ~/.bashrc
4. **Reboot** to finalize changes:
   ```bash
   sudo reboot
   ```
5. **Verify** driver and CUDA:
   ```bash
   nvidia-smi
   nvcc --version
   ```
The output of which should look something like:
```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 545.29.06              Driver Version: 545.29.06    CUDA Version: 12.3     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA RTX                     Off | 00000000:05:00.0 Off |                  Off |
| 30%   34C    P0              70W / 300W |  400MiB / 4914000MiB |      4%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Fri_Nov__3_17:16:49_PDT_2023
Cuda compilation tools, release 12.3, V12.3.103
Build cuda_12.3.r12.3/compiler.33492891_0
```

### NVIDIA Docker Support
Enable GPU functionality within Docker containers:

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list \
  | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update
sudo apt-get install -y nvidia-container-toolkit
sudo apt install -y nvidia-docker2
```

### Install Python Dependencies (includes Bittensor)
Now that CUDA and NVIDIA Docker are set up, we can safely install the Python dependencies:

1. **Navigate to the project directory and activate venv** (if not already active):
   ```bash
   cd SN27
   source venv/bin/activate
   ```

2. **Install all dependencies** (this automatically installs the correct Bittensor version):
   ```bash
   pip install -e . -r requirements.txt
   ```

3. **Install additional system dependencies**:
   ```bash
   sudo apt -y install ocl-icd-libopencl1 pocl-opencl-icd
   ```

4. **Verify Bittensor Installation**:
   ```bash
   btcli --help
   ```

   This will give you an output similar to below:
   ```
   usage: btcli <command> <command args>

   bittensor cli v6.9.4

   positional arguments:
     {subnets,s,subnet,root,r,roots,wallet,w,wallets,stake,st,stakes,sudo,su,sudos,legacy,l,info,i}
       subnets (s, subnet)
                           Commands for managing and viewing subnetworks.
       root (r, roots)     Commands for managing and viewing the root network.
       wallet (w, wallets)
                           Commands for managing and viewing wallets.
       stake (st, stakes)  Commands for staking and removing stake from hotkey accounts.
       sudo (su, sudos)    Commands for subnet management
       legacy (l)          Miscellaneous commands.
       info (i)            Instructions for enabling autocompletion for the CLI.

   options:
     -h, --help            show this help message and exit
     --print-completion {bash,zsh,tcsh}
                           Print shell tab completion script
   ```

> **Note**: Bittensor is automatically installed with the correct version when you install the project requirements. The Bittensor repository is: https://github.com/opentensor/bittensor

### Create or Regenerate Keys
Now that Bittensor is installed, you can create your wallet keys:

1. **Create new coldkey** (stores funds):
   ```bash
   btcli w new_coldkey
   ```
2. **Create new hotkey** (used for daily operations e.g. mining/validating/registration):
   ```bash
   btcli w new_hotkey
   ```
3. **Regenerate existing keys** if needed (to import them on this machine):
   ```bash
   btcli w regen_coldkeypub #see below
   btcli w regen_coldkey
   btcli w regen_hotkey
   ```

> **Tip**: For security, you can generate your coldkey on a secure offline machine and only load the public portion onto your miner or validator servers.

### WandB Setup
1. **Create a free WandB account**: [wandb.ai](https://wandb.ai/)
2. **Obtain an API Key** and place it into your `.env` file:
   ```bash
   cd SN27
   read -p "Enter WanDB API key: " wandb_api_key && sed -e s/"your_api_key"/"${wandb_api_key}"/ .env.miner > .env
3. **Monitor stats** at [WandB: neuralinternet/opencompute](https://wandb.ai/neuralinternet/opencompute)

### PM2 Setup
**PM2** Proccess manager is great for monitoring the operation and helps keep your miner or validator running persistently.

```bash
sudo apt update
sudo apt install npm -y
sudo npm install pm2 -g
pm2 ls
```

---

## Networking and Firewall

Your miner requires **multiple essential ports** to be opened:

- **Port 4444 (SSH)**: Used by validators to access your miner for PoG (Proof of GPU) validation. Validators verify GPU functionality, available resources, and hardware specs through this port. **Required for miners to appear in the network.**
- **Port 8091 (Axon)**: Used for Bittensor validator-miner communication. **Critical for network functionality.**
- **Ports 27015-27018 (External Ports)**: Multiple external ports that clients can use for their own purposes during allocations. **Validators verify these ports are accessible - if not open, miners will not appear in the dashboard or pass validation requirements.** Default ports are 27015, 27016, 27017, and 27018, but these can be customized.

### Port Validation Tool

Before configuring your firewall, we recommend using our **port validation script** to test if your ports are accessible from the internet:

```bash
python3 scripts/validate_miner_ports.py
```

**⚠️ IMPORTANT: Ensure the ports you're testing are NOT being used by other services (including your miner) before running this validation script. The script needs to bind to these ports to test accessibility.**

This tool will:
- ✅ **Test external accessibility** of all required ports (SSH, Axon, and multiple external ports)
- ✅ **Check firewall configuration** (UFW status and rules)
- ✅ **Detect NAT/router issues** for home networks
- ✅ **Provide specific troubleshooting steps** based on your setup

**Custom port testing:**
```bash
python3 scripts/validate_miner_ports.py --ssh-port 4444 --axon-port 8091 --external-ports 27015,27016,27017,27018
```

The validator provides real-time feedback and troubleshooting guidance for:
- Cloud platforms requiring security group configuration
- Home networks requiring port forwarding
- Firewall configuration issues

### Configure UFW Firewall

1. **Install and configure `ufw`**:
   ```bash
   sudo apt install ufw
   sudo ufw allow 4444       # SSH port for PoG validation
   sudo ufw allow 22/tcp     # Standard SSH
   sudo ufw allow 8091/tcp   # Axon port - can be customized
   sudo ufw allow 27015/tcp  # External port 1
   sudo ufw allow 27016/tcp  # External port 2
   sudo ufw allow 27017/tcp  # External port 3
   sudo ufw allow 27018/tcp  # External port 4
   sudo ufw enable
   sudo ufw status
   ```

2. **Verify your configuration** by running the validation script again:
   ```bash
   python3 scripts/validate_miner_ports.py
   ```

> **Custom Ports**: You can use different ports by opening them with `sudo ufw allow XXXX/tcp` and specifying them in your miner configuration with `--axon.port XXXX`, `--ssh.port XXXX`, or `--external.ports XXXX,YYYY,ZZZZ,WWWW`. If using a cloud server, ensure these ports are also opened in your provider's firewall/security groups.

3. **Add user to docker group** (if not already):
   ```bash
   sudo groupadd docker
   sudo usermod -aG docker $USER
   sudo systemctl start docker
   sudo apt install at  # ensures certain dependencies for Docker
   sudo service docker status  # should show 'active (running)'
   ```

---

## Registering Your Hotkey
You need **$TAO** tokens in your coldkey to register the hotkey on the chosen netuid.

- **Main Network** (netuid = 27, known as `finney`):
  ```bash
  btcli s register --subtensor.network finney --netuid 27
  ```
- **Test Network** (netuid = 15, known as `test`):
  ```bash
  btcli s register --subtensor.network test --netuid 15
  ```

> If you get the error `too many registrations this interval`, wait for the next interval and retry.
> **Registration cost** can be checked [Here](https://taostats.io/subnets/netuid-27/#registration). Or check using the CLI w/ `btcli s list` (Cost is 'RECYCLE').

---

## Running a Miner
The miner contributes GPU resources to the network. Make sure your ports are open and your environment is set up as described above.

**General Miner Command**:
```bash
pm2 start ./neurons/miner.py --name <MINER_NAME> --interpreter python3 -- \
  --netuid 27 \
  --subtensor.network finney \
  --wallet.name <COLDKEY_NAME> \
  --wallet.hotkey <HOTKEY_NAME> \
  --axon.port 8091 \
  --logging.debug
```

- **`--netuid`**: Subnet ID (27 for main, 15 for test).
- **`--subtensor.network`**: Your Bittensor chain endpoint.
  - **Main**: `finney`
  - **Test**: `test`
  - Or use a custom endpoint, e.g. `subvortex.info:9944` (recommended)
- **`--wallet.name`** & **`--wallet.hotkey`**: The coldkey/hotkey names you created [above](#create-or-regenerate-keys) and used in registration (btcli's defaults are `default` and `default` but both can be freely customized)
- **`--axon.port`**: default 8091 can be replaced with any port number allowed by ufw as instructed [above](#networking-and-firewall) to serve your axon. Important for proper functionality and miner<->validator communication.
- **`--ssh.port`**: A port opened with UFW as instructed [above](#networking-and-firewall) (e.g., 4444) used for allocating your miner via ssh.
- **`--external.ports`**: Comma-separated list of external ports opened with UFW as instructed [above](#networking-and-firewall) (default: 27015,27016,27017,27018) that clients can use for their own purposes during allocations. Required for validation.
- **`--auto-update`**: Enables automatic updating of the miner. When enabled, the miner will internally perform the update process (e.g., running `git pull`, installing dependencies, and restarting via PM2) so that no manual action is required.

### Configuration File Usage (Alternative to Long CLI Commands)

Instead of using long PM2 commands with many flags, you can use a configuration file:

#### Option 1: Command Line Flags (Current)
```bash
pm2 start ./neurons/miner.py --name MINER --interpreter python3 -- \
  --netuid 27 --subtensor.network finney --wallet.name default --wallet.hotkey default \
  --axon.port 8091 --ssh.port 4444 --external.ports 27015,27016,27017,27018 --logging.debug --auto_update yes
```

#### Option 2: Configuration File (Recommended)
1. Copy the example config: `cp miner.config.example.json miner.config.json`
2. Edit `miner.config.json` with your settings
3. Run: `pm2 start ./neurons/miner.py --name MINER --interpreter python3 -- --config miner.config.json`

**Benefits:** Cleaner commands, easier management, less error-prone.

### Miner Options
- `--miner.whitelist.not.enough.stake`: Whitelist validators lacking sufficient stake (default: False).
- `--miner.whitelist.not.updated`: Whitelist validators not on the latest code version (default: False).
- `--miner.whitelist.updated.threshold`: Quorum threshold (%) before starting the whitelist (default: 60).

### Checking Miner Logs
To inspect your miner’s logs:

```bash
pm2 logs
pm2 monit
```
- **`pm2 logs`** shows rolling log output.
- **`pm2 monit`** provides a live interface with CPU usage, memory, etc.

> Run pm2 logs to debug any issues and view information on your miner. **Specifically find your wandb run to view more live information. In the wandb project you can view the scores you receive from validators.**

![Successful Validator<>Miner PoG](docs/pm2_miner_logs_pog_container.png)
Successful Validator<>Miner PoG
![Check Miner Score Wandb 1](docs/wandb_logs_validator_run1.png)
Check Miner Score Wandb Step 1: Find a Validator run
![Check Miner Score Wandb 2](docs/wandb_logs_validator_run2.png)
Check Miner Score Wandb Step 2: Search your miner UID or Hotkey
![Check Miner Score Wandb 3](docs/wandb_logs_validator_run3.png)
Check Miner Score Wandb Step 3: Check Stats/Score

---

**Note**: Validators are required to operate on a local Subtensor instance (subvortex recommended), as using Finney may result in rate limitations.

## Running a Validator
Validators measure and score miner performance, adjusting on-chain weights accordingly.

**Environment Setup for Validators**: Validators use `.env.validator` for additional configuration variables not needed by miners.

**General Validator Command**:
```bash
pm2 start ./neurons/validator.py --name <VALIDATOR_NAME> --interpreter python3 -- \
  --netuid 27 \
  --subtensor.network finney \
  --wallet.name <COLDKEY_NAME> \
  --wallet.hotkey <HOTKEY_NAME> \
  --logging.debug
```

### Validator Options
- `--validator.whitelist.unrecognized`: Whitelist unrecognized miners (default: False).
- `--validator.perform.hardware.query`: Gather hardware specs from miners (default: True).
- `--validator.specs.batch.size`: Batch size for specs queries (default: 64).
- `--validator.force.update.prometheus`: Force upgrade of Prometheus if needed (default: False).
- `--validator.whitelist.updated.threshold`: Quorum threshold (%) before starting the whitelist (default: 60).
- `--auto-update`: Enables automatic updating of the validator. When enabled, the validator will internally perform the update process (e.g., running `git pull`, installing dependencies, and restarting via PM2) so that no manual action is required.

---

## GPU Scoring
**Subnet 27** uses a performance-based scoring system centered on GPU hardware. Below are **example base scores**:

| GPU Model                        | Base Score |
|---------------------------------|------------|
| NVIDIA H200                      | 4.00       |
| NVIDIA H100 80GB HBM3           | 3.30       |
| NVIDIA H100                      | 2.80       |
| NVIDIA A100-SXM4-80GB           | 1.90       |

1. **Base GPU Score**: Tied to the GPU model.
2. **Scaling**: Up to 8 GPUs can be recognized. The top theoretical scenario (8 of the highest GPU model) is set to 50 points.

---

## Resource Allocation Mechanism
Validators reserve resources from miners by specifying required CPU, GPU count, memory, etc. The subnet dynamically allocates and deallocates miner resources based on **availability** and **network demands**. A validator can request resources from a miner using scripts such as [neurons/register.py](neurons/register.py). Example resource request:
```
{"cpu":{"count":1}, "gpu":{"count":1}, "hard_disk":{"capacity":10737418240}, "ram":{"capacity":1073741824}}
```

---
## Network Overview Diagram
![Network Overview Diagram](docs/sn27_networkoverview1.png)

## Troubleshooting
- **No requests received (no ‘Challenge’ or ‘Specs’ events)**:
  - Check your open ports (default allocation port: 4444). Check your Axon port is open with your machine or cloud provider. Use `pm2 describe <PROCCESS_NAME>` and `pm2 show <PROCCESS_NAME>` to view the arguments you used to run your miner e.g. `--axon.port` and `--ssh.port` and check with `sudo ufw status` that the right ports are open with UFW as well.
  - Check your pm2 logs for any errors or tracebacks to help troubleshoot.
  - Ensure the miner is running properly and not blacklisted.
- **Deregistered unexpectedly**:
  - Competition on the network is high; more powerful devices may outcompete you.
  - Connection or environment issues.
  - Make sure scripts and Docker containers are running stably.

---
## Actions To Update

__**No action required when using auto-update flag**__.

```sh
git pull
python -m pip install -r requirements.txt
python -m pip install -e .
pm2 restart <id>
```

## Verify Installation

You can verify the installation and check the version by running:

```sh
pip show NI-Compute
```

Example output:

```
Name: nicompute
Version: 1.8.3
Summary: nicompute-subnet27
Home-page: https://github.com/neuralinternet/SN27
License: MIT
```

---

## Reward Program for Contributions
We encourage community involvement in improving **Compute Subnet**. A **bounty program** is in place to reward valuable contributions.
See the **[Reward Program for Valuable Contributions](https://github.com/neuralinternet/SN27/blob/main/CONTRIBUTING.md)** for details.


---

## License

```
The MIT License (MIT)
© 2023 Neural Internet

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the “Software”), to deal in
the Software without restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
