![image](https://github.com/user-attachments/assets/8ad5a694-e287-4d45-ba57-203f58a19714)

# Run RL Swarm (Testnet) Node
RL Swarm is a fully open-source framework developed by GensynAI for building reinforcement learning (RL) training swarms over the internet. This guide walks you through setting up an RL Swarm node and a web UI dashboard to monitor swarm activity.

## Hardware Requirements
Currently in the new recent update, Gensyn testnet is running and training the [reasoning-gym](https://github.com/open-thought/reasoning-gym/tree/main) swarm datasets on the Testnet. This swarm is supporting the current list of default models:
* Gensyn/Qwen2.5-0.5B-Instruct
* Qwen/Qwen3-0.6B
* nvidia/AceInstruct-1.5B
* dnotitia/Smoothie-Qwen3-1.7B
* Gensyn/Qwen2.5-1.5B-Instruct

Your hardware requirements will vary depending on model you choose. Users with less powerful hardware should select a smaller model (e.g. `Qwen2.5-0.5B` or `Qwen3-0.6B`). Users with more powerful hardware can select a larger models.

### CPU & GPU support
* `CPU-only`: arm64 or x86 CPU with minimum `32gb` ram (note that if you run other applications during training it might crash training).

OR

* `GPU`: 
  * RTX 3090
  * RTX 4090
  * RTX 5090
  * A100
  * H100
  * `≥24GB vRAM` GPU is recommended, but Gensyn now supports `<24GB vRAM` GPUs too.
  * `≥12.4` CUDA Driver.

#

* This guide is going through the easiet default way to participate on testnet, you can checkout [Official Repo](https://github.com/gensyn-ai/rl-swarm/tree/main) for more details.

#

## Enviorements

## Method 1 - Windows Users (Home PC):
If you are a windows user, you may need to install `Ubuntu` on your windows.
* Install Ubuntu on Windows: [Guide](https://github.com/0xmoei/Install-Linux-on-Windows)
* After you installed `Ubuntu` on Windows, Verify you already have `NVIDIA Driver` & `CUDA Toolkit` ready:
```console
# Install NVIDIA Toolkit
sudo apt-get update
sudo apt-get install -y nvidia-cuda-toolkit

# Verify NVIDIA Driver
nvidia-smi

# Verify CUDA Toolkit:
nvcc --version
```

#

## Method 2 - Rent Cloud GPU:
You can use [Rent and Config GPU Guide](https://github.com/0xmoei/Rent-and-Config-GPU) to get a fully detailed guide or follow the steps below.
### 1- Rent Vast.ai GPUs
* 1- Register in [Vast.ai](https://cloud.vast.ai/?ref_id=228875)
* 2- Create ssh key in your local system (If you don't have already) with this [Guide: step 1](https://github.com/0xmoei/Rent-and-Config-GPU)
* 3- Create an SSH key in Vast.ai by going to `three-lines > Keys > SSH Keys` [here](https://cloud.vast.ai/manage-keys/)
* 4- Paste SSH public key created in your local pc in step 2.
* 5- Select Pytorch(Vast) template [here](https://cloud.vast.ai/?ref_id=62897&creator_id=62897&name=PyTorch%20(Vast))
* 6- Choose a supported GPU (I recommend =24GB GPU vRAM, but Gensyn now supporting even 8GB GPU vRAM)
* 7- Increase `Disk Space` slidebar to `50GB`
* 8- Top-up credits with crypto and rent it.
* 9- Go to [instances](https://cloud.vast.ai/instances/), refresh the page, click on `key` button.
* 10- If you don't see a ssh command in `Direct ssh connect:` section, then you have to press on Add SSH Key.
* 11- Copy SSH Command, and Replace `-L 3000:localhost:3000` in front of the command.
* 12- Enter the command in `Windows Powershell` and run it.
* 13- It prompts you for your ssh public key password (if you set before), then your GPU terminal appears.

---

### 2- Rent [QuickPod](https://console.quickpod.io?affiliate=f621de18-b6ac-4416-b87f-01f29f8339b5) GPUs: Cheap, No SSH-key needed
  * Visit **[QuickPod](https://console.quickpod.io?affiliate=f621de18-b6ac-4416-b87f-01f29f8339b5)**
  * Signup and verify your email in inbox
  * Deposit crytocurrency by clicking on  `Add` in the top right
  * Click on `Templates` and then find `Cuda 12.4`
  * It redirects you to `Search Console` section to select your GPU nd click on `Create Pod`
  * Recommended: Filter 4090/3090 GPUs and Sort by cheapest price to find the most affordable 3090/4090s
  * You can even rent lower-end GPUs for cheaper price (12GB vRAM supported by Gensyn)
  * Go to `Pods` section and wait until your GPU be deployed
  * Click on `Connect` and choose one of two options below:
  * 1- `Connect to web`: To redirect you to a web based terminal of your GPU
  * 2- `SSH Command`: Copy the SSH command and Execute it in a terminal in your system (e.g. Windows Powershell, Mobaxterm, Termius)

---

### 3- Rent Hyperbolic GPUs
* 1- Register In [Hyperbolic Dashboard](https://app.hyperbolic.xyz/invite/gqYoHbUk7)
* 2- then, Visit **Settings**
* 3- Create a new Public SSH key and paste your pubkey into it and save it!
* 4- Choose a GPU (.eg RTX 4090) [here](https://app.hyperbolic.xyz/invite/gqYoHbUk7) by going to `Home > GPU List ` and click on `Rent`
* 5- Make sure you select `1` as `GPU Count`.
* 6- Select `pytorch` as `Template`.
* 7- Rent it.
* 8- By clicking on your gpu instance, if gives you a SSH command to connect to your GPU terminal.
* 9- Add this flag: `-L 3000:localhost:3000` in front of your Hyperbolic's SSH command, this will allow you to access to port 3000 on your local system.
* 10- Paste and Enter the command you copied in `Windows PowerShell` to access your server.
* 11- It prompts you for your ssh public key password (if you set before), then your GPU terminal appears.

---

### Method 3: VPS servers
While i recommend to use GPU, I am currently running `CPU only` of this node's version successfully on a VPS with 16core, 64GB RAM.
* Official recommended hardware for `CPU-only`: arm64 or x86 CPU with minimum `32gb` ram.
* I recommend to buy a VPS from [Hostbrr](https://my.hostbrr.com/order/forms/a/NTMxNw==) and for begin.
* For beginners, you can learn to buy & set up your VPS via this [detailed guide](https://github.com/0xmoei/Linux_Node_Guide/).

#

## 1) Install Dependencies
**1. Update System Packages**
```bash
apt update && apt upgrade -y
```
**2. Install General Utilities and Tools**
```bash
apt install screen curl iptables build-essential git wget lz4 jq make gcc nano automake autoconf tmux htop nvme-cli libgbm1 pkg-config libssl-dev libleveldb-dev tar clang bsdmainutils ncdu unzip libleveldb-dev  -y
```

**3. Install Python**
```bash
apt install python3 python3-pip python3-venv python3-dev -y
```

**4. Install Node**
```
apt update
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
apt install -y nodejs
node -v
npm install -g yarn
yarn -v
```

**5. Install Yarn**
```bash
curl -o- -L https://yarnpkg.com/install.sh | bash
```
```bash
export PATH="$HOME/.yarn/bin:$HOME/.config/yarn/global/node_modules/.bin:$PATH"
```
```bash
source ~/.bashrc
```

---

## 2) Get HuggingFace Access token
**1- Create account in [HuggingFace](https://huggingface.co/)**

**2- Create an Access Token with `Write` permissions [here](https://huggingface.co/settings/tokens) and save it**

---

## 3) Clone the Repository
```bash
git clone https://github.com/gensyn-ai/rl-swarm/
```

---

## 4) Run the swarm
* If you are an **existing user**, you must have your node's `swarm.pem` present in `rl-swarm` directory before starting the node, follow [Recover](#recover) instructions if need to recover `swarm.pem` file
* Use on of the `Bash` or `Docker` methods to run your node

### Bash Method (GPU)
1- Open a screen to run it in background
```bash
screen -S swarm
```
2- Get into the `rl-swarm` directory
```
cd rl-swarm
```
3- Install swarm
```bash
python3 -m venv .venv

source .venv/bin/activate
# if not worked, then: . .venv/bin/activate

./run_rl_swarm.sh
```

### Using Docker (GPU, Mac, CPU)
* GPU cloud users:
  * This method is only available on GPU providers supporting `Ubuntu VM` templates like [Vast](https://cloud.vast.ai/?ref_id=62897&creator_id=62897&name=Ubuntu%2022.04%20VM).
  * If you are on Quickpod, Hyperbolic, etc., use **Bash Method**.
 * A good option for Mac users or CPU-only VPS servers.

1- Install docker, docker-compose with this [guide](https://github.com/0xmoei/Linux_Node_Guide/blob/main/linux-config.md#docker-docker-compose)

2- Create a screen session
```bash
screen -S swarm
```

3- Get into the `rl-swarm` directory
```
cd rl-swarm
```

4- Install swarm
* Mac or CPU-Only
```bash
docker compose run --rm --build -Pit swarm-cpu
```

* GPU
```bash
docker compose run --rm --build -Pit swarm-gpu
```

---

## 5) Login
**1- You have to receive `Waiting for userData.json to be created...` in logs**

![image](https://github.com/user-attachments/assets/140f7d32-844f-4cf0-aac4-a91e9a14c1aa)

**2- Open login page in browser**
* **Local PC:** Open `http://localhost:3000/` in your browser
* **GPU Cloud & VPS Users: Tunnel to external URL:**
  * 1- Open a new terminal
  * 2- Install **localtunnel**:
    ```
    npm install -g localtunnel
    ```
  * 3- Get a password:
    ```
    curl https://loca.lt/mytunnelpassword
    ```
  * The password is actually your VPS IP
  * 4- Get URL
    ```
    lt --port 3000
    ```
  * Visit the prompted url, and enter your password to access Gensyn login page

**3- Login with your preferred method**

![image](https://github.com/user-attachments/assets/f33ea530-b15f-4af7-a317-93acd8618a9f)

* After login, your terminal starts installation.

**4- Answer prompts:**
* `Would you like to push models you train in the RL swarm to the Hugging Face Hub? [y/N]` >>> Press `N` to join testnet
  * `HuggingFace` needs `2GB` upload bandwidth for each model you train, you can press `Y`, and enter your access-token.
* `Enter the name of the model you want to use in huggingface repo/name format, or press [Enter] to use the default model.` >>> For default model, press `Enter`  or choose one of these (More model parameters (B) need more vRAM):
  * `Gensyn/Qwen2.5-0.5B-Instruct`
  * `Qwen/Qwen3-0.6B`
  * `nvidia/AceInstruct-1.5B`
  * `dnotitia/Smoothie-Qwen3-1.7B`
  * `Gensyn/Qwen2.5-1.5B-Instruct`

---

### Node Name
* Now your node started running, Find your name after word `Hello`, like mine is `whistling hulking armadillo` as in the image below (You can use `CTRL+SHIFT+F` to search Hello in terminal)

![image](https://github.com/user-attachments/assets/a1abdb1a-aa11-407f-8e5b-abe7d0a6b0f3)

---

### Screen commands
* Minimize: `CTRL` + `A` + `D`
* Return: `screen -r swarm`
* Stop and Kill: `screen -XS swarm quit`

---

## Backup
**You need to backup `swarm.pem`**.
### `VPS`:
Connect your VPS using `Mobaxterm` client to be able to move files to your local system. Back up these files:**
* `/root/rl-swarm/swarm.pem`

### `WSL`:
Search `\\wsl.localhost` in your ***Windows Explorer*** to see your Ubuntu directory. Your main directories are as follows:
* If installed via a username: `\\wsl.localhost\Ubuntu\home\<your_username>`
* If installed via root: `\\wsl.localhost\Ubuntu\root`
* Look for `rl-swarm/swarm.pem`

### `GPU servers (e.g., Vast, Hyperbolic)`:
**1- Connect to your GPU server by entering this command in `Windows PowerShell` terminal**
```
sftp -P PORT ubuntu@xxxx.hyperbolic.xyz
```
* Replace `ubuntu@xxxx.hyperbolic.xyz` with your given GPU hostname
* Replace `PORT` with your server port (in your server ssh connection command)
* `ubuntu` is the user of my hyperbolic gpu, it can be ***anything else*** or it's `root` if you test it out for `vps`

Once connected, you’ll see the SFTP prompt:
```
sftp>
```

**2- Navigate to the Directory Containing the Files**
 * After connecting, you’ll start in your home directory on the server. Use the `cd` command to move to the directory of your files:
 ```
 cd /home/ubuntu/rl-swarm
 ```

**3- Download Files**
 * Use the `get` command to download the files to your `local system`. They’ll save to your current local directory unless you specify otherwise:
 ```
 get swarm.pem
 ```
* Downloaded file is in the main directory of your `Powershell` or `WSL` where you entered the sFTP command.
  * If entered sftp command in `Porwershell`, the `swarm.pem` file might be in `C:\Users\<pc-username>`.
* You can now type `exit` to close connection. The files are in the main directory of your `Powershell` or `WSL` where you entered the first SFTP command.

---

### Recover
If you need to upload files from your `local machine` to the `server`.
* `WSL` & `VPS`: Drag & Drop option.

**`GPU servers (.eg, Hyperbolic)`:**

1- Connect to your GPU server using sFTP

2- Upload Files Using the `put` Command:

In SFTP, the put command uploads files from your local machine to the server. 
```bash
put swarm.pem /home/ubuntu/rl-swarm/swarm.pem
```

---

# Node Health
### Official Dashboards
https://dashboard.gensyn.ai/

![image](https://github.com/user-attachments/assets/cd8e8cd3-f057-450a-b1a2-a90ca10aa3a6)

### Contract
Query your Node's peer ID for eoa address, rewards, wins, etc.
https://gensyn-testnet.explorer.alchemy.com/address/0xFaD7C5e93f28257429569B854151A1B8DCD404c2?tab=read_proxy

---

# Update Node
### 1- Stop Node
```console
# list screens
screen -ls

# kill swarm screens (replace screen-id)
screen -XS screen-id quit

# You can kill by name
screen -XS swarm quit
```

### 2- Update Node Repository
**Method 1** (test this first): If you cloned official repo with no local changes:
```bash
cd rl-swarm
git pull
```

**Method 2**: If you cloned official repo with local Changes:
```console
cl rl-swarm

# Reset local changes:
git reset --hard
# Pull updates:
git pull

# Alternatively:
git fetch
git reset --hard origin/main
```
* You have to do your local changes again.

**Method 3**: Cloned unofficial repo or Try from scratch (**Recommended**):
```console
cd rl-swarm

# backup .pem
cp ./swarm.pem ~/swarm.pem

cd ..

# delete rl-swarm dir
rm -rf rl-swarm

# clone new repo
git clone https://github.com/gensyn-ai/rl-swarm

cd rl-swarm

# Recover .pem
cp ~/swarm.pem ./swarm.pem
```
* If you had any local changes, you have to do it again.

### 3- Re-run Node
Head back to [4) Run the swarm](https://github.com/0xmoei/gensyn-ai/blob/main/README.md#4-run-the-swarm) and re-run Node.

---

# Troubleshooting:

### CPU Configuration
Fix 1:
```
cd rl-swarm

nano hivemind_exp/configs/mac/grpo-qwen-2.5-0.5b-deepseek-r1.yaml
```
* Change `bf16` value to `false`
* Reduce `max_steps` to `5`

Fix 2: 
Use this as a run command instead:
```
python3 -m venv .venv
source .venv/bin/activate
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 && ./run_rl_swarm.sh
```

### ⚠️ Stuck at loading localhost page
```bash
cd rl-swarm

sed -i '/^  return (/i\  useEffect(() => {\n    if (!user && !signerStatus.isInitializing) {\n      openAuthModal();\n    }\n  }, [user, signerStatus.isInitializing]);\n\n' modal-login/app/page.tsx
```

### ⚠️ Error: PS1 unbound variable
```
sed -i '1i # ~/.bashrc: executed by bash(1) for non-login shells.\n\n# If not running interactively, don'\''t do anything\ncase $- in\n    *i*) ;;\n    *) return;;\nesac\n' ~/.bashrc
```

### ⚠️ Daemon failed to start in 15.0 seconds
* Enter `rl-swarm` directory:
```bash
cd rl-swarm
```
* Activate python venv:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
* Open Daemon config file:
```
nano $(python3 -c "import hivemind.p2p.p2p_daemon as m; print(m.__file__)")
```
* Search for line: `startup_timeout: float = 15`, then change `15` to `120` to increate the Daemon's timeout. the line should look like this: `startup_timeout: float = 120`
* To save the file: Press `Ctrl + X`, `Y` & `Enter`
