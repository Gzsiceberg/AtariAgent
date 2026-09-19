# Vast.ai worker setup

Use fresh Ubuntu 24.04 instances with root SSH access and Python installed. Never run setup over an existing jobd worker.

## Agent preparation

1. On the current host, ensure Ansible, jobd, and authenticated `gh` are available.
2. Generate `/root/.ssh/atari_deploy` on each instance and register its public key as a **read-only** GitHub deploy key. Keep private keys on their instances. Add GitHub's published SSH host keys to each instance's `known_hosts`.
3. Ask the user for the worker-token duration. Create one shared token on the current host:

   ```bash
   umask 077
   state="$HOME/.local/state/atariagent/provision"
   mkdir -p "$state"
   chmod 700 "$state"
   jobd auth create-worker-token --duration USER_CHOSEN_DURATION > "$state/worker-token"
   chmod 600 "$state/worker-token"
   ```

   Never copy `JOBD_MASTER_KEY` to instances. Supply `WANDB_API_KEY` through jobd queue secrets, not SSH.
4. Copy `vastai/inventory.example.yml` to `$state/inventory.yml` and fill in SSH addresses and ports. Ensure the desired bootstrap changes are pushed to `main`.

## Run Ansible

```bash
ansible-playbook -i "$state/inventory.yml" vastai/setup.yml
```

Ansible installs system dependencies, clones `main`, installs uv, and launches bootstrap with **async**. Each instance progresses independently. Bootstrap syncs dependencies, checks OpenCV/CUDA, installs latest jobd, verifies the token, then starts the worker.

Ansible reports success or failure. Logs stay in `/root/bootstrap.log` on each instance. It also prints async job IDs for manual `async_status` checks if disconnected. Do not rerun setup blindly after a failure or disconnect.

To add an instance, add its SSH details to the inventory and use `--limit new_worker_name`.

## Verify W&B authentication

Submit authentication-only jobs from the current host:

```bash
jobd bash -c 'cd /workspace/AtariAgent && exec /root/.local/bin/uv run --no-sync python scripts/verify_wandb_auth.py'
```

Check successful execution on every instance; shared-queue jobs may land on the same worker. The test creates no W&B run and removes temporary login files.

Reusing a token preserves its original expiry. Token expiry does **not** stop rental billing.
