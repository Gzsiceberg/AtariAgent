# Vast.ai worker setup

Use the existing Vast CLI and Ansible playbooks. No provisioning daemon or billing
watchdog is needed. Fresh Ubuntu 24.04 instances only; never prepare over an
existing jobd worker. **Preparation and activation are separate.**

## 1. Configure your controller and keep one private inventory

Use your own [jobd](https://github.com/Gzsiceberg/jobd) HTTPS controller. There is
no AtariAgent controller default. Export the same endpoint and queue used to
create worker tokens and submit jobs; the example inventory reads the endpoint
from `JOBD_CONTROLLER`.

```bash
export JOBD_CONTROLLER="https://YOUR_JOBD_CONTROLLER"
export JOBD_QUEUE=default
```

Authenticate the submitting host with your own jobd admin credentials. Never send
those credentials to workers. Setup and activation reject a missing or non-HTTPS
controller before making remote changes. A custom inventory must set
`jobd_controller` explicitly (or pass it with `-e`).

```bash
umask 077
state="$HOME/.local/state/atariagent/provision"
mkdir -p "$state"
chmod 700 "$state"
# First use only; never overwrite an existing inventory.
test -e "$state/inventory.yml" || cp vastai/inventory.example.yml "$state/inventory.yml"
chmod 600 "$state/inventory.yml"
```

Add each new instance to this inventory rather than creating another inventory.
Record its instance ID, SSH endpoint, preparation status and token expiry. Do not
put token values in inventory. Always use `--limit` when running playbooks against
new workers. Keep credentials, rental responses and provisioning logs in this
private directory, never in Git.

## 2. Rent safely

Check availability and price **including 30 GB storage** immediately before renting.
If a broad search misses an offer, recheck its known `machine_id`; do not assume it
is unavailable. Review CPU performance and verification status as described in
`vastai_instances.md`. Never substitute offers without approval.

For each approved offer, give the rental a unique label and save that label before
submitting. Capture the rental response directly to a private file: it contains
an instance API key. Print only the success flag and new instance ID.

```bash
# Set OFFER to the user-approved offer ID. Run once, not in an automatic retry loop.
(
set -euo pipefail
set -C  # Refuse to overwrite an earlier rental intent or response.
umask 077
label="atari-${OFFER}-$(date +%s)"
printf '%s\n' "$label" > "$state/rental-${OFFER}.label"
vastai create instance "$OFFER" \
  --image nvidia/cuda:13.0.2-devel-ubuntu24.04 --disk 30 --ssh --direct \
  --cancel-unavail --label "$label" \
  --onstart-cmd 'apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3' \
  --raw > "$state/rental-${OFFER}.json" 2> "$state/rental-${OFFER}.err"
uv run python - "$state/rental-${OFFER}.json" <<'PY'
import json, sys
with open(sys.argv[1]) as file:
    result = json.load(file)
print({key: result.get(key) for key in ('success', 'new_contract')})
PY
)
```

After a timeout, malformed response or client error, **do not retry creation**.
Inspect account instances for the saved unique label and record any resulting
instance ID first. A failed client response does not mean rental failed. Do not
print raw rental responses or delete their records to start over.

Attach the submitting machine's SSH public key to the new instance before waiting
for SSH. Add its endpoint to the existing inventory immediately.

## 3. Check SSH before provisioning

Use `BatchMode=yes`, a short `ConnectTimeout`, and `StrictHostKeyChecking=accept-new`.
Make an explicit SSH probe and inspect its error before launching Ansible:

- Timeout/refused while the instance is loading: wait and retry within a deadline.
- Authentication failure: check the attached public key; do not blindly retry.
- Missing Python: inspect the instance's startup installation.
- **Changed host key: stop immediately.** Independently verify its fingerprint or
  ask whether to stop/destroy the instance. Never automatically remove known keys
  or use `StrictHostKeyChecking=no`.

New host keys use trust on first use; Vast's verification status is not SSH identity
verification. Do not suppress SSH stderr in a readiness loop.

## 4. Prepare, without starting training

On the submitting host, ensure `ansible-playbook` and `jobd` are available.
Commit and push the intended bootstrap revision to `main` first.

The public repository is cloned over HTTPS using
`https://github.com/Gzsiceberg/AtariAgent.git`. No GitHub login, deploy key, or
GitHub SSH host-key setup is required. Existing private inventories should change
`repo_url` from the SSH URL to this HTTPS URL. SSH access to the Vast.ai instance
itself is still required.

1. Confirm your inventory's `jobd_controller` and `jobd_queue` match the submitting
   host's `JOBD_CONTROLLER` and `JOBD_QUEUE`.
2. Ask for the worker-token lifetime, then generate a new token on the submitting
   host. Use a distinct private file for each batch so existing credentials are
   not overwritten:

   ```bash
   token_file="$state/worker-token-$(date +%Y%m%d_%H%M%S)"
   jobd auth create-worker-token --duration USER_CHOSEN_DURATION > "$token_file"
   chmod 600 "$token_file"
   ```

   Record its expiry alongside the corresponding inventory entries. Never copy
   `JOBD_MASTER_KEY` to workers. W&B authentication is supplied through jobd queue
   secrets; **no W&B verification step is required**.

```bash
ansible-playbook -i "$state/inventory.yml" vastai/setup.yml \
  --limit new_worker_names -e "worker_token_file=$token_file"
```

Setup installs dependencies, clones `main`, and runs bootstrap asynchronously.
Bootstrap checks CUDA and the token and records the prepared Git revision. It
**does not start jobd**. Logs stay at `/root/bootstrap.log`; Ansible reports async
IDs. After a disconnect, inspect that async job before rerunning—do not launch
overlapping preparation. Mark successful instances prepared in the inventory.

## 5. Activate explicitly and ensure a stop job

```bash
ansible-playbook -i "$state/inventory.yml" vastai/activate.yml \
  --limit prepared_worker_names -e "worker_token_file=$token_file"
```

Activation checks the prepared revision and token and refuses an already-running
worker. It starts jobd without restarting anything, enables persistent local jobs,
and runs `queue_stop.yml`. Workers may immediately consume queued training jobs.
Existing workers are not restarted to change persistence settings.

Stop scheduling is safe to rerun:

```bash
ansible-playbook -i "$state/inventory.yml" vastai/queue_stop.yml --limit worker_names
```

It checks the live JSON local queue under a file lock and skips an existing queued
or running stop command for that instance. Inspection errors fail closed; uncertain
submissions are not automatically retried. The reported job IDs can be recorded in
the inventory. Existing duplicates are reported, not deleted. Use this playbook
for all self-stop submissions so callers share the lock.

## If any step fails

Report the instance ID, failed stage and continued billing immediately. Ask the
user to choose **keep running, stop, or destroy**. Never leave the billing consequence
implicit and never destroy an instance without approval. Proceed independently
with other healthy instances; preserve successful setup rather than restarting it.

- Stop preserves disks; storage charges continue.
- Destroy permanently deletes instance data.
- Confirm the actual state with Vast after either request.
- Local stop jobs run after current work and a successful empty controller claim.
  They cannot guarantee shutdown if the worker crashes, the controller is
  unreachable or its token expires. **Token expiry is not a billing deadline.**

No background monitor is installed. If a hard spending cutoff is required, agree
on that separately; do not silently add a daemon or assume local stop jobs enforce it.
