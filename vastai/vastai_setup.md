# Vast.ai worker setup

1. Generate an SSH key on each instance and add it as a read-only deploy key for this repository.
2. Clone AtariAgent on each instance and install `uv` if needed.
3. On the current host, use `jobd` to create one worker key. Ask the user for its duration; never assume it. Never copy `JOBD_MASTER_KEY` to instances.
4. Pass this `JOBD_WORKER_TOKEN` through the Ansible task's `environment`, using a protected variable and `no_log: true` to avoid exposing credentials. Supply `WANDB_API_KEY` through jobd-worker, not the SSH environment.
5. Use **Ansible async** to launch `bootstrap.sh` on every instance from its repository directory. Set `async` to a sufficient setup timeout (for example, `7200` seconds) and `poll: 0`. No worker should already be running during setup.
6. Save each instance's `ansible_job_id`, then monitor with `async_status`. Launch all instances before polling; each starts its worker independently when ready. Do not use a `nohup` wrapper or `bootstrap.exit` file.
7. Check `finished`, `rc`, `stdout`, and `stderr` for each instance. A successful launch is not successful completion: require exit code `0`. Investigate failures or timeouts before retrying, without exposing credentials.

Bootstrap syncs dependencies, installs the latest jobd, checks `JOBD_WORKER_TOKEN`, restarts the worker, then verifies the token. Verification failure does not stop an already-started worker.

Finally, submit remote smoke-test jobs on every instance **only to verify W&B authentication** using credentials supplied by jobd-worker. Use `wandb.login(key=os.environ["WANDB_API_KEY"], verify=True)` and require a successful result. Do not call `wandb.init()`, start training, or create any W&B runs. Never print the API key.
