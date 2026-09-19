1. generate key in the instances, add it into deploy keys for current repos, read only.
2. git clone AtariAgent repos into multiple instances.
3. use jobd to create one worker_key, ask me for the duration. not assume it. nerver copy JOBD_MASTER_KEY into instances.
4. add this one JOBD_WORKER_TOKEN into the instances env.
5. if multiple instance, use pssh to handle it. or use ansible
6. install latest version jobd via curl -fsSL https://raw.githubusercontent.com/Gzsiceberg/jobd/main/install.sh | sh -s -- --version latest
7. verify the JOBD_WORKER_TOKEN works
8. schedu uv sync and uv sync --extrac wandb into local queue via jobd in every instance.
9. schedule some remote jobs to verify every instances work, especitally verfiy if wandb work
