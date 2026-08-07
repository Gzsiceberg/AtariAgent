# EfficientZero V1 Replay Buffer, Reanalysis, and Off-Policy Correction

This note summarizes the behavior verified against the official EfficientZero V1 repository:

- Repository: `~/repos/EfficientZero`
- Upstream: `YeWR/EfficientZero`
- Commit inspected: `468bb0309f6d5a632a53da9c7d329f88fc9ebf8e`
- Paper excerpt: [`replay-buffer.png`](./replay-buffer.png)

The findings below concern EfficientZero V1, not EfficientZero V2.

## Summary

The default EfficientZero V1 data path is:

```text
self-play with MCTS
        ↓
400-transition GameHistory blocks
        ↓
prioritized replay buffer
        ↓
sample starting transitions
        ↓
reanalyze value targets for every sample
reanalyze policy targets for 99% of the batch
        ↓
train the online network
        ↓
replace each sampled starting transition's priority
```

Important conclusions:

1. EfficientZero V1 really uses prioritized experience replay in its supplied training command.
2. The 400-move sequences are replay storage blocks, not 400-step training unrolls.
3. A newly inserted transition starts with the maximum replay priority.
4. After it is sampled, its priority is replaced by its scalar value error.
5. EfficientZero V1 always reconstructs the value target using a delayed target/reanalysis network.
6. Disabling policy reanalysis does **not** disable value reanalysis.
7. With policy reanalysis disabled, the policy target is the MCTS visit distribution stored during self-play.
8. Policy reanalysis is implemented in the V1 code and defaults to 99% of each batch.
9. Reanalysis is performed on the fly and does not rewrite the stored policy targets.
10. Parallel workers may reanalyze the same replay state multiple times.

---

## 1. What self-play stores

For each agent transition, self-play records:

- observation frames,
- selected action,
- clipped environment reward,
- MCTS root value,
- normalized MCTS root visit counts.

The normalized root visit counts are stored as the original policy target:

\[
\pi_t^{\text{stored}}(a)
=
\frac{N_t(a)}{\sum_b N_t(b)}.
\]

They are stored in `GameHistory.child_visits`. Root values are stored in `GameHistory.root_values`.

Relevant code:

- `~/repos/EfficientZero/core/game.py::GameHistory.store_search_stats`
- `~/repos/EfficientZero/core/selfplay_worker.py::DataWorker.run`

## 2. Meaning of the 400-move sequence

Atari trajectories may be very long, so EfficientZero splits them into `GameHistory` blocks:

```python
history_length = 400
```

The 400 transitions are storage blocks. Replay still samples an individual starting transition from a block, and training uses a short model unroll.

The block boundary is padded with observations, rewards, root values, and policies from the following block so that value targets and model unroll targets can be constructed near the boundary. Padding does not increase the logical transition count because `GameHistory.__len__()` counts actions.

For Atari, one agent action uses frame skip 4, so a 400-action block covers approximately 1,600 emulator frames.

Relevant code:

- `~/repos/EfficientZero/config/atari/__init__.py`
- `~/repos/EfficientZero/core/game.py::GameHistory.pad_over`
- `~/repos/EfficientZero/core/selfplay_worker.py::DataWorker.put_last_trajectory`

The paper sentence saying that priority gives only a small improvement does not mean priority was disabled. The priority mechanism and 400-step block storage are separate design choices.

---

## 3. `--use_priority` versus `--use_max_priority`

The supplied EfficientZero V1 command enables both:

```bash
--use_priority \
--use_max_priority
```

Relevant command: `~/repos/EfficientZero/train.sh`.

| Flags | Replay sampling | Initial priority for new transitions |
|---|---|---|
| Neither | Uniform | Priority values are ignored by sampling |
| `--use_priority` | Prioritized | Self-play value error, intended as `abs(network value - MCTS root value) + eps` |
| `--use_priority --use_max_priority` | Prioritized | Current maximum raw priority in replay |

`--use_priority` enables priority-based sampling. `--use_max_priority` only changes how newly collected transitions are initialized. It has no effect unless priority sampling is enabled:

```python
self.use_max_priority = args.use_max_priority if self.use_priority else False
```

With the official flags, self-play sends `priorities=None`, which causes replay to assign the current maximum priority:

```python
max_prio = self.priorities.max() if self.buffer else 1
```

The first replay data therefore receives priority `1`.

---

## 4. Replay representation and sampling

Replay stores complete `GameHistory` blocks, but priority belongs to individual transition positions:

```text
buffer[block_id]       → GameHistory block
priorities[index]      → raw priority of one transition
game_look_up[index]    → (block_id, position_in_block)
```

The sampling probability is:

\[
P(i)=\frac{p_i^\alpha}{\sum_k p_k^\alpha},
\qquad \alpha=0.6.
\]

The implementation uses NumPy rather than a sum tree:

```python
probs = self.priorities ** self._alpha
probs /= probs.sum()
indices = np.random.choice(total, batch_size, p=probs, replace=False)
```

Starting indices cannot repeat within a single sampling call because `replace=False`. They can repeat across batches and across workers.

Relevant code: `~/repos/EfficientZero/core/replay_buffer.py::ReplayBuffer.prepare_batch_context`.

---

## 5. Importance-sampling weight

Prioritized sampling changes the training distribution away from uniform replay. The uniform-to-prioritized probability ratio is:

\[
\frac{P_{\mathrm{uniform}}(i)}{P(i)}
=
\frac{1/N}{P(i)}
=
\frac{1}{N P(i)}.
\]

EfficientZero uses the partial correction:

\[
w_i=\left(\frac{1}{N P(i)}\right)^\beta.
\]

Here:

- `N` is the current replay transition count,
- `P(i)` is the transition's prioritized sampling probability,
- `beta` controls correction strength.

`beta` is linearly annealed from `0.4` to `1.0`:

- `beta = 0`: no correction,
- `0 < beta < 1`: partial correction,
- `beta = 1`: full correction toward the uniform-replay objective.

The implementation normalizes by the largest weight in the sampled batch:

```python
weights = (total * probs[indices]) ** (-beta)
weights /= weights.max()
```

The importance weight multiplies the complete per-sample loss, including policy, value, value-prefix, and consistency terms:

```python
weighted_loss = (weights * loss).mean()
```

High-probability transitions therefore receive smaller gradient weights, while rare transitions receive larger relative weights.

---

## 6. Priority lifecycle

### 6.1 Initial insertion

A new transition receives the current maximum priority:

\[
p_t \leftarrow \max_j p_j.
\]

It keeps this priority until it is sampled and trained.

### 6.2 Sampling and target construction

For sampled starting state `s_t`, EfficientZero constructs a scalar target `z_t`. In V1 this is the off-policy-corrected, reanalyzed n-step target described later in this note.

### 6.3 New priority

Before the current optimizer update, the online model predicts:

\[
v_\theta(s_t).
\]

The new raw priority is:

\[
p_t^{\mathrm{new}}
=
\left|v_\theta(s_t)-z_t\right|+\epsilon,
\qquad \epsilon=10^{-6}.
\]

In code:

```python
value_priority = L1Loss(reduction='none')(
    scaled_value.squeeze(-1),
    target_value[:, 0],
)
value_priority = value_priority.data.cpu().numpy() + config.prioritized_replay_eps
```

The error is:

- a scalar value error,
- not the total loss,
- not multiplied by the importance weight,
- computed only for the sampled starting transition,
- not computed for every state in its five-step model unroll.

The replay entry is replaced:

```python
self.priorities[idx] = new_priority
```

Example:

```text
initial maximum priority = 5.0
online value prediction  = 2.2
reanalyzed value target  = 4.0
new priority             = abs(2.2 - 4.0) + 1e-6
                         = 1.800001
```

Future batches recompute `P(i)` using `1.800001 ** 0.6`.

### 6.4 Before or after the optimizer update?

`v_theta(s_t)` is predicted before the current optimizer update. The order is:

```text
online forward pass
    ↓
calculate and detach value error
    ↓
backward and optimizer.step()
    ↓
send the already calculated priority to replay
```

Although the replay RPC occurs after `optimizer.step()`, there is no second forward pass. The priority uses the pre-update prediction.

Relevant code: `~/repos/EfficientZero/core/train.py::update_weights`.

---

## 7. Replay limits and eviction

EfficientZero V1 has two limits with different behavior.

### 7.1 Collection limit: `total_transitions`

Standard Atari configuration:

```python
total_transitions = 100_000
```

Before inserting a block:

```python
if self.get_total_len() >= self.config.total_transitions:
    return
```

After the collection limit is reached, new blocks are rejected. Existing data is not replaced. Because the check occurs before inserting a complete block, the count can overshoot slightly on the last accepted block.

### 7.2 Generic replay-window limit: `transition_top`

The rolling window is:

```python
transition_top = int(config.transition_num * 1_000_000)
```

With the Atari default `transition_num = 1`, this is one million transitions.

Every 1,000 training updates, `remove_to_fit()` checks this limit. If it is exceeded, replay removes the oldest complete `GameHistory` blocks until the transition count is at or below the limit.

Eviction is:

- FIFO,
- block-level,
- independent of transition priority.

It does not remove the lowest-priority transitions.

For the standard experiment:

```text
collection limit     = 100,000
replay-window limit  = 1,000,000
```

The collection limit is reached first, so rolling FIFO eviction normally does not occur in the standard 100k setting.

Relevant code:

- `~/repos/EfficientZero/core/replay_buffer.py::ReplayBuffer.save_game`
- `~/repos/EfficientZero/core/replay_buffer.py::ReplayBuffer.remove_to_fit`
- `~/repos/EfficientZero/core/train.py::_train`

---

## 8. Online, target/reanalysis, and self-play networks

EfficientZero V1 has conceptually separate network roles:

1. **Online network `theta`**: updated by every training batch.
2. **Target/reanalysis network `theta_bar`**: generates bootstrap values and reanalyzed search targets; it receives no training gradients.
3. **Self-play snapshots**: periodically copied from the online network to self-play workers.

The target network uses periodic hard copies, not Polyak/EMA updates. Atari config uses:

```python
target_model_interval = 200
checkpoint_interval = 100
```

Because the implementation copies a delayed `recent_weights` snapshot and batch production is asynchronous, target parameters are approximately 200--400 online updates behind.

Relevant code:

- `~/repos/EfficientZero/core/train.py::_train`
- `~/repos/EfficientZero/core/reanalyze_worker.py::BatchWorker_GPU`

---

## 9. Reanalyzed value target and model-based off-policy correction

For sampled transition `t`, EfficientZero uses stored real rewards for a dynamic horizon `l`, then bootstraps from a freshly evaluated endpoint:

\[
z_t^{\mathrm{OPC}}
=
\sum_{j=0}^{l-1}\gamma^j r_{t+j}
+
\gamma^l \nu_{t+l}.
\]

The endpoint value is freshly computed from the delayed target/reanalysis model.

### Default released-code endpoint

Without `--use_root_value`:

\[
\nu_{t+l}=v_{\bar\theta}(s_{t+l}).
\]

This is direct initial inference by the target/reanalysis network.

### Optional MCTS endpoint

With `--use_root_value`:

\[
\nu_{t+l}
=
\operatorname{MCTSValue}(s_{t+l};\bar\theta).
\]

The MCTS uses the representation, dynamics, reward/value-prefix, policy, and value heads of the recent target/reanalysis network. It is model-based and does not restore or roll the real Atari environment forward.

The paper describes the MCTS root empirical mean value. However, the supplied V1 `train.sh` does not enable `--use_root_value`; the released default uses direct target-network value prediction. The repository states that root values provided limited improvement while requiring more GPU actors.

### Relationship to priority

The MCTS or direct-network endpoint `nu_(t+l)` is only one component of the full target `z_t`. Priority compares the online prediction at the starting state against the complete reward-plus-bootstrap target:

\[
\nu_{t+l}
\longrightarrow
z_t^{\mathrm{OPC}}
\longrightarrow
p_t=
\left|v_\theta(s_t)-z_t^{\mathrm{OPC}}\right|+\epsilon.
\]

There are not two unrelated values called an off-policy value and a priority reanalysis value. The off-policy-corrected target is the value target used by the priority calculation.

---

## 10. Dynamic horizon `l`

`l` is the number of stored real trajectory rewards used before bootstrapping. It is not:

- the 400-step replay block length,
- the number of MCTS simulations,
- MCTS search depth,
- the model-unroll length, although both maximum values happen to be 5 in this config.

V1 computes:

```python
delta_td = (total_transitions - idx) // config.auto_td_steps
td_steps = config.td_steps - delta_td
td_steps = np.clip(td_steps, 1, 5)
```

For standard Atari:

```python
td_steps = 5
auto_td_steps = 0.3 * 100_000 = 30_000
```

Therefore:

\[
l=
\operatorname{clip}
\left(
5-\left\lfloor\frac{N-i}{30{,}000}\right\rfloor,
1,5
\right),
\]

where `N - i` approximates transition age.

| Approximate age | `l` |
|---:|---:|
| 0--29,999 | 5 |
| 30,000--59,999 | 4 |
| 60,000--89,999 | 3 |
| 90,000--119,999 | 2 |
| 120,000 or more | 1 |

Example with `l = 3`:

\[
z_t
=
r_t+\gamma r_{t+1}+\gamma^2r_{t+2}
+\gamma^3\nu_{t+3}.
\]

Older data receives a shorter reward horizon so the target relies on fewer actions selected by an old behavior policy. In Atari, each agent step has frame skip 4, so `l = 5` spans up to 20 emulator frames.

Relevant code:

- `~/repos/EfficientZero/core/reanalyze_worker.py::BatchWorker_CPU._prepare_reward_value_context`
- `~/repos/EfficientZero/core/reanalyze_worker.py::BatchWorker_GPU._prepare_reward_value`

---

## 11. Policy reanalysis: verified V1 behavior

EfficientZero V1 really does reanalyze policy targets.

The command-line default is:

```python
revisit_policy_search_rate = 0.99
```

For batch size 256:

```python
re_num = int(256 * 0.99)  # 253
```

Approximately 253 samples use freshly searched policy targets and three use stored self-play targets.

For a reanalyzed sample, V1:

1. Gets the stored real observations at the sampled state and all model-unroll target states.
2. Runs target/reanalysis-network initial inference.
3. Creates MCTS roots.
4. Runs fresh MCTS using `self.model`, which is the recent target/reanalysis model.
5. Normalizes the new root visit counts.
6. Places those distributions into the current training batch.

Core code:

```python
# do MCTS for a new policy with the recent target model
MCTS(self.config).search(
    roots,
    self.model,
    hidden_state_roots,
    reward_hidden_roots,
)

roots_distributions = roots.get_distributions()
policy = [visit_count / sum(distributions) for visit_count in distributions]
target_policies.append(policy)
```

Relevant locations:

- `~/repos/EfficientZero/main.py` (`revisit_policy_search_rate=0.99`)
- `~/repos/EfficientZero/core/reanalyze_worker.py::BatchWorker_CPU.make_batch`
- `~/repos/EfficientZero/core/reanalyze_worker.py::BatchWorker_GPU._prepare_policy_re`

### Reanalysis does not rewrite replay

The fresh policy target is ephemeral. It is included in `targets_batch`, but not saved into `GameHistory.child_visits`. The potential replay mutation is explicitly commented out:

```python
# game.store_search_stats(distributions, value, current_index)
```

Therefore, the same transition can receive another fresh search target the next time it is sampled.

---

## 12. Especially important: behavior with no policy reanalysis

Set:

```bash
--revisit_policy_search_rate 0
```

This disables only policy reanalysis.

### Policy target

The policy target is the normalized MCTS root visit distribution generated during the original self-play and stored in:

```python
GameHistory.child_visits
```

When sampled, the non-reanalysis path copies it directly:

```python
target_policies.append(child_visit[current_index])
```

No new policy MCTS is run.

### Value target

The value target is still reanalyzed. `revisit_policy_search_rate` does not control value-target preparation. Every sampled transition still gets:

\[
z_t
=
\sum_{j=0}^{l-1}\gamma^j r_{t+j}
+
\gamma^l v_{\bar\theta}(s_{t+l})
\]

by default, or the fresh MCTS root value when `--use_root_value` is enabled.

Thus, with policy reanalysis disabled:

```text
policy target → stored self-play MCTS child-visit distribution
value target  → fresh delayed-target-network bootstrap
rewards       → stored real environment rewards
```

The released V1 code does not expose one flag that completely disables value reanalysis/off-policy target construction.

If all reanalysis were removed in a different implementation, both targets would be based on self-play data:

```text
policy target → stored child visits
value target  → stored rewards + stored self-play MCTS root value
```

---

## 13. MuZero baseline versus MuZero Reanalyze

The original MuZero paper has a distinction that is easy to miss.

### Base MuZero pseudocode

The base pseudocode does not include policy reanalysis. It samples replay positions and uses:

```text
policy target → child visits stored during self-play
value target  → stored rewards + stored self-play MCTS root value
```

### MuZero Reanalyze variant

Appendix H describes an additional Reanalyze configuration that is not fully incorporated into the base pseudocode:

- approximately 80% of updates use a fresh MCTS policy target,
- approximately 20% retain the stored self-play policy,
- the value target uses stored rewards and a recent target-network bootstrap,
- Atari uses a fixed value horizon around 5.

Reanalysis does not generate a replacement environment trajectory. It reuses stored observations/actions/rewards and refreshes selected targets.

### Difference from EfficientZero

| Behavior | Base MuZero pseudocode | MuZero Reanalyze | EfficientZero V1 default |
|---|---:|---:|---:|
| Replay buffer | Yes | Yes | Yes |
| Fresh policy MCTS | No | About 80% | 99% |
| Fresh target-network value bootstrap | No | Yes | Yes |
| Horizon depends on trajectory age | No | No | Yes |
| Stored actions/rewards retained | Yes | Yes | Yes |

EfficientZero's distinctive model-based off-policy correction is the age-dependent shortening from `l = 5` toward `l = 1`.

Original MuZero source: Appendix H of [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265).

---

## 14. Parallel reanalysis and duplicate work

The same replay transition can be reanalyzed multiple times.

### Same sampling call

Exact starting indices are unique because replay samples with `replace=False`.

However, model-unroll windows may overlap:

```text
sample A starts at t:      t, t+1, t+2, t+3, t+4, t+5
sample B starts at t+2:          t+2, t+3, t+4, t+5, t+6, t+7
```

The overlapping states may receive separate MCTS searches even within one batch.

### Different workers or batches

CPU batch workers sample independently. Replay does not reserve or mark transitions as currently being reanalyzed. Consequently:

```text
worker A samples transition 100
worker B samples transition 100
both generate fresh targets
```

This is valid because reanalysis is meant to refresh a target whenever a transition is used, not exactly once per transition. Results can differ because:

- MCTS root noise is sampled independently,
- workers may hold different target-network snapshots,
- the target model may have advanced between samples.

Fresh policies are not written back into replay, so there is no policy-write conflict. Priority updates are handled by the Ray replay-buffer actor; a later processed update replaces the earlier priority for that index. The replay timestamp check protects against applying updates to data removed by FIFO eviction, not against duplicate reanalysis.

---

## 15. Compact data-flow reference

### Default EfficientZero V1

```text
Self-play MCTS
  stores observations, actions, rewards, root values, child visits
        ↓
Replay insertion
  new raw priority = current max priority
        ↓
Prioritized sample
  P(i) proportional to p_i^0.6
  importance weight = (1 / (N * P(i)))^beta
        ↓
Target construction
  policy: fresh MCTS for 99%, stored child visits for 1%
  value: stored rewards for dynamic l + fresh target-network bootstrap
        ↓
Online forward pass
  v_theta(s_t)
        ↓
Train with importance-weighted loss
        ↓
Priority replacement
  p_i = abs(v_theta_before_update(s_t) - z_t) + 1e-6
```

### EfficientZero with no policy reanalysis

```text
--revisit_policy_search_rate 0

policy target = stored self-play child visits
value target  = stored rewards + fresh target-network bootstrap
priority      = abs(online value - fresh value target) + epsilon
```
