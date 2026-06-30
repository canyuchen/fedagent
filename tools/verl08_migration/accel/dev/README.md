# Equivalence / persistence dev smokes

Small-model (mostly TinyGuess) A/B smokes that validated the **persistent-trainer (#4)**, **cross-round
persistence**, and **PPO critic-reload** paths produce equivalent results to the subprocess-per-(client,
round) baseline. Analysis: [`acceleration.md`](../../../../fedagent/docs/acceleration.md) §7.1 / §7.2.
Output → gitignored.

| config(s) | experiment | doc |
|---|---|---|
| `persist_full.yaml`, `subproc_full.yaml` | persistent (`persistent: true`) vs subprocess full-loop equivalence (GRPO) — max\|Δ\|=1.13e-5 | §7.1 |
| `xround_full.yaml`, `xround_recheck.yaml`, `xround_ppo.yaml` | cross-round persistence (`cross_round: true`, one process for the whole run) | §7.2 |
| `ppo_ab_subproc.yaml`, `ppo_ab_xround.yaml`, `ppo_persist_smoke.yaml` | PPO + cross-round persistence (critic + actor reload) | §7.2 |
| `tinyguess_baseline.yaml`, `tinyguess_subproc_ab.yaml`, `tinyguess_windowed_check.yaml` | TinyGuess equivalence baselines + windowed-default no-crash check | §7.1 / §7.5 |

Drivers: `../run_persistent_smoke.sh`, `../run_xround_full.sh`, `../run_xround_recheck.sh`,
`../run_ppo_persist_smoke.sh` (historical — reference the retired `_scratch/accel/` base; see `../README.md`).
