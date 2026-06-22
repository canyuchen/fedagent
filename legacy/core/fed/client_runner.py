"""Client training execution + training-log metric parsing.

Extracted from FederatedServer. Runs a single client's training script via
subprocess, streams the output to both terminal and log file, parses
training metrics from JSON logs (primary) or stdout (fallback).
"""

import ast
import json
import os
import re
import shutil
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import OmegaConf

_PATHS_YAML = "./config/paths.yaml"


def _load_paths_cfg():
    """Lazily load config/paths.yaml relative to the current working directory.

    This is deferred out of module import time so that `import core...` from
    outside the repository root does not crash; the file is only required when
    a client training subprocess is actually launched. Raises a clear
    FileNotFoundError if the path config is missing.
    """
    if not os.path.exists(_PATHS_YAML):
        raise FileNotFoundError(
            f"[client_runner] config/paths.yaml not found from cwd={os.getcwd()}; "
            f"copy config/paths.yaml.example to config/paths.yaml and run from the "
            f"repository root."
        )
    return OmegaConf.load(_PATHS_YAML)


class ClientRunner:
    def __init__(self, checkpoint_manager, logger):
        self.checkpoint_manager = checkpoint_manager
        self.logger = logger

    def run_client_training(self, client_id: int, round_num: int,
                            script_path: str) -> Dict[str, Any]:
        """Run a single client's training script and collect the result."""
        path_cfg = _load_paths_cfg()
        client_dir = Path(script_path).parent
        log_file = client_dir / "training.log"

        process = None
        try:
            # start_new_session=True: put verl + Ray raylets + vLLM workers in a
            # dedicated process group so _cleanup_after_client can kill the whole
            # tree if anything detaches via setsid (which Ray routinely does).
            process = subprocess.Popen(
                [script_path],
                cwd=path_cfg.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                start_new_session=True,
            )

            stdout_lines = []
            with open(log_file, 'w') as log_f:
                log_f.write(f"=== Client {client_id} Training Log (Round {round_num}) ===\n")
                log_f.write(f"Start Time: {datetime.now().isoformat()}\n")
                log_f.write("=" * 50 + "\n\n")

                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        print(output.strip())
                        log_f.write(output)
                        log_f.flush()
                        stdout_lines.append(output)

            return_code = process.wait()
            stdout_content = ''.join(stdout_lines)
            metrics = self.parse_training_metrics(stdout_content, "")

            if return_code == 0:
                self.logger.info(f"Client {client_id} training completed successfully")
                self.checkpoint_manager.ensure_final_model_save(client_dir, client_id, round_num)
                return {
                    'success': True,
                    'client_id': client_id,
                    'round_num': round_num,
                    'metrics': metrics,
                    'log_file': str(log_file),
                    'model_path': self.checkpoint_manager.find_client_model_path(client_dir),
                }

            self.logger.error(f"Client {client_id} training failed with return code {return_code}")
            return {
                'success': False,
                'client_id': client_id,
                'round_num': round_num,
                'error': stdout_content,
                'log_file': str(log_file),
            }

        except subprocess.TimeoutExpired:
            # NOTE: currently unreachable. Neither subprocess.Popen above nor
            # process.wait() is given a `timeout=`, so no TimeoutExpired is ever
            # raised from this block. The '3600 seconds' figure written to the
            # log below is therefore stale/aspirational, not an enforced limit.
            # Kept so that adding a wait/Popen timeout later activates a clean
            # failure path; if you re-enable a timeout, update the 3600s text.
            self.logger.error(f"Client {client_id} training timed out")
            with open(log_file, 'w') as f:
                f.write(f"=== Client {client_id} Training Log (Round {round_num}) ===\n")
                f.write(f"Start Time: {datetime.now().isoformat()}\n")
                f.write("=" * 50 + "\n\n")
                f.write("TIMEOUT: Training exceeded 3600 seconds\n")
                f.write(f"End Time: {datetime.now().isoformat()}\n")
                f.write("=" * 50 + "\n")
            return {
                'success': False,
                'client_id': client_id,
                'round_num': round_num,
                'error': 'Training timed out',
            }
        except Exception as e:
            self.logger.error(f"Client {client_id} training error: {str(e)}")
            return {
                'success': False,
                'client_id': client_id,
                'round_num': round_num,
                'error': str(e),
            }
        finally:
            # Always clean up: Ray raylets / vLLM workers often outlive the
            # parent on crash, holding pinned RAM + GPU memory. Leaking these
            # across clients is what triggered the OOM-killer kills.
            self._cleanup_after_client(process, client_id)

    def _cleanup_after_client(self, process, client_id: int) -> None:
        """Tear down Ray cluster + child process tree + wait for GPU to free."""
        # 1) Kill the subprocess's whole process group (we set start_new_session)
        if process is not None and process.pid:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

        # 2) ray stop --force: stop the local raylet + GCS cleanly
        try:
            subprocess.run(
                ["ray", "stop", "--force"],
                check=False, timeout=30,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            self.logger.warning(f"Client {client_id}: 'ray stop --force' failed: {e}")

        # 3) Belt-and-suspenders: pkill remaining raylet / ray:: actors / vllm
        #    workers under *this user only* (safe on shared nodes).
        if shutil.which("pkill"):
            uid = str(os.getuid())
            for pattern in ("raylet", "ray::", "vllm"):
                try:
                    subprocess.run(
                        ["pkill", "-9", "-u", uid, "-f", pattern],
                        check=False, timeout=10,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                except Exception:
                    pass

        # 4) Block until GPU memory actually drops before returning — this is
        #    what lets the next client's vLLM engine reserve gpu_memory_utilization
        #    without double-booking residual allocations.
        self._wait_gpu_free(client_id, threshold_mb=2048, timeout_s=120)

    def _wait_gpu_free(self, client_id: int, threshold_mb: int, timeout_s: int) -> None:
        """Poll nvidia-smi until max used mem < threshold_mb, or timeout."""
        if not shutil.which("nvidia-smi"):
            return
        deadline = time.monotonic() + timeout_s
        last_max = None
        while time.monotonic() < deadline:
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    timeout=10, text=True,
                )
                used = [int(x.strip()) for x in out.splitlines() if x.strip()]
                last_max = max(used) if used else 0
                if last_max < threshold_mb:
                    self.logger.info(
                        f"Client {client_id}: GPU memory cleared "
                        f"(max used={last_max}MB < {threshold_mb}MB)"
                    )
                    return
            except Exception as e:
                self.logger.warning(f"Client {client_id}: nvidia-smi probe failed: {e}")
                return
            time.sleep(3)
        self.logger.warning(
            f"Client {client_id}: GPU memory did not drop below {threshold_mb}MB "
            f"within {timeout_s}s (last max={last_max}MB); proceeding anyway"
        )

    def parse_training_metrics(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse training metrics: prefer the JSON logs, fall back to stdout parsing."""
        json_metrics = self._load_metrics_from_json(stdout)
        if json_metrics:
            return json_metrics
        return self._parse_metrics_from_stdout(stdout, stderr)

    def _load_metrics_from_json(self, stdout: str) -> Optional[Dict[str, Any]]:
        """Load metrics from the JSON log file written during training."""
        try:
            client_dir = None
            for line in stdout.split('\n'):
                if 'trainer.save_dir=' in line:
                    m = re.search(r"trainer\.save_dir='([^']+)'", line)
                    if m:
                        client_dir = Path(m.group(1)).parent
                        break

            if not client_dir or not client_dir.exists():
                return None

            json_logs_dir = client_dir / "json_logs"
            metrics_file = json_logs_dir / "metrics.json"
            if not metrics_file.exists():
                return None

            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)

            training_metrics = []
            validation_metrics = []
            final_metrics = {}

            for entry in metrics_data:
                step = entry['step']
                metrics = entry['metrics']
                step_metrics = {'step': step}
                for key, value in metrics.items():
                    if 'val/' in key:
                        validation_metrics.append({key: value})
                    elif any(kw in key for kw in ['loss', 'reward', 'accuracy', 'success_rate']):
                        step_metrics[key] = value
                if len(step_metrics) > 1:
                    training_metrics.append(step_metrics)

            if validation_metrics:
                final_metrics = validation_metrics[-1]

            return {
                'training_metrics': training_metrics,
                'validation_metrics': validation_metrics,
                'final_metrics': final_metrics,
            }
        except Exception as e:
            self.logger.warning(f"Failed to load metrics from JSON: {str(e)}")
            return None

    def _parse_metrics_from_stdout(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse metrics from stdout (the fallback used when JSON logs are unavailable)."""
        metrics = self._parse_text_for_metrics(stdout)

        if not metrics['training_metrics'] and not metrics['validation_metrics']:
            for line in stdout.split('\n'):
                line = line.strip()
                if 'loss' in line.lower() and '=' in line:
                    try:
                        parts = line.split('=')
                        if len(parts) >= 2:
                            metrics['final_metrics']['loss'] = float(parts[1].strip().split()[0])
                    except Exception:
                        pass
                elif 'reward' in line.lower() and '=' in line:
                    try:
                        parts = line.split('=')
                        if len(parts) >= 2:
                            metrics['final_metrics']['reward'] = float(parts[1].strip().split()[0])
                    except Exception:
                        pass
        return metrics

    @staticmethod
    def _parse_text_for_metrics(text: str) -> Dict[str, Any]:
        """Shared parser body for both log-file and stdout text sources."""
        metrics: Dict[str, Any] = {
            'training_metrics': [],
            'validation_metrics': [],
            'final_metrics': {},
        }
        current_step = None
        training_keywords = ['loss', 'reward', 'accuracy', 'success_rate', 'episode']
        metric_names_train = [
            'loss', 'reward', 'accuracy', 'success_rate',
            'episode/reward/mean', 'episode/success_rate',
            'val/success_rate', 'val/test_score/text',
        ]
        metric_names_val = [
            'val/success_rate', 'val/test_score/text',
            'val/webshop_task_score (not success_rate)',
        ]

        for line in text.split('\n'):
            line = line.strip()

            if 'step:' in line and '-' in line:
                try:
                    step_part = line.split('step:')[1].split('-')[0].strip()
                    current_step = int(step_part)
                except Exception:
                    pass

            if (current_step is not None and 'step:' in line
                    and any(kw in line.lower() for kw in training_keywords)):
                step_metrics = {'step': current_step}
                for name in metric_names_train:
                    if name in line:
                        try:
                            parts = line.split(name + ':')
                            if len(parts) > 1:
                                step_metrics[name] = float(parts[1].split()[0])
                        except Exception:
                            pass
                if len(step_metrics) > 1:
                    metrics['training_metrics'].append(step_metrics)

            elif 'val/' in line and any(kw in line for kw in
                                        ['success_rate', 'test_score', 'webshop_task_score']):
                val_metrics = {}
                for name in metric_names_val:
                    if name in line:
                        try:
                            parts = line.split(name + ':')
                            if len(parts) > 1:
                                val_metrics[name] = float(parts[1].split()[0])
                        except Exception:
                            pass
                if val_metrics:
                    metrics['validation_metrics'].append(val_metrics)

            elif 'Final validation metrics:' in line or 'Initial validation metrics:' in line:
                try:
                    m = re.search(r'\{.*\}', line)
                    if m:
                        metrics['final_metrics'].update(ast.literal_eval(m.group()))
                except Exception:
                    pass

        return metrics
