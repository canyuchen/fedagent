#!/usr/bin/env python3
"""Federated training runner (replaces smart_federated_runner.sh logic).

The shell entry point scripts/smart_federated_runner.sh is now a thin shim
that exec's this script. start_federated.sh remains the per-run launcher.
"""

import argparse
import datetime
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
START_FEDERATED = str(PROJECT_ROOT / 'scripts' / 'start_federated.sh')
MONITOR_SCRIPT = HERE / 'monitor' / 'checkpoint_monitor.py'
PATHS_FILE = './config/paths.yaml'

sys.path.insert(0, str(HERE))
from resolve_paths import resolve  # noqa: E402


# ---------------------------------------------------------------- logging
_COLORS = {'info': '\033[0;34m', 'ok': '\033[0;32m',
           'warn': '\033[1;33m', 'err': '\033[0;31m'}
_USE_COLOR = sys.stdout.isatty()

def _log(level, label, msg):
    prefix = f"{_COLORS[level]}[{label}]\033[0m" if _USE_COLOR else f"[{label}]"
    print(f"{prefix} {msg}", flush=True)

def info(msg): _log('info', 'INFO', msg)
def ok(msg):   _log('ok',   'SUCCESS', msg)
def warn(msg): _log('warn', 'WARNING', msg)
def err(msg):  _log('err',  'ERROR', msg)


# ---------------------------------------------------------------- monitor
MONITOR_PIDS = []


def find_latest_checkpoint(output_dir):
    p = Path(output_dir)
    if not p.is_dir():
        return None
    ckpts = sorted(
        (d for d in p.iterdir() if d.is_dir() and d.name.startswith('checkpoint-')),
        key=lambda d: d.name,
    )
    return ckpts[-1] if ckpts else None


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def start_checkpoint_monitor(output_dir, keep_count=1):
    if not MONITOR_SCRIPT.is_file():
        warn("monitor script not found, skipping")
        return False

    output_dir = Path(output_dir).resolve()
    pid_file = output_dir / '.checkpoint_monitor.pid'
    log_file = output_dir / 'checkpoint_monitor.log'

    if pid_file.exists():
        try:
            old_pid = int(pid_file.read_text().strip())
            if _pid_alive(old_pid):
                info(f"monitor already running (PID: {old_pid})")
                MONITOR_PIDS.append(old_pid)
                return True
        except ValueError:
            pass
        pid_file.unlink(missing_ok=True)

    if output_dir.is_dir():
        info(f"running an initial cleanup before starting monitor: {output_dir}")
        subprocess.run(
            [sys.executable, str(MONITOR_SCRIPT), '--keep-num', str(keep_count),
             '--interval', '60', str(output_dir), '--run-once'],
            check=False,
        )
    else:
        info("monitored directory does not exist yet; monitor will wait for it to be created")

    info(f"starting checkpoint monitor: {output_dir} (keep {keep_count}, interval 60s)")
    rc = subprocess.run(
        [sys.executable, str(MONITOR_SCRIPT), '--daemon',
         '--keep-num', str(keep_count), '--interval', '60',
         '--pid-file', str(pid_file), '--log-file', str(log_file),
         str(output_dir)],
    ).returncode
    if rc != 0:
        err("failed to start monitor")
        return False

    # Poll for the child process to write the PID file (instead of a fixed sleep;
    # reliable even when the filesystem is slow).
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if pid_file.exists():
            break
        time.sleep(0.2)
    if not pid_file.exists():
        warn(f"monitor failed to start (no PID file, 10s timeout); see {log_file}")
        return False
    try:
        pid = int(pid_file.read_text().strip())
    except ValueError:
        warn(f"monitor PID file is malformed; see {log_file}")
        return False
    if not _pid_alive(pid):
        warn(f"monitor failed to start (process not found); see {log_file}")
        return False

    ok(f"monitor started (PID: {pid})")
    MONITOR_PIDS.append(pid)
    return True


# Only cleanup on user interrupt (SIGINT/SIGTERM). On normal exit the monitor
# is intentionally left running — the old shell trap EXIT killed it too, which
# contradicted the "monitor persists" intent; this is the explicit fix.
def cleanup_monitors_on_signal(signum, _frame):
    for pid in MONITOR_PIDS:
        if not _pid_alive(pid):
            continue
        info(f"terminating monitor (PID: {pid})")
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            continue
        for _ in range(5):
            if not _pid_alive(pid):
                break
            time.sleep(1)
        if _pid_alive(pid):
            warn(f"monitor not responding (PID: {pid}); force-killing")
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    sys.exit(130 if signum == signal.SIGINT else 143)


def check_monitor_status(output_dir):
    pid_file = Path(output_dir) / '.checkpoint_monitor.pid'
    log_file = Path(output_dir) / 'checkpoint_monitor.log'
    if not pid_file.is_file():
        print(f"monitor not running (PID file does not exist: {pid_file})")
        return 1
    try:
        pid = int(pid_file.read_text().strip())
    except ValueError:
        print("monitor PID file is malformed")
        return 1
    if not _pid_alive(pid):
        print("monitor has stopped")
        return 1
    print(f"monitor is running (PID: {pid})")
    if log_file.is_file():
        print(f"log: {log_file}")
        print("last 5 lines:")
        for line in log_file.read_text().splitlines()[-5:]:
            print(f"  {line}")
    return 0


def update_checkpoint_contents(config_file):
    from omegaconf import OmegaConf
    conf = OmegaConf.load(config_file)
    contents = ['model', 'optimizer', 'extra']
    if OmegaConf.select(conf, 'actor_rollout_ref.actor.checkpoint') is not None:
        conf.actor_rollout_ref.actor.checkpoint.contents = contents
    if OmegaConf.select(conf, 'critic.checkpoint') is not None:
        conf.critic.checkpoint.contents = contents
    OmegaConf.save(conf, config_file)
    info("updated checkpoint contents to the full set")


# ---------------------------------------------------------------- runs
def _run(*args):
    info(f"running: bash {START_FEDERATED} {' '.join(args)}")
    return subprocess.run(['bash', START_FEDERATED, *args]).returncode == 0


def run_init(verl_config, *extra):
    return _run('--verl-config', verl_config, *extra)


def run_resume(output_dir, config_path):
    return _run('--smart-resume', '--output-dir', str(output_dir),
                '--config', str(config_path))


def run_loop(verl_config, output_dir, config_path, total_runs, init_extra):
    """Each iteration: resume if a checkpoint exists, else init.

    Returns (succeeded, total) — caller decides final status/exit code.
    """
    succeeded = 0
    for i in range(1, total_runs + 1):
        info(f"=== run {i}/{total_runs} ===")
        if find_latest_checkpoint(output_dir):
            success = run_resume(output_dir, config_path)
        elif verl_config:
            success = run_init(verl_config, *init_extra)
        else:
            err(f"directory has no checkpoint and no verl_config, cannot init: {output_dir}")
            return succeeded, total_runs
        if success:
            succeeded += 1
            ok(f"run {i} completed")
        else:
            warn(f"run {i} failed, continuing to the next one")
        if i < total_runs:
            info("waiting 5 seconds before continuing")
            time.sleep(5)
    return succeeded, total_runs


def _finish(succeeded, total, output_dir):
    """Report final status and exit non-zero if anything failed."""
    if succeeded == total:
        ok(f"all completed: {output_dir}")
        return
    if succeeded == 0:
        err(f"all failed ({succeeded}/{total}): {output_dir}")
    else:
        warn(f"partial failure ({succeeded}/{total}): {output_dir}")
    sys.exit(1)


def resolve_target(verl_config, *, no_timestamp=False, timestamp=None):
    fields = resolve(verl_config=verl_config, paths_file=PATHS_FILE,
                     no_timestamp=no_timestamp, timestamp=timestamp)
    info("resolved:")
    info(f"  output dir: {fields['OUTPUT_DIR']}")
    info(f"  config file: {fields['CONFIG_FILE']}")
    info(f"  meta_info: {fields['META_INFO']}")
    return fields


# ---------------------------------------------------------------- modes
def mode_normal(args):
    info("=== normal mode: single run ===")
    rc = subprocess.run(
        ['bash', START_FEDERATED, '--verl-config', args.verl_config]
    ).returncode
    sys.exit(rc)


def mode_smart(args):
    info(f"=== smart-resume: init first, then loop {args.total_runs} times ===")
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    f = resolve_target(args.verl_config, timestamp=ts)
    succeeded, total = run_loop(
        args.verl_config, f['OUTPUT_DIR'], f['CONFIG_FILE'],
        args.total_runs, ['--timestamp', ts],
    )
    _finish(succeeded, total, f['OUTPUT_DIR'])


def mode_restart(args):
    info("=== restart-resume: smart detection of the output directory ===")
    f = resolve_target(args.verl_config, no_timestamp=True)
    pre_existed = Path(f['OUTPUT_DIR']).is_dir()
    if pre_existed:
        ok(f"found existing directory: {f['OUTPUT_DIR']}")
    else:
        info("directory does not exist; it will be created on the first run")
    succeeded, total = run_loop(
        args.verl_config, f['OUTPUT_DIR'], f['CONFIG_FILE'],
        args.total_runs, ['--no-timestamp'],
    )
    if not pre_existed and Path(f['OUTPUT_DIR']).is_dir():
        start_checkpoint_monitor(f['OUTPUT_DIR'])
    _finish(succeeded, total, f['OUTPUT_DIR'])


def mode_direct(args):
    info(f"=== direct-resume: loop {args.total_runs} times using the given directory ===")
    info(f"  output dir: {args.output_dir}")
    info(f"  config file: {args.config}")
    succeeded, total = run_loop(None, args.output_dir, args.config, args.total_runs, [])
    _finish(succeeded, total, args.output_dir)


def mode_centralized(args):
    info("=== centralized-resume-epoch ===")
    f = resolve_target(args.verl_config, no_timestamp=True)
    update_checkpoint_contents(f['CONFIG_FILE'])
    os.environ['CENTRALIZED_RESUME_EPOCH'] = 'true'
    os.environ['EPOCH_SAVE_FREQ'] = str(args.epoch_save_freq)
    info(f"set CENTRALIZED_RESUME_EPOCH=true, EPOCH_SAVE_FREQ={args.epoch_save_freq}")
    start_checkpoint_monitor(f['OUTPUT_DIR'])
    succeeded, total = run_loop(
        args.verl_config, f['OUTPUT_DIR'], f['CONFIG_FILE'],
        args.total_runs, ['--no-timestamp'],
    )
    info("monitor will keep running in the background")
    _finish(succeeded, total, f['OUTPUT_DIR'])


def mode_check(args):
    info(f"checking checkpoint monitor status: {args.output_dir}")
    sys.exit(check_monitor_status(args.output_dir))


HANDLERS = {
    'normal':      mode_normal,
    'smart':       mode_smart,
    'restart':     mode_restart,
    'direct':      mode_direct,
    'centralized': mode_centralized,
    'check':       mode_check,
}


# ---------------------------------------------------------------- CLI
def build_parser():
    p = argparse.ArgumentParser(
        prog='run_federated',
        description='Smart federated-learning runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s fed_webshop_ppo_total-100_cl-per-rd-3_...
  %(prog)s --restart-resume todo/fed_webshop_grpo_... 5
  %(prog)s --smart-resume fed_webshop_ppo_... 10
  %(prog)s --direct-resume --output-dir DIR --config PATH 10
  %(prog)s --centralized-resume-epoch --epoch-save-freq 5 fed_webshop_... 10
  %(prog)s --check-monitor --output-dir DIR
""")
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--smart-resume',             dest='mode', action='store_const', const='smart',       help="initialize once (with a timestamp), then loop resuming")
    modes.add_argument('--restart-resume',           dest='mode', action='store_const', const='restart',     help="smart-detect the no-timestamp directory: if it exists, loop resuming; otherwise init first")
    modes.add_argument('--direct-resume',            dest='mode', action='store_const', const='direct',      help="loop resuming using --output-dir / --config directly")
    modes.add_argument('--centralized-resume-epoch', dest='mode', action='store_const', const='centralized', help="epoch-granular resume (sets env vars + starts monitor)")
    modes.add_argument('--check-monitor',            dest='mode', action='store_const', const='check',       help="only check the checkpoint monitor status")

    p.add_argument('--output-dir')
    p.add_argument('--config', '--config-path', dest='config')
    p.add_argument('--epoch-save-freq', type=int, default=10)
    p.add_argument('positional', nargs='*',
                   help="[verl_config] [total_runs]  (direct mode only needs [total_runs])")
    return p


def parse_positional(args):
    if args.mode == 'direct':
        args.verl_config = None
        args.total_runs = int(args.positional[0]) if args.positional else 1
    else:
        args.verl_config = args.positional[0] if args.positional else None
        if len(args.positional) >= 2:
            args.total_runs = int(args.positional[1])
        else:
            args.total_runs = 1


def validate(args):
    m = args.mode
    if m in ('normal', 'smart', 'restart', 'centralized'):
        if not args.verl_config:
            err(f"{m} mode requires verl_config"); sys.exit(1)
        cfg = Path(f"./config/{args.verl_config}.yaml")
        if not cfg.is_file():
            err(f"config file does not exist: {cfg}"); sys.exit(1)
    if m in ('smart', 'restart', 'centralized') and args.total_runs < 1:
        err("total_runs must be > 0"); sys.exit(1)
    if m == 'direct':
        if not args.output_dir: err("direct mode requires --output-dir"); sys.exit(1)
        if not args.config:     err("direct mode requires --config");     sys.exit(1)
        if not Path(args.output_dir).is_dir(): err(f"output-dir does not exist: {args.output_dir}"); sys.exit(1)
        if not Path(args.config).is_file():    err(f"config does not exist: {args.config}");         sys.exit(1)
        if args.total_runs < 1: err("total_runs must be > 0"); sys.exit(1)
    if m == 'check':
        if not args.output_dir: err("check-monitor requires --output-dir"); sys.exit(1)
        if not Path(args.output_dir).is_dir(): err(f"output-dir does not exist: {args.output_dir}"); sys.exit(1)
    if m == 'centralized' and args.epoch_save_freq < 1:
        err("epoch-save-freq must be > 0"); sys.exit(1)


def main():
    signal.signal(signal.SIGINT,  cleanup_monitors_on_signal)
    signal.signal(signal.SIGTERM, cleanup_monitors_on_signal)

    args = build_parser().parse_args()
    args.mode = args.mode or 'normal'
    parse_positional(args)
    validate(args)

    info(f"mode: {args.mode}")
    if args.verl_config:
        info(f"verl_config: {args.verl_config}")
    if args.mode not in ('normal', 'check'):
        info(f"total_runs: {args.total_runs}")

    HANDLERS[args.mode](args)


if __name__ == '__main__':
    main()
