#!/usr/bin/env python3
"""Checkpoint monitor.

Watches the checkpoints under a given directory and automatically cleans up old
checkpoints, keeping only the most recent N.
"""

import os
import sys
import time
import argparse
import logging
import signal
import shutil
from pathlib import Path
from typing import Optional

# Configure the log format.
logging.basicConfig(
    level=logging.INFO,
    format='[CHECKPOINT_MONITOR] %(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class CheckpointMonitor:
    """Checkpoint monitor."""

    def __init__(self, output_dir: str, keep_num: int = 1, interval: int = 300, daemon: bool = False,
                 pid_file: Optional[str] = None, log_file: Optional[str] = None):
        """Initialize the monitor.

        Args:
            output_dir: Output directory path.
            keep_num: Number of checkpoints to keep (default: 1).
            interval: Check interval in seconds. This class default is 300 (5 minutes),
                but the CLI entry point (main) passes its own argparse default of 60
                (1 minute), so command-line runs use 60s unless --interval is given.
            daemon: Whether to run in the background.
            pid_file: PID file path (daemon mode).
            log_file: Log file path (daemon mode).
        """
        # Ensure output_dir is an absolute path.
        if not output_dir:
            raise ValueError("output_dir must not be empty")
        self.output_dir = Path(output_dir).resolve()
        self.keep_num = keep_num
        self.interval = interval
        self.daemon = daemon
        self.pid_file = Path(pid_file).resolve() if pid_file else None
        self.log_file = Path(log_file).resolve() if log_file else None
        self.running = True
        # In daemon mode, record the original parent PID to detect whether the parent has exited.
        self._orig_ppid: Optional[int] = None
        # Consecutive-error count; trips a circuit breaker once it exceeds the threshold.
        self._consecutive_errors = 0
        self._max_consecutive_errors = 10

        # Log information about the monitored directory.
        logger.info(f"initializing monitor - monitored directory: {self.output_dir}")
        logger.info(f"monitored directory (absolute path): {self.output_dir.absolute()}")
        logger.info(f"monitored directory exists: {self.output_dir.exists()}")
        if self.output_dir.exists():
            try:
                contents = list(self.output_dir.iterdir())[:5]
                logger.info(f"monitored directory contents (first 5): {[item.name for item in contents]}")
            except Exception as e:
                logger.warning(f"could not list the monitored directory contents: {e}")

        # Set up signal handling.
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        # If a log file was specified, output to both the file and the console.
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            # Truncate the log file on each start ('w' mode) rather than appending.
            file_handler = logging.FileHandler(self.log_file, mode='w')
            file_handler.setFormatter(logging.Formatter(
                '[CHECKPOINT_MONITOR] %(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            logger.addHandler(file_handler)
    
    def _signal_handler(self, signum, frame):
        """Signal handler."""
        logger.info(f"received signal {signum}, stopping the monitor...")
        self.running = False

    def _parent_died(self) -> bool:
        """In daemon mode, detect whether the parent process has exited (a changed ppid means it was reparented)."""
        if self._orig_ppid is None:
            return False
        try:
            return os.getppid() != self._orig_ppid
        except OSError:
            return True

    def _interruptible_sleep(self, seconds: float) -> bool:
        """Sleep in segments, checking the running state and parent liveness in between.

        Returns:
            True if the sleep was interrupted (the loop should exit); False if it
            completed normally.
        """
        end = time.monotonic() + seconds
        while self.running:
            remaining = end - time.monotonic()
            if remaining <= 0:
                return False
            if self._parent_died():
                logger.info("parent process has exited (adopted by init/subreaper); monitor stopping automatically")
                self.running = False
                return True
            time.sleep(min(2.0, remaining))
        return True

    def _write_pid_atomic(self, pid: int):
        """Write the PID file atomically (tmp + rename) to avoid concurrent starts overwriting each other."""
        assert self.pid_file is not None
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.pid_file.with_name(self.pid_file.name + '.tmp')
        tmp.write_text(str(pid))
        os.replace(tmp, self.pid_file)

    @staticmethod
    def _redirect_std_fds_to_devnull():
        """Daemonize: close the inherited stdin/stdout/stderr and redirect them to /dev/null."""
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        devnull = os.open(os.devnull, os.O_RDWR)
        for fd in (0, 1, 2):
            try:
                os.dup2(devnull, fd)
            except OSError:
                pass
        if devnull > 2:
            os.close(devnull)
    
    def cleanup_old_checkpoints(self, checkpoints_dir: Path) -> int:
        """
        Clean up old checkpoints, keeping only the most recent N.

        Args:
            checkpoints_dir: Checkpoint directory path.

        Returns:
            Number of checkpoints deleted.
        """
        if not checkpoints_dir.exists() or not checkpoints_dir.is_dir():
            return 0

        # Find all global_step directories.
        global_step_dirs = []
        for item in checkpoints_dir.iterdir():
            if item.is_dir() and item.name.startswith('global_step_'):
                try:
                    # Extract the step number for sorting.
                    step_num = int(item.name.split('_')[2])
                    global_step_dirs.append((step_num, item))
                except (ValueError, IndexError):
                    logger.warning(f"could not parse the checkpoint directory name: {item.name}")
                    continue

        # If there are no global_step directories, try other checkpoint patterns.
        if not global_step_dirs:
            for item in checkpoints_dir.iterdir():
                if item.is_dir() and item.name.startswith('checkpoint-'):
                    # Fallback for HF-style 'checkpoint-<n>' dirs (no 'global_step_<n>' present).
                    # NOTE: these are all given the same sort key (0), so the later
                    # sort(key=x[0]) leaves them in unspecified filesystem-iteration order,
                    # not numeric or name order. The "keep the most recent" guarantee below
                    # therefore does not hold for this path; it keeps an arbitrary keep_num
                    # of them. Parse the trailing step number here if true recency is needed.
                    global_step_dirs.append((0, item))

        if not global_step_dirs:
            return 0

        # Sort by step number.
        global_step_dirs.sort(key=lambda x: x[0])

        # If the number of checkpoints does not exceed the keep count, no cleanup is needed.
        if len(global_step_dirs) <= self.keep_num:
            return 0

        # Compute how many checkpoints need to be deleted.
        delete_count = len(global_step_dirs) - self.keep_num
        logger.info(f"found {len(global_step_dirs)} checkpoints; need to delete {delete_count} old checkpoints")

        # Delete the old checkpoints, keeping the most recent.
        deleted_count = 0
        for i in range(delete_count):
            _, checkpoint_dir = global_step_dirs[i]
            try:
                shutil.rmtree(checkpoint_dir)
                logger.info(f"deleted old checkpoint: {checkpoint_dir.name}")
                deleted_count += 1
            except Exception as e:
                logger.warning(f"failed to delete checkpoint {checkpoint_dir}: {e}")

        if deleted_count > 0:
            logger.info(f"successfully deleted {deleted_count} old checkpoints, keeping the most recent {self.keep_num}")
        
        return deleted_count
    
    def cleanup_all_client_checkpoints(self) -> int:
        """
        Clean up the client checkpoints for all rounds.

        Returns:
            Total number of checkpoints deleted.
        """
        logger.info(f"starting checkpoint cleanup - monitored directory: {self.output_dir}")
        logger.info(f"monitored directory (absolute path): {self.output_dir.absolute()}")
        if not self.output_dir.exists():
            logger.warning(f"output directory does not exist: {self.output_dir}")
            logger.warning(f"output directory (absolute path): {self.output_dir.absolute()}")
            return 0

        total_deleted = 0

        # Find all round directories.
        round_dirs = []
        for item in self.output_dir.iterdir():
            if item.is_dir() and item.name.startswith('round_'):
                try:
                    round_num = int(item.name.split('_')[1])
                    round_dirs.append((round_num, item))
                except (ValueError, IndexError):
                    continue

        round_dirs.sort(key=lambda x: x[0])

        # Iterate over each round directory.
        for round_num, round_dir in round_dirs:
            # Find all client directories under this round.
            client_dirs = []
            for item in round_dir.iterdir():
                if item.is_dir() and item.name.startswith('client_'):
                    try:
                        client_id = int(item.name.split('_')[1])
                        client_dirs.append((client_id, item))
                    except (ValueError, IndexError):
                        continue

            # Iterate over each client directory and clean up its checkpoints.
            for client_id, client_dir in client_dirs:
                checkpoints_dir = client_dir / "checkpoints"
                if checkpoints_dir.exists() and checkpoints_dir.is_dir():
                    deleted = self.cleanup_old_checkpoints(checkpoints_dir)
                    total_deleted += deleted
        
        return total_deleted
    
    def run(self, run_once: bool = False):
        """Run the monitoring loop.

        Args:
            run_once: If True, perform a single cleanup and exit without entering the loop.
        """
        logger.info("=" * 60)
        logger.info("Checkpoint monitor starting")
        logger.info(f"monitored directory: {self.output_dir}")
        logger.info(f"monitored directory (absolute path): {self.output_dir.absolute()}")
        logger.info(f"monitored directory exists: {self.output_dir.exists()}")
        if self.output_dir.exists():
            try:
                contents = list(self.output_dir.iterdir())
                logger.info(f"monitored directory contains {len(contents)} items")
                if contents:
                    logger.info(f"monitored directory contents (first 10): {[item.name for item in contents[:10]]}")
            except Exception as e:
                logger.warning(f"could not list the monitored directory contents: {e}")
        else:
            logger.warning(f"monitored directory does not exist; will wait for it to be created")
        logger.info(f"checkpoints to keep: {self.keep_num}")
        logger.info(f"check interval: {self.interval}s ({self.interval // 60} min)")
        if self.pid_file:
            logger.info(f"PID file: {self.pid_file}")
        if self.log_file:
            logger.info(f"log file: {self.log_file}")
        if run_once:
            logger.info("run mode: single run (perform one cleanup and exit)")
        logger.info("=" * 60)

        # Perform an initial cleanup.
        logger.info("running the initial cleanup...")
        deleted = self.cleanup_all_client_checkpoints()
        if deleted > 0:
            logger.info(f"the initial cleanup deleted {deleted} old checkpoints")
        else:
            logger.info("the initial cleanup found no checkpoints to clean up")

        # If running only once, exit after the cleanup.
        if run_once:
            logger.info("single-run mode: cleanup complete, exiting")
            return

        # Main loop.
        while self.running:
            try:
                if self._interruptible_sleep(self.interval):
                    break

                logger.info("running the periodic cleanup...")
                logger.info(f"current monitored directory: {self.output_dir}")
                logger.info(f"monitored directory exists: {self.output_dir.exists()}")
                deleted = self.cleanup_all_client_checkpoints()
                if deleted > 0:
                    logger.info(f"this cleanup deleted {deleted} old checkpoints")
                else:
                    logger.info("this check found no checkpoints to clean up (they may already be up to date)")
                self._consecutive_errors = 0

            except KeyboardInterrupt:
                logger.info("received keyboard interrupt, stopping the monitor")
                self.running = False
                break
            except Exception as e:
                self._consecutive_errors += 1
                logger.error(
                    f"error in the monitoring loop ({self._consecutive_errors}/{self._max_consecutive_errors}): {e}",
                    exc_info=True,
                )
                if self._consecutive_errors >= self._max_consecutive_errors:
                    logger.error(
                        f"{self._max_consecutive_errors} consecutive failures; tripping the circuit breaker, monitor exiting"
                    )
                    self.running = False
                    break
                # After an error, wait (interruptibly) before continuing.
                if self._interruptible_sleep(60):
                    break

        logger.info("monitor stopped")
        # Clean up the PID file.
        if self.pid_file and self.pid_file.exists():
            try:
                self.pid_file.unlink()
                logger.info(f"removed the PID file: {self.pid_file}")
            except Exception as e:
                logger.warning(f"failed to remove the PID file: {e}")
    
    def start_daemon(self):
        """Start the background process (simplified version)."""
        if not self.pid_file:
            logger.error("daemon mode requires a PID file")
            return False

        # Check whether a process is already running.
        if self.pid_file.exists():
            try:
                old_pid = int(self.pid_file.read_text().strip())
                if os.path.exists(f"/proc/{old_pid}"):  # Simple check for whether the process exists.
                    logger.error(f"monitor process already running (PID: {old_pid})")
                    return False
            except:
                pass
            self.pid_file.unlink()  # Clean up the stale PID file.

        # Create the background process with fork.
        try:
            pid = os.fork()
            if pid == 0:
                # Child process: run detached in the background. Record the original
                # parent PID now, before setsid(), so the liveness check used during the
                # sleep loop (_parent_died via _interruptible_sleep) can later notice if
                # that parent exits: when it does, this child is reparented to init/a
                # subreaper, getppid() changes away from this recorded value, and the
                # monitor shuts itself down instead of lingering as an orphan.
                self._orig_ppid = os.getppid()
                os.setsid()

                # Write the PID file atomically.
                self._write_pid_atomic(os.getpid())

                # Make sure the log file is reconfigured in the child (logging handlers can be problematic after fork).
                if self.log_file:
                    # Remove the existing handlers and re-add them.
                    for handler in logger.handlers[:]:
                        logger.removeHandler(handler)
                    # Add the file handler (truncated on each start).
                    file_handler = logging.FileHandler(self.log_file, mode='w')
                    file_handler.setFormatter(logging.Formatter(
                        '[CHECKPOINT_MONITOR] %(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S'
                    ))
                    logger.addHandler(file_handler)

                # Close the inherited stdin/stdout/stderr to avoid writing to closed FDs after the parent exits.
                # Note: do this after the log-file handler has been re-attached to avoid losing log records.
                self._redirect_std_fds_to_devnull()

                # Run the monitoring loop.
                self.run()
                sys.exit(0)
            else:
                # Parent process: return immediately.
                time.sleep(0.5)  # Briefly wait for the child to start.
                return True
        except OSError as e:
            logger.error(f"failed to start the background process: {e}")
            return False


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description='Checkpoint monitor')
    parser.add_argument('output_dir', help='Output directory path to monitor')
    parser.add_argument('--keep-num', type=int, default=1, help='Keep the most recent N checkpoints (default: 1)')
    parser.add_argument('--interval', type=int, default=60, help='Check interval in seconds (default: 60, i.e. 1 minute)')
    parser.add_argument('--daemon', action='store_true', help='Run in the background')
    parser.add_argument('--pid-file', type=str, help='PID file path (daemon mode)')
    parser.add_argument('--log-file', type=str, help='Log file path (daemon mode)')
    parser.add_argument('--run-once', action='store_true', help='Perform a single cleanup and exit without entering the loop')
    args = parser.parse_args()

    # Create the monitor.
    monitor = CheckpointMonitor(
        output_dir=args.output_dir,
        keep_num=args.keep_num,
        interval=args.interval,
        daemon=args.daemon,
        pid_file=args.pid_file,
        log_file=args.log_file
    )
    
    # Start monitoring.
    if args.daemon:
        if not args.pid_file:
            logger.error("daemon mode requires the --pid-file argument")
            sys.exit(1)
        if args.run_once:
            logger.error("daemon mode cannot be combined with --run-once")
            sys.exit(1)
        success = monitor.start_daemon()
        sys.exit(0 if success else 1)
    else:
        # Run in the foreground.
        monitor.run(run_once=args.run_once)


if __name__ == '__main__':
    main()

