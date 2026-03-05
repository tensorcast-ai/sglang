import logging
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

from sglang.srt.debug_utils import cuda_coredump
from sglang.srt.utils.common import kill_process_tree
from sglang.test.ci.ci_register import CIRegistry

# Configure logger to output to stdout
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TestFile:
    name: str
    estimated_time: float = 60


# Patterns that indicate retriable accuracy/performance failures
RETRIABLE_PATTERNS = [
    r"AssertionError:.*not greater than",
    r"AssertionError:.*not less than",
    r"AssertionError:.*not equal to",
    r"AssertionError:.*!=.*expected",
    r"accuracy",
    r"score",
    r"latency",
    r"throughput",
    r"timeout",
]

# Patterns that indicate non-retriable failures (real code errors)
NON_RETRIABLE_PATTERNS = [
    r"SyntaxError",
    r"ImportError",
    r"ModuleNotFoundError",
    r"NameError",
    r"TypeError",
    r"AttributeError",
    r"RuntimeError",
    r"CUDA out of memory",
    r"OOM",
    r"Segmentation fault",
    r"core dumped",
    r"ConnectionRefusedError",
    r"FileNotFoundError",
]


def is_retriable_failure(output: str) -> tuple[bool, str]:
    """
    Determine if a test failure is retriable based on output patterns.

    Returns:
        tuple: (is_retriable, reason)
    """
    # Check for non-retriable patterns first
    for pattern in NON_RETRIABLE_PATTERNS:
        if re.search(pattern, output, re.IGNORECASE):
            return False, f"non-retriable error: {pattern}"

    # Check for retriable patterns
    for pattern in RETRIABLE_PATTERNS:
        if re.search(pattern, output, re.IGNORECASE):
            return True, f"retriable pattern: {pattern}"

    # If we have an AssertionError but didn't match non-retriable, assume retriable
    if re.search(r"AssertionError", output):
        return True, "AssertionError (assuming retriable)"

    # Default: not retriable
    return False, "unknown failure type"


def _get_ancestor_pids():
    """Return the set of PIDs from the current process up to PID 1."""
    try:
        import psutil
    except ImportError:
        return {os.getpid()}
    pids = set()
    try:
        proc = psutil.Process(os.getpid())
        while proc is not None:
            pids.add(proc.pid)
            proc = proc.parent()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    pids.add(os.getpid())
    return pids


def _kill_orphan_processes():
    """Kill leftover processes from previous test runs.

    After each test, tearDownClass should kill the server, but if a test
    crashes or times out, orphan processes may linger and consume GPU memory,
    system memory, or file descriptors, causing flaky failures in later tests.

    Strategy: In CI containers we can be aggressive. Kill every Python process
    that is NOT an ancestor of the current test-runner process (i.e. not our
    own process chain). This catches sglang servers, multiprocessing workers,
    torch.distributed workers, triton compilation daemons, etc.
    """
    try:
        import psutil
    except ImportError:
        logger.warning(
            "[cleanup] psutil not available, skipping orphan process cleanup"
        )
        return

    protected = _get_ancestor_pids()
    killed = []

    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            pid = proc.pid
            if pid in protected or pid <= 1:
                continue
            name = proc.info.get("name") or ""
            cmdline = " ".join(proc.info.get("cmdline") or [])
            is_python = "python" in name.lower()
            is_gpu = False
            if not is_python:
                try:
                    open_files = proc.open_files()
                    is_gpu = any(
                        "/dev/kfd" in f.path
                        or "/dev/dri" in f.path
                        or "/dev/nvidia" in f.path
                        for f in open_files
                    )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            if is_python or is_gpu:
                logger.info(
                    f"[cleanup] Killing orphan pid={pid} ({name}): {cmdline[:120]}"
                )
                kill_process_tree(pid)
                killed.append(pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if killed:
        logger.info(
            f"[cleanup] Killed {len(killed)} orphan process tree(s), waiting 5s..."
        )
        time.sleep(5)
    else:
        logger.info("[cleanup] No orphan processes found")


def _clear_shared_memory():
    """Remove leftover POSIX shared memory segments that may leak GPU/CPU resources.

    NCCL/RCCL can leave behind both files and directories in /dev/shm.
    """
    shm_dir = "/dev/shm"
    if not os.path.isdir(shm_dir):
        return
    uid = os.getuid()
    removed = 0
    for name in os.listdir(shm_dir):
        path = os.path.join(shm_dir, name)
        try:
            if os.stat(path).st_uid != uid:
                continue
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)
            removed += 1
        except OSError:
            continue
    if removed:
        logger.info(f"[cleanup] Removed {removed} shared memory entry(s) from /dev/shm")


def _log_resource_status():
    """Log disk, memory, and GPU status for post-mortem debugging."""
    lines = ["[cleanup] === Resource status ==="]
    for cmd, label in [
        (["free", "-h"], "Memory"),
        (["df", "-h", "/"], "Disk"),
    ]:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                lines.append(r.stdout.strip())
        except Exception:
            pass

    for gpu_cmd in [
        ["rocm-smi", "--showmemuse"],
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv"],
    ]:
        try:
            r = subprocess.run(gpu_cmd, capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                lines.append(r.stdout.strip())
                break
        except FileNotFoundError:
            continue
        except Exception:
            break

    lines.append("[cleanup] === End resource status ===")
    logger.info("\n".join(lines))


def _verify_python_env():
    """Quick sanity check that the Python environment is intact.

    If a previous test corrupted the environment (e.g., disk full causing
    partial writes, or stale .pyc files), catching it here gives a clear
    diagnostic instead of a cryptic failure deep in the next test.
    """
    checks = [
        "import transformers; from transformers import pytorch_utils",
        "import torch",
    ]
    for check in checks:
        try:
            r = subprocess.run(
                ["python3", "-c", check],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if r.returncode != 0:
                logger.warning(
                    f"[cleanup] ENV CHECK FAILED: `{check}`\n"
                    f"  stdout: {r.stdout.strip()}\n"
                    f"  stderr: {r.stderr.strip()}"
                )
            else:
                logger.info(f"[cleanup] ENV OK: `{check}`")
        except Exception as e:
            logger.warning(f"[cleanup] ENV CHECK ERROR: `{check}` -> {e}")


def cleanup_between_tests():
    """Run between test files to prevent resource leaks from causing flaky failures."""
    _kill_orphan_processes()
    _clear_shared_memory()
    _log_resource_status()
    _verify_python_env()


def run_with_timeout(
    func: Callable,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    timeout: float = None,
):
    """Run a function with timeout."""
    ret_value = []

    def _target_func():
        ret_value.append(func(*args, **(kwargs or {})))

    t = threading.Thread(target=_target_func)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        raise TimeoutError()

    if not ret_value:
        raise RuntimeError()

    return ret_value[0]


def write_github_step_summary(content: str):
    """Write content to GitHub Step Summary if available."""
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a") as f:
            f.write(content)


def run_unittest_files(
    files: Union[List[TestFile], List[CIRegistry]],
    timeout_per_file: float,
    continue_on_error: bool = False,
    enable_retry: bool = False,
    max_attempts: int = 2,
    retry_wait_seconds: int = 60,
):
    """
    Run a list of test files.

    Args:
        files: List of TestFile objects to run
        timeout_per_file: Timeout in seconds for each test file
        continue_on_error: If True, continue running remaining tests even if one fails.
                          If False, stop at first failure (default behavior for PR tests).
        enable_retry: If True, retry failed tests that appear to be accuracy/performance
                     assertion failures (not code errors).
        max_attempts: Maximum number of attempts per file including initial run (default: 2).
        retry_wait_seconds: Seconds to wait between retries (default: 60).
    """
    coredump_enabled = cuda_coredump.is_enabled()
    if coredump_enabled:
        cuda_coredump.cleanup_dump_dir()

    tic = time.perf_counter()
    success = True
    passed_tests = []
    failed_tests = []
    retried_tests = []  # Track which tests were retried

    for i, file in enumerate(files):
        if isinstance(file, CIRegistry):
            filename, estimated_time = file.filename, file.est_time
        else:
            # FIXME: remove this branch after migrating all tests to use CIRegistry
            filename, estimated_time = file.name, file.estimated_time

        if i > 0 and os.environ.get("SGLANG_IS_IN_CI_AMD") == "1":
            logger.info(
                f"\n[cleanup] Running AMD CI cleanup before test {i}/{len(files) - 1}: {filename}"
            )
            cleanup_between_tests()

        process = None
        output_lines = []

        def run_one_file(filename, capture_output=False):
            nonlocal process, output_lines

            full_path = os.path.join(os.getcwd(), filename)
            logger.info(
                f".\n.\nBegin ({i}/{len(files) - 1}):\npython3 {full_path}\n.\n.\n"
            )
            file_tic = time.perf_counter()

            if capture_output:
                # Capture output for retry decision
                process = subprocess.Popen(
                    ["python3", full_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    errors="ignore",  # Ignore non-UTF-8 bytes to prevent UnicodeDecodeError
                )
                output_lines = []
                for line in process.stdout:
                    logger.info(line.rstrip())
                    output_lines.append(line)
                process.wait()
            else:
                process = subprocess.Popen(
                    ["python3", full_path], stdout=None, stderr=None
                )
                process.wait()

            elapsed = time.perf_counter() - file_tic

            logger.info(
                f".\n.\nEnd ({i}/{len(files) - 1}):\n{filename=}, {elapsed=:.0f}, {estimated_time=}\n.\n.\n"
            )
            return process.returncode

        # Retry loop for each file
        attempt = 1
        file_passed = False
        was_retried = False

        while attempt <= (max_attempts if enable_retry else 1):
            if attempt > 1:
                logger.info(
                    f"\n[CI Retry] Attempt {attempt}/{max_attempts} for {filename}\n"
                )
                was_retried = True

            try:
                ret_code = run_with_timeout(
                    run_one_file,
                    args=(filename,),
                    kwargs={"capture_output": enable_retry},
                    timeout=timeout_per_file,
                )

                if ret_code == 0:
                    file_passed = True
                    if was_retried:
                        logger.info(
                            f"\n✓ PASSED on retry (attempt {attempt}): {filename}\n"
                        )
                        retried_tests.append((filename, attempt, "passed"))
                    passed_tests.append(filename)
                    break
                else:
                    # Check if we should retry
                    if enable_retry and attempt < max_attempts:
                        output = "".join(output_lines)
                        is_retriable, reason = is_retriable_failure(output)

                        if is_retriable:
                            logger.info(f"\n[CI Retry] {filename} failed with {reason}")
                            logger.info(
                                f"[CI Retry] Waiting {retry_wait_seconds}s before retry...\n"
                            )
                            time.sleep(retry_wait_seconds)
                            attempt += 1
                            continue
                        else:
                            logger.info(
                                f"\n[CI Retry] {filename} failed with {reason} - not retrying\n"
                            )

                    # No retry or not retriable
                    logger.info(
                        f"\n✗ FAILED: {filename} returned exit code {ret_code}\n"
                    )
                    if was_retried:
                        retried_tests.append((filename, attempt, "failed"))
                    failed_tests.append((filename, f"exit code {ret_code}"))
                    break

            except TimeoutError:
                kill_process_tree(process.pid)
                time.sleep(5)
                logger.info(
                    f"\n✗ TIMEOUT: {filename} after {timeout_per_file} seconds\n"
                )
                if was_retried:
                    retried_tests.append((filename, attempt, "timeout"))
                failed_tests.append((filename, f"timeout after {timeout_per_file}s"))
                break

        if not file_passed:
            success = False
            if not continue_on_error:
                break

    elapsed_total = time.perf_counter() - tic

    if coredump_enabled and not success:
        cuda_coredump.report()

    if success:
        logger.info(f"Success. Time elapsed: {elapsed_total:.2f}s")
    else:
        logger.info(f"Fail. Time elapsed: {elapsed_total:.2f}s")

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Test Summary: {len(passed_tests)}/{len(files)} passed")
    if enable_retry and retried_tests:
        logger.info(f"Retries: {len(retried_tests)} test(s) were retried")
    logger.info(f"{'='*60}")
    if passed_tests:
        logger.info("✓ PASSED:")
        for test in passed_tests:
            logger.info(f"  {test}")
    if failed_tests:
        logger.info("\n✗ FAILED:")
        for test, reason in failed_tests:
            logger.info(f"  {test} ({reason})")
    if retried_tests:
        logger.info("\n↻ RETRIED:")
        for test, attempts, result in retried_tests:
            logger.info(f"  {test} ({attempts} attempts, {result})")
    logger.info(f"{'='*60}\n")

    # Write GitHub Step Summary only if retries occurred
    if retried_tests:
        passed_on_retry = [t for t, _, r in retried_tests if r == "passed"]
        failed_after_retry = [t for t, _, r in retried_tests if r != "passed"]
        summary = f"**↻ Retried {len(retried_tests)} test(s):**\n"
        if passed_on_retry:
            summary += f"- ✓ Passed on retry: {', '.join(passed_on_retry)}\n"
        if failed_after_retry:
            summary += f"- ✗ Still failed: {', '.join(failed_after_retry)}\n"
        write_github_step_summary(summary)

    return 0 if success else -1
