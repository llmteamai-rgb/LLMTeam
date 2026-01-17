#!/usr/bin/env python3
"""
Memory-safe test runner for llmteam.

This script runs tests in groups to prevent memory exhaustion.
It automatically handles cleanup and provides different modes.

Usage:
    # Run all tests sequentially (safest)
    python run_tests.py

    # Run with limited parallelism (balanced)
    python run_tests.py --parallel 2

    # Run specific module
    python run_tests.py --module tenancy

    # Run fast tests only
    python run_tests.py --fast

    # Run with coverage
    python run_tests.py --coverage
"""

import sys
import subprocess
import argparse
from pathlib import Path


# Test modules in dependency order
TEST_MODULES = [
    "tenancy",
    "audit",
    "context",
    "ratelimit",
    "licensing",
    "execution",
    "roles",
    "actions",      # v1.9.0
    "human",        # v1.9.0
    "persistence",  # v1.9.0
    "runtime",      # v2.0.0
    "events",       # v2.0.0
    "canvas",       # v2.0.0
]


def run_command(cmd: list[str], env_update: dict = None) -> int:
    """Run a command and return exit code."""
    import os

    env = os.environ.copy()
    if env_update:
        env.update(env_update)

    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, env=env)
    return result.returncode


def run_tests_sequential(coverage: bool = False, module: str = None) -> int:
    """Run tests sequentially (safest for memory)."""
    base_cmd = [sys.executable, "-m", "pytest", "-v", "--tb=short"]

    if coverage:
        base_cmd.extend(["--cov=llmteam", "--cov-report=term-missing"])

    # Set PYTHONPATH
    env_update = {"PYTHONPATH": "src"}

    if module:
        # Run specific module
        cmd = base_cmd + [f"tests/{module}/"]
        return run_command(cmd, env_update)
    else:
        # Run all modules one by one
        failed_modules = []

        for mod in TEST_MODULES:
            test_dir = Path("tests") / mod
            if not test_dir.exists():
                continue

            cmd = base_cmd + [str(test_dir)]
            exit_code = run_command(cmd, env_update)

            if exit_code != 0:
                failed_modules.append(mod)

        if failed_modules:
            print(f"\n{'='*60}")
            print(f"FAILED MODULES: {', '.join(failed_modules)}")
            print(f"{'='*60}\n")
            return 1

        return 0


def run_tests_parallel(workers: int = 2, coverage: bool = False) -> int:
    """Run tests with limited parallelism."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-v",
        "--tb=short",
        f"-n={workers}",
        "--dist=loadgroup",
    ]

    if coverage:
        cmd.extend(["--cov=llmteam", "--cov-report=term-missing"])

    cmd.append("tests/")

    env_update = {"PYTHONPATH": "src"}
    return run_command(cmd, env_update)


def run_tests_fast() -> int:
    """Run only fast unit tests."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-v",
        "--tb=short",
        "-m",
        "unit",
        "tests/",
    ]

    env_update = {"PYTHONPATH": "src"}
    return run_command(cmd, env_update)


def main():
    parser = argparse.ArgumentParser(description="Memory-safe test runner")
    parser.add_argument(
        "--parallel",
        type=int,
        default=0,
        help="Number of parallel workers (0 = sequential, default)",
    )
    parser.add_argument(
        "--module",
        choices=TEST_MODULES,
        help="Run specific module only",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run only fast unit tests",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report",
    )

    args = parser.parse_args()

    # Ensure we're in the right directory
    if not Path("pyproject.toml").exists():
        print("Error: Must run from llmteam/ directory")
        return 1

    # Choose test mode
    if args.fast:
        return run_tests_fast()
    elif args.parallel > 0:
        return run_tests_parallel(args.parallel, args.coverage)
    else:
        return run_tests_sequential(args.coverage, args.module)


if __name__ == "__main__":
    sys.exit(main())
