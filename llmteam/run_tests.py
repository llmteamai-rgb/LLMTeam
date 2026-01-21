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
import os
import subprocess
import argparse
import shlex
from pathlib import Path
from typing import List, Optional, Dict


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
    "engine",       # v5.0.0 (formerly canvas)
    "providers",    # v2.0.3
    "testing",      # v2.0.3
    "quality",      # RFC-008
]


class TestRunner:
    """Handles test execution configuration and running."""

    def __init__(self, parallel: int = 0, coverage: bool = False, fast: bool = False):
        self.parallel = parallel
        self.coverage = coverage
        self.fast = fast
        self.env = os.environ.copy()
        self.env["PYTHONPATH"] = "src"

    def _log(self, message: str, header: bool = False) -> None:
        """Print formatted logs."""
        if header:
            print(f"\n{'='*60}")
            print(f"{message}")
            print(f"{'='*60}\n")
        else:
            print(message)

    def _build_base_cmd(self) -> List[str]:
        """Build the base pytest command."""
        cmd = [sys.executable, "-m", "pytest", "-v", "--tb=short"]
        
        if self.coverage:
            cmd.extend(["--cov=llmteam", "--cov-report=term-missing"])
            
        return cmd

    def run_command(self, cmd: List[str]) -> int:
        """Run a subprocess command and return exit code."""
        cmd_str = ' '.join(shlex.quote(s) for s in cmd)
        self._log(f"Running: {cmd_str}", header=True)
        
        result = subprocess.run(cmd, env=self.env)
        return result.returncode

    def run_sequential(self, module: Optional[str] = None) -> int:
        """Run tests modules sequentially."""
        base_cmd = self._build_base_cmd()

        if module:
            target = Path("tests") / module
            if not target.exists():
                self._log(f"Error: Module {module} not found at {target}")
                return 1
            return self.run_command(base_cmd + [str(target)])
        
        failed_modules = []
        for mod in TEST_MODULES:
            test_dir = Path("tests") / mod
            if not test_dir.exists():
                self._log(f"Skipping {mod} (directory not found)")
                continue

            exit_code = self.run_command(base_cmd + [str(test_dir)])
            if exit_code != 0:
                failed_modules.append(mod)

        if failed_modules:
            self._log(f"FAILED MODULES: {', '.join(failed_modules)}", header=True)
            return 1
            
        return 0

    def run_parallel(self) -> int:
        """Run tests with limited parallelism using pytest-xdist."""
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-v",
            "--tb=short",
            f"-n={self.parallel}",
            "--dist=loadgroup",
        ]

        if self.coverage:
            cmd.extend(["--cov=llmteam", "--cov-report=term-missing"])

        cmd.append("tests/")
        return self.run_command(cmd)

    def run_fast(self) -> int:
        """Run only tests marked as 'unit'."""
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
        return self.run_command(cmd)

    def execute(self, module: Optional[str] = None) -> int:
        """Execute tests based on configuration."""
        if self.fast:
            return self.run_fast()
        elif self.parallel > 0:
            if module:
                self._log("Warning: --module ignored when running in parallel mode")
            return self.run_parallel()
        else:
            return self.run_sequential(module)


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

    runner = TestRunner(
        parallel=args.parallel,
        coverage=args.coverage,
        fast=args.fast
    )
    
    return runner.execute(module=args.module)


if __name__ == "__main__":
    sys.exit(main())
