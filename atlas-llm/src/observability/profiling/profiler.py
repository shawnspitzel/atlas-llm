import os
import sys
import signal
import atexit
import cProfile
import pstats
from datetime import datetime
from typing import Callable, Optional, Any
from line_profiler import LineProfiler


class ProfileManager:
    """Manages cProfile and line_profiler for training/inference profiling.

    Usage:
        profiler = ProfileManager(profile_dir="src/benchmarks/profiling", base_name="pretrain")
        profiler.add_function(my_function)  # Add functions for line profiling

        with profiler:
            # Your code here
            result = profiler.wrap(main_function)(args)
    """

    def __init__(self, profile_dir: str = "src/benchmarks/profiling", base_name: str = "profile"):
        self.profile_dir = profile_dir
        self.base_name = base_name
        self.cprofile: Optional[cProfile.Profile] = None
        self.line_profiler: Optional[LineProfiler] = None
        self._original_sigint = None
        self._original_sigterm = None
        self._active = False

        os.makedirs(profile_dir, exist_ok=True)

    def add_function(self, func: Callable) -> None:
        """Add a function to be profiled by line_profiler."""
        if self.line_profiler is None:
            self.line_profiler = LineProfiler()
        self.line_profiler.add_function(func)

    def wrap(self, func: Callable) -> Callable:
        """Wrap a function with line_profiler."""
        if self.line_profiler is None:
            self.line_profiler = LineProfiler()
        return self.line_profiler(func)

    def __enter__(self) -> "ProfileManager":
        self._start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is None:
            self._save_results(interrupted=False)
        self._cleanup()
        return False

    def _start(self) -> None:
        """Start profiling and register signal handlers."""
        self.cprofile = cProfile.Profile()
        if self.line_profiler is None:
            self.line_profiler = LineProfiler()

        self._active = True

        # Register signal handlers
        self._original_sigint = signal.signal(signal.SIGINT, self._signal_handler)
        self._original_sigterm = signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self._atexit_handler)

        self.cprofile.enable()

    def _signal_handler(self, signum, frame) -> None:
        """Handle interrupt signals by saving profile data."""
        print("\n\nReceived interrupt signal, saving profiling data...")
        self._save_results(interrupted=True)
        self._cleanup()
        sys.exit(0)

    def _atexit_handler(self) -> None:
        """Save profile data on exit if still active."""
        if self._active and self.cprofile is not None:
            self._save_results(interrupted=True)

    def _save_results(self, interrupted: bool = False) -> None:
        """Save profiling results to files."""
        if self.cprofile is None:
            return

        self.cprofile.disable()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "_interrupted" if interrupted else ""
        status = "INTERRUPTED" if interrupted else ""

        # Save cProfile text results
        cprofile_file = os.path.join(
            self.profile_dir,
            f"{self.base_name}_cprofile_{timestamp}{suffix}.txt"
        )
        with open(cprofile_file, 'w') as f:
            stats = pstats.Stats(self.cprofile, stream=f)
            f.write("=" * 80 + "\n")
            f.write(f"FUNCTION-LEVEL PROFILE (cProfile){' - ' + status if status else ''}\n")
            f.write("=" * 80 + "\n\n")
            f.write("Top 30 functions by cumulative time:\n")
            f.write("-" * 80 + "\n")
            stats.sort_stats('cumulative')
            stats.print_stats(30)
            f.write("\n\nTop 20 functions by total time:\n")
            f.write("-" * 80 + "\n")
            stats.sort_stats('time')
            stats.print_stats(20)

        # Save line_profiler results
        lineprofile_file = os.path.join(
            self.profile_dir,
            f"{self.base_name}_lineprofile_{timestamp}{suffix}.txt"
        )
        with open(lineprofile_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"LINE-BY-LINE PROFILE (line_profiler){' - ' + status if status else ''}\n")
            f.write("=" * 80 + "\n\n")
            self.line_profiler.print_stats(stream=f)

        # Save binary cProfile data
        cprofile_binary = os.path.join(
            self.profile_dir,
            f"{self.base_name}_cprofile_{timestamp}{suffix}.prof"
        )
        self.cprofile.dump_stats(cprofile_binary)

        # Print summary
        print("\n" + "=" * 80)
        print(f"PROFILING {'SAVED (Training Interrupted)' if interrupted else 'COMPLETE'}")
        print("=" * 80)
        print(f"\nProfile results saved to:")
        print(f"  1. Function-level (cProfile): {cprofile_file}")
        print(f"  2. Line-by-line (line_profiler): {lineprofile_file}")
        print(f"  3. Binary cProfile data: {cprofile_binary}")
        if not interrupted:
            print(f"\nTo view binary cProfile data interactively:")
            print(f"  python -m pstats {cprofile_binary}")
        print("=" * 80 + "\n")

        self._active = False

    def _cleanup(self) -> None:
        """Restore original signal handlers and cleanup."""
        self._active = False

        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)
        if self._original_sigterm is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm)

        try:
            atexit.unregister(self._atexit_handler)
        except Exception:
            pass

        self.cprofile = None
