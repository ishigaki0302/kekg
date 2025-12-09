"""Logging utilities for progress and metrics reporting."""

import sys
from typing import Any, Dict, Optional, Union
from datetime import datetime
from pathlib import Path
import json


class Logger:
    """Simple logger for experiment progress and metrics."""

    def __init__(self, log_file: Optional[Union[str, Path]] = None, verbose: bool = True):
        """
        Initialize logger.

        Args:
            log_file: Optional file path to write logs to
            verbose: Whether to print to stdout
        """
        self.log_file = Path(log_file) if log_file else None
        self.verbose = verbose

        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def info(self, message: str) -> None:
        """Log informational message."""
        self._log("INFO", message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self._log("WARN", message)

    def error(self, message: str) -> None:
        """Log error message."""
        self._log("ERROR", message)

    def metric(self, name: str, value: Any, step: Optional[int] = None) -> None:
        """
        Log a metric value.

        Args:
            name: Metric name
            value: Metric value
            step: Optional step/iteration number
        """
        step_str = f" [step={step}]" if step is not None else ""
        self._log("METRIC", f"{name}={value}{step_str}")

    def metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log multiple metrics at once.

        Args:
            metrics: Dictionary of metric name -> value
            step: Optional step/iteration number
        """
        step_str = f" [step={step}]" if step is not None else ""
        metrics_str = " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                   for k, v in metrics.items())
        self._log("METRICS", f"{metrics_str}{step_str}")

    def _log(self, level: str, message: str) -> None:
        """Internal logging method."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {level}: {message}"

        if self.verbose:
            print(log_line, file=sys.stdout)

        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_line + '\n')


class MetricsTracker:
    """Track metrics over time and export to various formats."""

    def __init__(self):
        self.history: Dict[str, list] = {}

    def add(self, name: str, value: Any, step: Optional[int] = None) -> None:
        """
        Add a metric value.

        Args:
            name: Metric name
            value: Metric value
            step: Optional step number
        """
        if name not in self.history:
            self.history[name] = []

        self.history[name].append({
            'value': value,
            'step': step,
            'timestamp': datetime.now().isoformat()
        })

    def get_latest(self, name: str) -> Optional[Any]:
        """Get the latest value for a metric."""
        if name not in self.history or not self.history[name]:
            return None
        return self.history[name][-1]['value']

    def get_all(self, name: str) -> list:
        """Get all values for a metric."""
        return self.history.get(name, [])

    def save_json(self, path: Union[str, Path]) -> None:
        """Save all metrics to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

    def save_csv(self, path: Union[str, Path]) -> None:
        """Save metrics to CSV file (one row per step)."""
        import csv

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Collect all unique steps
        steps = set()
        for entries in self.history.values():
            for entry in entries:
                if entry['step'] is not None:
                    steps.add(entry['step'])

        steps = sorted(steps)

        # Build table
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header
            header = ['step'] + list(self.history.keys())
            writer.writerow(header)

            # Data rows
            for step in steps:
                row = [step]
                for metric_name in self.history.keys():
                    # Find value for this step
                    value = None
                    for entry in self.history[metric_name]:
                        if entry['step'] == step:
                            value = entry['value']
                            break
                    row.append(value if value is not None else '')
                writer.writerow(row)
