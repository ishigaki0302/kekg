"""I/O utilities for loading and saving data files."""

import json
from pathlib import Path
from typing import Any, Dict, List, Iterator, Union
import yaml


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        path: Path to YAML file

    Returns:
        Parsed YAML content as dictionary
    """
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Save dictionary to YAML file.

    Args:
        data: Dictionary to save
        path: Output path
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)


def load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load JSONL file (one JSON object per line).

    Args:
        path: Path to JSONL file

    Returns:
        List of parsed JSON objects
    """
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def iter_jsonl(path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
    """
    Iterate over JSONL file without loading all into memory.

    Args:
        path: Path to JSONL file

    Yields:
        Parsed JSON objects one at a time
    """
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line.strip())


def save_jsonl(data: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    """
    Save list of dictionaries to JSONL file.

    Args:
        data: List of dictionaries to save
        path: Output path
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_json(path: Union[str, Path]) -> Any:
    """
    Load JSON file.

    Args:
        path: Path to JSON file

    Returns:
        Parsed JSON content
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to JSON file.

    Args:
        data: Data to save
        path: Output path
        indent: Indentation level
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
