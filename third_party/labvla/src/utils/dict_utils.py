"""Generic nested-dict helpers (no project dependencies).

``dataset/utils.py`` re-exports both names so existing importers keep working.
"""
from __future__ import annotations


def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    """Flatten a nested dictionary by joining keys with a separator.

    Example:
        >>> dct = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}
        >>> print(flatten_dict(dct))
        {'a/b': 1, 'a/c/d': 2, 'e': 3}

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str): The base key to prepend to the keys in this level.
        sep (str): The separator to use between keys.

    Returns:
        dict: A flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: dict, sep: str = "/") -> dict:
    """Unflatten a dictionary with delimited keys into a nested dictionary.

    Example:
        >>> flat_dct = {"a/b": 1, "a/c/d": 2, "e": 3}
        >>> print(unflatten_dict(flat_dct))
        {'a': {'b': 1, 'c': {'d': 2}}, 'e': 3}

    Args:
        d (dict): A dictionary with flattened keys.
        sep (str): The separator used in the keys.

    Returns:
        dict: A nested dictionary.
    """
    outdict = {}
    for key, value in d.items():
        parts = key.split(sep)
        d = outdict
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return outdict
