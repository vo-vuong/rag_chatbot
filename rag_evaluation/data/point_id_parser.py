"""
Point ID parsing utilities.

Handles parsing of Point_ids from various formats in test data files.
"""

import ast
from typing import List, Union

import pandas as pd


def parse_point_ids(point_ids_value: Union[int, str, list, float, None]) -> List[int]:
    """
    Parse Point_ids from various formats to list of integers.

    Handles multiple input formats commonly found in Excel/CSV test data:
    - Single integer: 5 -> [5]
    - String of single int: "5" -> [5]
    - String list: "[1, 2, 3]" -> [1, 2, 3]
    - Actual list: [1, 2, 3] -> [1, 2, 3]
    - Comma-separated: "1, 2, 3" -> [1, 2, 3]
    - None/NaN: -> []

    Args:
        point_ids_value: Raw value from DataFrame column

    Returns:
        List of integer point IDs

    Examples:
        >>> parse_point_ids(5)
        [5]
        >>> parse_point_ids("[1, 2, 3]")
        [1, 2, 3]
        >>> parse_point_ids("1, 2, 3")
        [1, 2, 3]
        >>> parse_point_ids(None)
        []
    """
    if point_ids_value is None:
        return []

    if isinstance(point_ids_value, float) and pd.isna(point_ids_value):
        return []

    if isinstance(point_ids_value, int):
        return [point_ids_value]

    if isinstance(point_ids_value, list):
        return [int(x) for x in point_ids_value]

    if isinstance(point_ids_value, str):
        point_ids_value = point_ids_value.strip()

        # Try parsing as Python literal (list format)
        if point_ids_value.startswith("["):
            try:
                parsed = ast.literal_eval(point_ids_value)
                return [int(x) for x in parsed]
            except (ValueError, SyntaxError):
                pass

        # Try comma-separated format
        if "," in point_ids_value:
            try:
                return [int(x.strip()) for x in point_ids_value.split(",") if x.strip()]
            except ValueError:
                return []

        # Single value string
        try:
            return [int(point_ids_value)]
        except ValueError:
            return []

    return []
