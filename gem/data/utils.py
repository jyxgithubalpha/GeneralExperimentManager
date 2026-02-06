from typing import List


def remove_quotes_from_list(items: List[str]) -> List[str]:
    """去除字符串列表中的引号"""
    return [item.strip('"\'') for item in items]