import re
import cdifflib

def non_space_diff_ratio(a, b):
    """Get distance between two strings, ignoring all whitespace"""
    a = re.sub(r'\s', '', a)
    b = re.sub(r'\s', '', b)
    return cdifflib.CSequenceMatcher(None, a, b).ratio()

def closest_in_list(a, lst, key=lambda x: x):
    """Find closest string in list to a"""
    closest = None
    closest_ratio = -1
    for b in lst:
        comp = key(b)
        ratio = non_space_diff_ratio(a, comp)
        if ratio > closest_ratio:
            closest_ratio = ratio
            closest = b
    return closest, closest_ratio
