'''Contains various useful utility functions.'''

def round5_down(value):
    return int(value) - (int(value) % 5)

def max_index(lst, key=None):
    max_idx = 0
    max_val = getattr(lst[0], key)
    for idx in range(1, len(lst)):
        if key(lst[idx]) > max_val:
            max_idx = idx
            max_val = key(lst[idx])
    return max_idx
