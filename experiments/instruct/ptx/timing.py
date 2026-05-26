from time import perf_counter


"""
Based on https://stackoverflow.com/a/69156219
"""

def func(in_fn):
    def out_fn(*args, **kwargs):
        start = perf_counter()
        out = in_fn(*args, **kwargs)
        elapsed_time = perf_counter() - start
        print(f'[timing] {in_fn.__name__}: {elapsed_time:.2f}s')
        return out
    return out_fn


class context:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.start = perf_counter()

    def __exit__(self, type, value, traceback):
        elapsed_time = perf_counter() - self.start
        print(f'[timing] {self.label}: {elapsed_time:.2f}s')
