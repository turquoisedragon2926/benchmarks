import subprocess
import time


def plot_graph(z):
    return # temp
    with open("t.dot", "w") as f:
        f.write(z.as_hlo_dot_graph())
    with open("t.png", "wb") as f:
        subprocess.run(["dot", "t.dot", "-Tpng"], stdout=f)


class Timing:
    def __init__(self,print_func):
        self.start = time.perf_counter()
        self.last = self.start
        self.print_func=print_func

    def log(self, message: str) -> None:
        now = time.perf_counter()
        delta = now - self.start
        self.print_func(f"{delta:.4f} / {now - self.last:.4f}: {message}")
        self.last = now
