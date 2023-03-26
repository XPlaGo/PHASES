import sys


def print_progress(step, frames, title, end):
    p = (step * 10) // frames
    sys.stdout.write(f"\r{title}: |" + ("█" * p) + (" " * (10 - p)) + ("| %i" % step) + (" from %i" % frames))
    if step == frames:
        sys.stdout.write("\n")
        print(end)
    sys.stdout.flush()
