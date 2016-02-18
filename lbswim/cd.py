import os
from contextlib import contextmanager

@contextmanager
def cd(dst_dir):
    """Let's you write:
    with cd('some/directory'):
        # Do stuff in some/directory

        pass
    
    # Go back to the directory you were in before

    Quite useful for managing run directories etc.
    """
    current_dir = os.getcwd()
    os.chdir(dst_dir)
    try:
        yield dst_dir
    finally:
        os.chdir(current_dir)
