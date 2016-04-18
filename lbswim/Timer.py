import time

class Timer(object):
    """Simple timer class.
    """
    def __init__(self):
        self._running = False

    def Start(self):
        assert not self._running
        self._running = True
        self._startTime = time.time()
        return

    def Stop(self):
        assert self._running
        self._stopTime = time.time()
        self._running = False
        return

    def GetTime(self):
        if self._running:
            return time.time() - self._startTime
        return self._stopTime - self._startTime

    pass
