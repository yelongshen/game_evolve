import threading
import random


class GlobalReplayBuffer:
    def __init__(self, capacity=int(1e5)):
        self.capacity = capacity
        self.storage = []
        self.lock = threading.Lock()

    def add(self, item):
        with self.lock:
            if len(self.storage) >= self.capacity:
                self.storage.pop(0)
            self.storage.append(item)

    def extend(self, items):
        with self.lock:
            for item in items:
                if len(self.storage) >= self.capacity:
                    self.storage.pop(0)
                self.storage.append(item)

    def sample(self, batch_size):
        with self.lock:
            n = len(self.storage)
            if n == 0:
                return []
            idxs = random.sample(range(n), min(batch_size, n))
            return [self.storage[i] for i in idxs]
        
    def pop_sample(self, batch_size):
        """Sample without replacement AND remove items from storage."""
        with self.lock:
            n = len(self.storage)
            if n == 0:
                return []
            idxs = random.sample(range(n), min(batch_size, n))
            # sort descending so we can pop safely
            idxs.sort(reverse=True)
            batch = []
            for i in idxs:
                batch.append(self.storage.pop(i))
            return batch