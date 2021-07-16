import numpy as np
from collections import defaultdict
from copy import deepcopy
import ray


class ReplayMemory(object):
    def __init__(self, config):
        self.config = config
        self.maxSize = self.config.BUFFER_SIZE
        self.batch_size = self.config.BATCH_SIZE
        self.reset()
        self.nSteps = self.config.REPLAY_NSTEP
        self.e = 0.001
        self.beta_increment = 0.001
        self.prio = self.config.REPLAY_PRIO

    def __len__(self):
        return len(self.buffer)

    def _evict(self):
        """Simplest FIFO eviction scheme."""
        if len(self) > self.maxSize + 100:
            del self.buffer[:-self.maxSize]
            if self.prio:
                self.priorities = self.priorities[-self.maxSize:]
            return True
        return False

    def append(self, entID, s, actions, r, dead):
        self.agents[entID].append((s, actions, r, dead))

    def push(self):
        dead = []
        for entID, tpl in self.agents.items():
            if tpl[-1][3]:
                dead.append(entID)
        for entID in dead:
            batch = self.agents.pop(entID)
            self.append_batch(batch)

    def append_batch(self, batch, priorities=None):
        self.buffer.extend(batch)
        if self.prio:
            if priorities is None:
                priorities = np.ones(len(batch), dtype='float32') * np.max(self.priorities) if \
                    len(self.priorities) > 0 else np.ones(len(batch), dtype='float32')
            self.priorities = np.append(self.priorities, priorities)

    def sample(self):
        idx, weights = self.select_batch()

        samples_of_samples = []
        for i in idx:
            samples_of_samples.append(self.buffer[i: i + 1 + self.nSteps])
        return idx, samples_of_samples, weights

    def update_priorities(self, idx, priorities):
        if self.prio:
            self.priorities[idx] = priorities
        return self._evict()

    def select_batch(self):
        l = len(self)
        batch_size = min(self.batch_size, l)
        if self.prio:
            self.beta = min(1, self.beta + self.beta_increment)
            self.priorities += self.e

            probs = self.priorities[:l - 1 - self.nSteps]
            probs /= probs.sum()
            idx = np.random.choice(l - 1 - self.nSteps, batch_size, p=probs)
            weights = np.apply_along_axis(self.calculate_weight, 0, probs[idx])
            weights /= weights.max()
        else:
            idx, weights = np.random.choice(l - 1 - self.nSteps, batch_size), None
        return idx, weights

    def calculate_weight(self, prob):
        return (len(self) * prob) ** -self.beta

    def reset(self):
        self.clear_buffer()
        self.agents = defaultdict(list)
        self.beta = 0.4

    def clear_buffer(self):
        self.buffer = []
        self.priorities = np.array([], dtype='float32')

    def send_buffer(self):
        buffer = deepcopy(self.buffer)
        self.clear_buffer()
        return buffer


@ray.remote
class ReplayMemoryMaster:  # ReplayMemory per agent
    def __init__(self, args, config):
        self.args, self.config = args, config
        self.memories = [ReplayMemory(self.config) for _ in range(config.NPOP)]

    def __len__(self):
        return len(self.memories[0])

    def len(self):
        return len(self)

    def sample(self):
        samples = [memory.sample() for memory in self.memories]
        idx, sample, weights = list(zip(*samples))
        return idx, sample, weights

    def update(self, updates):
        buffers, priorities = updates
        [memory.append_batch(buffer, priority) for memory, buffer, priority in zip(self.memories, buffers, priorities)]

    def update_priorities(self, idx, priorities):
        return [memory.update_priorities(i, priority) for memory, priority, i in zip(self.memories, priorities, idx)]


class ReplayMemoryLm(ReplayMemory):
    def __init__(self, config):
        super(ReplayMemoryLm, self).__init__(config)
        self.batch_size = self.config.BATCH_SIZE_LM
        self.nSteps = self.config.REPLAY_NSTEP_LM
        self.prio = False
        self.maxSize = self.config.BUFFER_SIZE_LM

    def append(self, entID, annID, s, actions, r, dead):
        self.agents[entID] = [s, annID, actions, r, dead]

    def push(self):
        batch = deepcopy(self.agents)
        self.append_batch([batch])
        self.agents = dict()

    def select_batch(self):
        l = len(self)
        batch_size = min(self.batch_size, l)
        if self.config.REPLAY_LM:
            idx = np.random.choice(l - 1 - self.nSteps, batch_size)
        else:
            idx = np.arange(l - 1 - self.nSteps)
        return idx, None


@ray.remote
class ReplayMemoryLmMaster:  # ReplayMemoryLm per worker
    def __init__(self, args, config):
        self.args, self.config = args, config
        self.memories = [ReplayMemoryLm(self.config)
                         for _ in range(args.nEnvs)]

    def __len__(self):
        return len(self.memories[0])

    def len(self):
        return len(self)

    def sample(self):
        ints = list(range(len(self.memories)))
        for _ in range(len(ints)):
            i = np.random.choice(ints, 1).item()
            if len(self.memories[i]) > 0:
                self.memories[i]._evict()
                ans = self.memories[i].sample(), i
                if not self.config.REPLAY_LM:
                    self.memories[i].reset()
                return ans
            ints.remove(i)
        return None, None

    def update(self, update, i):
        self.memories[i].append_batch(update)
