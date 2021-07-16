import ray

from source.env import core


class Worker:
    def __init__(self, config, args, agent):
        self.config, self.args = config, args
        self.envs = {i: core.NativeRealm.remote(agent, config, args, i) for i in range(args.nEnvs)}
        self.tasks = [e.run.remote(None) for e in self.envs.values()]

    def append(self, idx, update):
        self.tasks.append(self.envs[idx].run.remote(update))

    def run(self):
        done_id, tasks = ray.wait(self.tasks)
        self.tasks = tasks
        recvs = ray.get(done_id)[0]
        return recvs

    def send(self, swordUpdate):
        [e.recvSwordUpdate.remote(swordUpdate) for e in self.envs]


class Runner:
    def __init__(self, config, args, agent):
        ray.init()
        self.pantheon = agent.Learner(config, args)
        self.env = Worker(config, args, agent)

    def run(self):
        while True:
            recvs = self.env.run()
            idx = recvs[0]
            recvs = recvs[1:]
            self.pantheon.step(*recvs)
            self.env.append(idx, self.pantheon.model())
