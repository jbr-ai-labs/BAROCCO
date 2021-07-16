import ray

from source.env import core


class Worker:
    def __init__(self, config, args):
        self.config, self.args = config, args
        self.envs = {i: core.Realm.remote(config, i) for i in range(args.nEnvs)}
        self.tasks = [e.run.remote(None) for e in self.envs.values()]

    def append(self, idx, update):
        self.tasks.append(self.envs[idx].run.remote(update))

    def run(self):
        done_id, tasks = ray.wait(self.tasks)
        self.tasks = tasks
        recvs = ray.get(done_id)[0]
        return recvs


class Runner:
    def __init__(self, config, args, agent):
        ray.init()
        self.learner = agent.Learner(config)
        self.env = Worker(config, args)

    def run(self):
        while True:
            recvs = self.env.run()
            idx = recvs[0]
            recvs = recvs[1:]
            self.learner.step(*recvs)
            self.env.append(idx, self.learner.model())
