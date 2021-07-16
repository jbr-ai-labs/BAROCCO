import ray
from time import sleep

from source.env import core
from source.env.lib.log import Quill
from source.utils.torch.replay import ReplayMemoryMaster, ReplayMemoryLmMaster


@ray.remote
class RemoteStateDict:
    def __init__(self, config):
        self.config = config
        self.stateDict, self.stateDictTarget, self.stateDictLm = None, None, None
        self.counter = -config.MIN_BUFFER
        self.collectFlag = True

    def updateAnns(self, state):
        self.stateDict = state

    def updateTargetAnns(self, state):
        self.stateDictTarget = state

    def updateLm(self, state):
        self.stateDictLm = state

    def sendAnns(self):
        return self.stateDict

    def sendTargetAnns(self):
        return self.stateDictTarget

    def sendLm(self):
        return self.stateDictLm

    def send(self):
        return self.sendAnns(), self.sendLm()

    def increaseCounter(self, n):
        self.counter += n
        return self.counter

    def decreaseCounter(self, n):
        self.counter -= n / 4
        if self.counter > 0:
            self.collectFlag = False
        else:
            self.collectFlag = True
        return self.counter

    def getFlag(self):
        return self.collectFlag

    def count(self):
        return self.counter


class Worker:
    def __init__(self, config, args):
        self.envs = [core.Realm.remote(config, args, i) for i in range(args.nEnvs)]
        self.tasks = [e.run.remote(None) for e in self.envs]
        self.tasks_eval = []

    def clientData(self):
        return self.envs[0].clientData.remote()

    def step(self):
        recvs = [e.step.remote() for e in self.envs]
        return ray.get(recvs)

    def run(self):
        done_id, self.tasks = ray.wait(self.tasks)
        recvs = ray.get(done_id)[0]
        return recvs

    def run_eval(self):
        done_id, self.tasks_eval = ray.wait(self.tasks_eval)
        recvs = ray.get(done_id)[0]
        return recvs

    def append(self, idx, update):
        self.tasks.append(self.envs[idx].run.remote(update))

    def append_eval(self, idx, update):
        self.tasks_eval.append(self.envs[idx].run.remote(update, evaluate=True))


class Runner:
    def __init__(self, config, args, agent):
        self.config, self.args = config, args
        self.agent = agent

        self.renderStep = self.step
        ray.init()
        self.idx = 0

    def run(self):
        sharedReplay = ReplayMemoryMaster.remote(self.args, self.config)
        sharedReplayLm = ReplayMemoryLmMaster.remote(self.args, self.config)
        sharedStateDict = RemoteStateDict.remote(self.config)

        self.run_actors.remote(self, sharedReplay, sharedReplayLm, sharedStateDict)

        pantheonProcessId = self.run_learner.remote(self, sharedReplay, sharedStateDict)
        if self.config.REPLAY_LM:
            self.run_learner_social.remote(self, sharedReplayLm, sharedStateDict)
        else:
            self.run_learner_social_online.remote(self, sharedReplayLm, sharedStateDict)
        ray.get(pantheonProcessId)

    @ray.remote(num_gpus=0)
    def run_actors(self, sharedReplay, sharedReplayLm, sharedStateDict):
        env = Worker(self.config, self.args)
        quill = Quill(self.config.MODELDIR)
        quill_eval = Quill(self.config.MODELDIR)

        tick = 0
        while True:
            if not ray.get(sharedStateDict.getFlag.remote()):
                sleep(5)
                continue

            idx, buffer, bufferLm, logs = env.run()

            if buffer is not None:
                sharedReplay.update.remote(buffer)
            if bufferLm is not None:
                sharedReplayLm.update.remote(bufferLm, idx)

            if logs is not None:
                quill.scrawl([logs])

            sharedStateDict.increaseCounter.remote(len(buffer[0][0]))

            tick += 1

            if tick % self.config.EVAL_FREQ == 0:
                all_logs = []
                for i in range(self.config.EVAL_EPISODES):
                    env.append_eval(idx, None)
                    _, _, _, logs = env.run_eval()
                    all_logs.append(logs)
                quill_eval.scrawl(all_logs)

            env.append(idx, ray.get(sharedStateDict.send.remote()))

    @ray.remote
    def run_learner(self, sharedReplay, sharedStateDict):
        learner = self.agent.Learner(self.config, self.args)
        sharedStateDict.updateAnns.remote(learner.net.sendAnns())
        sharedStateDict.updateTargetAnns.remote(learner.net.sendTargetAnns())

        while ray.get(sharedReplay.len.remote()) < self.config.MIN_BUFFER:
            sleep(5)

        while True:
            learner.net.loadLmFrom(ray.get(sharedStateDict.sendLm.remote()))

            idx, sample, weights = ray.get(sharedReplay.sample.remote())
            states, priorities = learner.step(sample, weights)
            sharedReplay.update_priorities.remote(idx, priorities)

            sharedStateDict.updateAnns.remote(states)

            sharedStateDict.decreaseCounter.remote(self.config.BATCH_SIZE)

    @ray.remote
    def run_learner_social(self, sharedReplayLm, sharedStateDict):
        learner = self.agent.LearnerSocial(self.config, self.args)
        sharedStateDict.updateLm.remote(learner.net.sendLm())

        while ray.get(sharedReplayLm.len.remote()) < self.config.MIN_BUFFER / self.args.nEnvs:
            sleep(5)

        while True:
            learner.net.loadAnnsFrom(ray.get(sharedStateDict.sendAnns.remote()))

            (idx, sample, weights), i = ray.get(sharedReplayLm.sample.remote())
            stateLm, priorities = learner.step(sample, weights)
            sharedReplayLm.update_priorities.remote(idx, priorities, i)

            sharedStateDict.updateLm.remote(stateLm)

    @ray.remote
    def run_learner_social_online(self, sharedReplayLm, sharedStateDict):
        learner = self.agent.LearnerSocial(self.config, self.args)
        sharedStateDict.updateLm.remote(learner.net.sendLm())
        N_EPOCHS = 2

        while True:
            (idx, sample, weights), i = ray.get(sharedReplayLm.sample.remote())
            if i is None:
                sleep(5)
                continue

            learner.net.loadAnnsFrom(ray.get(sharedStateDict.sendAnns.remote()))

            stateLm, priorities = learner.step(sample, None, N_EPOCHS)

            sharedStateDict.updateLm.remote(stateLm)

    def step(self):
        self.env.step()
