import ray

from copy import deepcopy

from source.agent import Controller
from source.env.core.env import HarvestEnv


@ray.remote
class Realm:
    def __init__(self, config, args, idx):
        self.env = HarvestEnv(num_agents=config.NPOP, norm=config.NORM_INPUT)
        self.horizon = config.HORIZON
        self.config, self.args = config, args
        self.controller = Controller(config, args, idx)
        self.idx = idx
        self.epoch = 0

    def recvSwordUpdate(self, update):
        if update is None:
            return
        self.controller.recvUpdate(update)

    def run(self, update, evaluate=False):
        """ Rollout several timesteps of an episode of the environment.
               Args:
                   horizon: The number of timesteps to roll out.
                   save_path: If provided, will save each frame to disk at this
                       location.
               """
        self.recvSwordUpdate(update)
        [ann.eval() if evaluate else ann.train() for ann in self.controller.anns]
        [social.eval() if evaluate else social.train() for social in self.controller.social]

        obs = self.env.reset()
        rewards = self.env.getInitialRewards()
        apples, rewards_stats = rewards, rewards
        for i in range(self.horizon):
            global_state = self.env.map_to_colors(norm=self.config.NORM_INPUT)
            agents = self.env.agents
            actions = {key: self.controller.decide(self.agentName(agents[key].entID), agents[key].annID, obs[key],
                                                   rewards[key], rewards_stats[key], apples[key], i == (self.horizon - 1),
                                                   evaluate=evaluate)
                       for key in agents.keys()}
            obs, rewards, dones, info, = self.env.step(actions)
            apples = info['apples']
            rewards_stats = deepcopy(rewards)

            if not evaluate:
                self.controller.ReplayMemoryLm.push(self.controller.prepareInput(global_state))
                self.controller.config.EPS_CUR = max(self.controller.config.EPS_MIN,
                                                     self.controller.config.EPS_CUR * self.controller.config.EPS_STEP)
                if self.config.NOISE:
                    [self.controller.reset_noise() for _ in range(self.idx + 1)]
                self.controller.tick += 1

            if self.config.COMMON:
                commonReward = eval(self.config.COMMON_FUN)(list(rewards.values()))
                for key in rewards.keys():
                    rewards[key] = commonReward

        for ent in self.env.agents.values():
            self.controller.collectRollout(self.agentName(ent.entID))

        self.epoch += 1

        updates, updates_lm, logs = self.controller.sendUpdate(evaluate=evaluate)
        return self.idx, updates, updates_lm, logs

    def agentName(self, name):
        return name + str(self.idx) + str(self.epoch)
