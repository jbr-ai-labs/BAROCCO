import ray
import numpy as np

from source.agent import Controller
from source.env.core.env import HarvestEnv


@ray.remote
class Realm:
    def __init__(self, config, idx):
        self.env = HarvestEnv(num_agents=config.NPOP, norm=config.NORM_INPUT)
        self.horizon = config.HORIZON
        self.config = config
        self.controller = Controller(config)
        self.idx = idx
        self.step = 0

    def recvUpdate(self, update):
        if update is None:
            return
        self.controller.recvUpdate(update)

    def run(self, update):
        self.recvUpdate(update)
        for epoch in range(self.config.EPOCHS):
            obs = self.env.reset()
            rewards = self.env.getInitialRewards()
            rewards_ineq = rewards
            rewards_ineq_arr = np.array(list(rewards_ineq.values()))
            apples = rewards
            for i in range(self.horizon):
                global_state = self.env.map_to_colors(norm=self.config.NORM_INPUT)
                self.step += 1
                agents = self.env.agents
                step = i + epoch * self.horizon
                actions = {key: self.controller.decide(agents[key], obs[key], rewards[key],
                                                       (i + 1) % self.config.LSTM_PERIOD == 0,
                                                       step, epoch, global_state, apples[key], rewards_ineq_arr)
                           for key in agents.keys()}
                obs, rewards, dones, info, = self.env.step(actions)
                apples = info['apples']

                if self.config.INEQ:
                    rewards_ineq = {key: self.config.GAMMA * self.config.INEQ_LAMBDA * rs + rewards[key] for
                                    key, rs in rewards_ineq.items()}
                    rewards_ineq_arr = np.array(list(rewards_ineq.values()))
                    for key in rewards.keys():
                        rs = rewards_ineq[key]
                        u = rewards[key]

                        disadv = rewards_ineq_arr - rs
                        disadv = (disadv * (disadv > 0)).sum()
                        u -= self.config.INEQ_ALPHA / (len(rewards) - 1) * disadv

                        adv = rs - rewards_ineq_arr
                        adv = (adv * (adv > 0)).sum()
                        u -= self.config.INEQ_BETA / (len(rewards) - 1) * adv

                        annID = agents[key].annID
                        self.controller.buffer[annID]['reward'][step] = u
                else:
                    for key in obs.keys():
                        annID = agents[key].annID
                        self.controller.buffer[annID]['reward'][step] = rewards[key]

                for agent in agents.keys():
                    annID = agents[agent].annID
                    dct = actions.copy()
                    dct.pop(agent)
                    actions_other = [int(dct[other]) for other in sorted(dct.keys())]
                    actions_other = one_hot(actions_other, 8)
                    self.controller.buffer[annID]['action_other'][step] = actions_other

            for agent in self.env.agents.values():
                self.controller.collectRollout(agent.agent_id + str(epoch), agent, self.step, epoch)
        self.controller.backward()
        logs = self.controller.sendLogUpdate()
        buf = self.controller.dispatchBuffer()
        return self.idx, buf, logs


def one_hot(lst, n):
    ary = np.zeros(n * len(lst))
    for i, v in enumerate(lst):
        ary[i * n + v] = 1
    return ary


