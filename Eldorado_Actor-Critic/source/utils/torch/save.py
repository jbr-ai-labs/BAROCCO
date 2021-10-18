import time
# import wandb

import torch


class Resetter:
    def __init__(self, resetTol):
        self.resetTicks, self.resetTol = 0, resetTol

    def step(self, best=False):
        if best:
            self.resetTicks = 0
        elif self.resetTicks < self.resetTol:
            self.resetTicks += 1
        else:
            self.resetTicks = 0
            return True
        return False


class Saver:
    def __init__(self, nANN, root, savef, bestf, lmSavef, resetTol):
        self.bestf, self.savef, self.lmSavef = bestf, savef, lmSavef,
        self.root, self.extn = root, '.pth'
        self.nANN = nANN

        self.resetter = Resetter(resetTol)
        self.best = 0
        self.best_lifetime = 0
        self.start, self.epoch = time.time(), 0
        self.resetTol = resetTol

    def save(self, params, opt, fname):
        data = {'param': [ann.state_dict() for ann in params],
                'opt': [o.state_dict() for o in opt],
                'epoch': self.epoch}
        torch.save(data, self.root + fname + self.extn)

    def checkpoint(self, logs, params, opt):
        reward, lifetime, _, _, _, _, _ = logs
        if self.epoch % 10 == 0:
            self.save(params, opt, self.savef)
        best = reward > self.best

        if best:
            self.best = reward
            self.save(params, opt, self.bestf)

        if lifetime > self.best_lifetime:
            self.best_lifetime = lifetime

        self.time = time.time() - self.start
        self.start = time.time()
        self.reward = reward
        self.epoch += 1

        if self.epoch % 500 == 0:
            self.save(params, opt, 'model' + str(self.epoch))

        return self.resetter.step(best)

    def checkpointSocial(self, params, opt):
        if self.epoch % 10 == 0:
            self.save(params, opt, 'social')
        if self.epoch % 500 == 0:
            self.save(params, opt, 'social' + str(self.epoch))

    def load(self, opt, anns, lmOpt, social, best=False):
        fname = self.bestf if best else self.savef
        data = torch.load(self.root + fname + self.extn)
        [ann.load_state_dict(st_dict) for ann, st_dict in zip(anns, data['params'])]
        if opt is not None:
            [o.load_state_dict(st_dict) for o, st_dict in zip(opt, data['opt'])]
        epoch = data['epoch']
        fname = self.lmSavef
        try:
            data = torch.load(self.root + fname + self.extn)
            [ann.load_state_dict(st_dict) for ann, st_dict in zip(social, data['params'])]
            if lmOpt is not None:
                [lmOpt.load_state_dict(st_dict) for o, st_dict in zip(opt, data['opt'])]
        except:
            print('social didn\'t load')

        return epoch

    def print(self, logs):
        reward, lifetime_agg, lifetime_0, lifetime_1, contact, attack, value = logs
        # Uncomment this if you want to use wandb
        # print(f'Game: {self.epoch:.0f}, '
        #       f'Time: {self.time:.3f}, '
        #       f'Lifetime Agg: {lifetime_agg:.3f} '
        #       f'Lifetime Agent_0: {lifetime_0:.3f} '
        #       f'Lifetime Agent_1: {lifetime_1:.3f} '
        #       f'Best: {self.best_lifetime:.3f}')
        #
        # wandb.log({'Game': self.epoch,
        #            'time': self.time,
        #            'lifetime_agg': lifetime_agg,
        #            'lifetime_0': lifetime_0,
        #            'lifetime_1': lifetime_1,
        #            'contact': contact,
        #            'attack': attack,
        #            'value': value})
