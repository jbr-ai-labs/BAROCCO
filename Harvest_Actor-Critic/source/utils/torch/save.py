import time

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
        self.start, self.epoch = time.time(), 0
        self.resetTol = resetTol

    def save(self, params, opt, fname):
        data = {'param': [ann.state_dict() for ann in params],
                'opt': [o.state_dict() for o in opt],
                'epoch': self.epoch}
        torch.save(data, self.root + fname + self.extn)

    def checkpoint(self, reward, params, opt):
        if self.epoch % 10 == 0:
            self.save(params, opt, self.savef)
        best = reward > self.best
        if best:
            self.best = reward
            self.save(params, opt, self.bestf)

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

    def print(self):
        print('Tick: ', self.epoch,
              ', Time: ', str(self.time)[:5],
              ', Lifetime: ', str(self.reward)[:5],
              ', Best: ', str(self.best)[:5])
