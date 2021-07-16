import os

from configs import Experiment

exps = {}


def makeExp(name, conf):
    ROOT = 'resource/exps/' + name + '/'
    try:
        os.makedirs(ROOT)
        os.makedirs(ROOT + 'model')
        os.makedirs(ROOT + 'train')
        os.makedirs(ROOT + 'test')
    except FileExistsError:
        pass
    MODELDIR = ROOT + 'model/'

    exp = conf(MODELDIR=MODELDIR)
    exps[name] = exp
    print(name, ', NENT: ', exp.NENT, ', NPOP: ', exp.NPOP)


makeExp('barocco', Experiment)
