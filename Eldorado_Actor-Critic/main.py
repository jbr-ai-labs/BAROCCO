import argparse
# import wandb
from collections import namedtuple
import torch
import experiments
from source.agent import runner, Controller, Learner


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nEnvs', type=int, default='1',
                        help='Number of environments (1 per core)')
    return parser.parse_args()


class NativeExample:
    def __init__(self, config, args):
        Agent = namedtuple('Agent', ['Controller', 'Learner'])
        self.env = runner.Runner(config, args, Agent(Controller, Learner))

    def run(self):
        self.env.run()


if __name__ == '__main__':
    args = parseArgs()
    print('cuda', torch.cuda.is_available())
    print(torch.version.cuda)
    config = experiments.exps['barocco']
    # Uncomment this if you want to use wandb
    # wandb.init(project='barocco', config=config, save_code=True)

    example = NativeExample(config, args)
    example.run()
