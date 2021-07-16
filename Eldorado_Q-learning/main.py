import argparse
from collections import namedtuple

import experiments
from source.agent import runner, Learner, Controller, LearnerSocial


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nEnvs', type=int, default='1',
                        help='Number of environments (1 per core)')
    return parser.parse_args()


class NativeExample:
    def __init__(self, config, args):
        Agent = namedtuple('Agent', ['Controller', 'LearnerSocial', 'Learner'])
        self.env = runner.Runner(config, args, Agent(Controller, LearnerSocial, Learner))

    def run(self):
        self.env.run()


if __name__ == '__main__':
    args = parseArgs()
    config = experiments.exps['barocco']

    example = NativeExample(config, args)
    example.run()
