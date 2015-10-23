import numpy as np

from rpy2 import robjects
from rpy2.robjects import packages


def library(pkg):
    packages.quiet_require(pkg)


def read_rds(filename):
    return Command('readRDS("{}")'.format(filename))


class Command:
    def __init__(self, cmd):
        self.cmd = cmd

    def __repr__(self):
        return self.cmd

    def execute(self):
        return np.array(robjects.r(self.cmd))

    def call_function(self, func_name):
        new_cmd = '{}({})'.format(func_name, self.cmd)
        return Command(new_cmd)

    def index_list(self, *indices):
        command = self
        
        for idx in indices:
            if isinstance(idx, int):
                new_cmd = '{}[[{}]]'.format(command.cmd, idx)
            elif isinstance(idx, str):
                new_cmd = '{}[["{}"]]'.format(command.cmd, idx)
            else:
                raise RuntimeError('Cannot use {} as an index.'.format(type(idx)))

            command = Command(new_cmd)

        return command

    def index_vector(self, idx):
        new_cmd = '{}[{}]'.format(self.cmd, idx)
        return Command(new_cmd)

    def index_matrix(self, idx1, idx2):
        new_cmd = '{}[{}, {}]'.format(self.cmd, idx1, idx2)
        return Command(new_cmd)
