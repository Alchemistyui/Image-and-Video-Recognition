import os
import argparse

class BaseOptions(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        self.opt = None

    def initialize(self):
        self.initialized = True
        self.parser.add_argument("--debug", action="store_true", help="debug mode, break all loops")
        self.parser.add_argument("--results_dir_base", type=str, default="results/results")
        self.parser.add_argument("--log_freq", type=int, default=800, help="print, save training info")
        self.parser.add_argument("--seed", type=int, default=2018, help="random seed")
    def display_save(self):
        args = vars(self.opt)
        # Display settings
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

    def parse(self):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()

        return opt







