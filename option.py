
import argparse
import os

class Options():
    def __init__(self):### 
        # Training settings
        parser = argparse.ArgumentParser(description='cs7643_proj')
        parser.add_argument('--dataset', default='mnist',type=str,
                            help='')
        parser.add_argument('--model', default='lenet_dropout',type=str,
                            help='')
        parser.add_argument('--batch_size', default=100,type=int,
                            help='')
        parser.add_argument('--seed', default=1,type=int,
                            help='')
        parser.add_argument('--epochs', default=1,type=int,
                    help='')
        parser.add_argument('--n_proj', default=1,type=int,
                            help='')
        parser.add_argument('--lambda_JR', default=0.1,type=float,
                            help='')
        parser.add_argument('--val', default=1,type=int,
                            help='')
        parser.add_argument('--wd', default=5e-4,type=float,
                            help='weight decay')
        parser.add_argument('--jacobian', default=1,type=int,
                            help='if jacobian regularization')
        parser.add_argument('--save_name', default='model',type=str,
                            help='model name & log name')
        parser.add_argument('--init_lr', default=0.1,type=float,
                            help='initial learning rate')
        parser.add_argument('--defense', default=None,type=str,
                            help='rand_size or None')

        
        



        self.parser = parser

    def parse(self):
        return self.parser.parse_args()
