
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
        parser.add_argument('--name_log', default='',type=str,
                            help='')
        parser.add_argument('--batch_size', default=64,type=int,
                            help='')
        parser.add_argument('--seed', default=1,type=int,
                            help='')
#         parser.add_argument('--epochs', default=5,type=int,
#                             help='')
        parser.add_argument('--epochs', default=1,type=int,
                    help='')
        parser.add_argument('--n_proj', default=1,type=int,
                            help='')
        parser.add_argument('--lambda_JR', default=0.1,type=float,
                            help='')
        
        



        self.parser = parser

    def parse(self):
        return self.parser.parse_args()
