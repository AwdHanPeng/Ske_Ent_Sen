import pickle
import torch
import argparse
from utils import *
from dataset import ValidDataset
from model import Model, Config


parser = argparse.ArgumentParser(description='Valid')
parser.add_argument('--valid-batch-size', type=int, default=1000,
                    help='input batch size for validation (default: 1000)')
parser.add_argument('--no-cuda', default=0, type=int,
                    help='disables CUDA training')
parser.add_argument('--valid', default=0, type=int,
                    help='whether to valid model right now')
parser.add_argument('--img', default=1, type=int,
                    help='whether to show history image right now')

args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
valid_root = ['utterance', 'responses', 'index']
valid_root_path = ['../Data/valid/' + i + '.pkl' for i in valid_root]
valid_loader = torch.utils.data.DataLoader(
    ValidDataset(root=valid_root_path, transforms=[padding_sentence, padding_utterance]),
    batch_size=args.valid_batch_size, shuffle=False, **kwargs)

if __name__ == '__main__':
    if args.valid:
        config = Config()
        model = Model(config)
        model.load_state_dict(
            torch.load('../model/{}.pt'.format(best_primitive_model)))
        model = model.to(device)
        valid(args, model, device, valid_loader)
    if args.img:
        with open('../model/history_{}_02.pkl'.format('Primitive'), 'rb') as f:
            history_list = pickle.load(f)
        imshow(history_list, title='Experiment 3'.format('Primitive'))
'''
{'MAP': 0.5882062698412642, 'R10_2': 0.5743, 'epoch': 9, 'R10_5': 0.8317, 'R10_1': 0.4145, 'P_1': 0.4145, 'MRR': 0.5882062698412642}
'''