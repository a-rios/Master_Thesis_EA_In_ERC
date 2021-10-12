# -*- coding: utf-8 -*-
import os
import argparse

import numpy as np
import torch
import pytorch_lightning as pl
import CustomDataloader
from torch.utils.data import DataLoader

import Utils
import EmoTrain
import Preprocess
import shutil

def get_args():
    """
        returns the Parser args
    """
    root_dir = os.getcwd()
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='0',		# Specify the GPU for training
	                    help='gpu: default 0')
    parser.add_argument('--emoset', type=str, default = 'emorynlp',
                        help = 'Emotion Training Set Name')
    parser.add_argument('--cache_dir', type=str, required=False,
                        help='Cache directory for transformer models.')
    parser.add_argument('--data_dir', type=str, required=False, default=os.path.join(os.path.dirname(__file__), "data"),
                        help='Data directory.')
    parser.add_argument('--checkpoint', type=str, metavar='PATH' ,required=True,
                        help='Path to checkpoint of trained model.')
    parser.add_argument('--json_out', type=str, metavar='PATH' ,
                        help='Path to json output file.')
    parser.add_argument('--csv_out', type=str, metavar='PATH',
                        help='Path to csv output file.')
    parser.add_argument('--print_logits',  action="store_true",
                        help = 'Print raw logits to json instead of probabilities.')
    args = parser.parse_args()
    return args

def main():
    # Remove existing logs
    if (os.path.isdir('lightning_logs')):
        shutil.rmtree('lightning_logs')
    args = get_args()
    print(args, '\n')
    args.emoset = args.emoset.lower()
    print(args.checkpoint)
    #checkpoint = torch.load(args.checkpoint)
    #print(checkpoint['hparams_name'])
    #print(checkpoint['hyper_parameters'])

    assert args.emoset  in ['emorynlp', 'emotionpush', 'friends','semeval', 'friends_german',
                        'emotionpush_german', 'emorynlp_german', 'semeval_german', 'meld_friends_german_aligned', 'meld_friends_english', 'meld_friends_german_deepl', 'vam' ]

    if not args.emoset == 'semeval':
        args.batch_size = 1

    device = torch.device("cuda:{}".format(int(args.gpu)) if torch.cuda.is_available() else "cpu")
    args.device = device
    print('Args.device = {}'.format(args.device))

    print('Creating eval dfs')
    df_test  = Utils.load_df(args, test_only=True)
    print('Finished Creating eval dfs')

    if args.emoset == 'meld_friends_german_aligned' or args.emoset == 'meld_friends_english' or args.emoset == 'meld_friends_german_deepl' or args.emoset == 'vam' : # no non-neutral in MELD version of this dataset (split seems to be the same as friends, check why the labels are different
        emo_dict = {'neutral': 0, 'sadness': 1, 'anger':2, 'joy':3, 'surprise':4, 'fear':5, 'disgust':6}
        focus_dict = ['neutral', 'sadness', 'anger', 'joy', 'surprise', 'fear', 'disgust' ]
        sentiment_dict = {'neutral': 0, 'sadness': 1, 'anger':1, 'joy':2, 'surprise':2,'fear':1, 'disgust':1}
    else:
        print("Cannot test on {}, not implemented".format(args.emoset))
        exit(1)

    model = EmoTrain.EmotionModel.load_from_checkpoint(args.checkpoint)
    for name in vars(args):
        model.args.__dict__[name] = getattr(args, name)
    print("new model args ", model.args)
    model.df_test = df_test

    gpu_list = [int(model.args.gpu)]
    trainer = pl.Trainer(gpus=(gpu_list if torch.cuda.is_available() else 0))
    trainer.test(model, verbose = model.verbosity)

if __name__ == '__main__':
	main()
