	# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 13:52:02 2021

@author: neera
"""
import json
import pickle
import torch
import os
import math
import random
import numpy as np

import time
import pandas as pd


def create_df_emorynlp(data, column_list):
    df = pd.DataFrame(columns = column_list)
    dialogue_idx = 0
    episode_idx = 0
    for episode in data['episodes']:
        scenes = episode['scenes']
        for i in range(len(scenes)):
            for j in range(len(scenes[i]['utterances'])):
                df2 = pd.DataFrame([[dialogue_idx, scenes[i]['utterances'][j]['transcript'] , scenes[i]['utterances'][j]['speakers'][0], len(scenes[i]['utterances'][j]['transcript'].split()), scenes[i]['utterances'][j]['emotion'].lower() ]], columns = column_list)
                df = df.append(df2, ignore_index=True)
            dialogue_idx += 1
        episode_idx = episode_idx+1



    return df


def create_df_friends(data, column_list):
    df = pd.DataFrame(columns = column_list)
    dialogue_idx = 0
    for dialog in data:
        for utter in dialog:
            df2 = pd.DataFrame([[dialogue_idx, utter['utterance'], utter['speaker'], len(utter['utterance'].split()), utter['emotion'].lower() ]], columns = column_list)
            df = df.append(df2, ignore_index=True)
        dialogue_idx += 1



    return df

def create_df(data, emoset):
    column_list = list(('dialogue_id', 'utterance', 'speaker','utterance_len', 'label'))
    if emoset == 'emorynlp':
        df=  create_df_emorynlp(data, column_list)
    elif emoset == 'friends' or emoset == 'emotionpush':
        df = create_df_friends(data, column_list)


    return df

def load_df(args, test_only=False):

    if args.emoset == 'emorynlp':
        print('Creating Training/Val/Test Dataframes for EmoryNLP Dataset')
        if not test_only:
            train_path = os.path.join(args.data_dir, 'EmoryNLP/English/emotion-detection-trn.json')
            val_path = os.path.join(args.data_dir, 'EmoryNLP/English/emotion-detection-dev.json')
        test_path = os.path.join(args.data_dir, 'EmoryNLP/English/emotion-detection-tst.json')
    elif args.emoset == 'emorynlp_german':
        print('Creating Training/Val/Test Dataframes for EmoryNLP German Dataset')
        if not test_only:
            train_path = os.path.join(args.data_dir, 'EmoryNLP/German/emorynlp_train_de.csv')
            val_path = os.path.join(args.data_dir, 'EmoryNLP/German/emorynlp_val_de.csv')
        test_path = os.path.join(args.data_dir, 'EmoryNLP/German/emorynlp_test_de.csv')

    elif  args.emoset == 'friends':
        print('Creating Training/Val/Test Dataframes for Friends Dataset')
        if not test_only:
            train_path = os.path.join(args.data_dir, 'Friends/Friends_English/friends.train.json')
            val_path = os.path.join(args.data_dir, 'Friends/Friends_English/friends.dev.json')
        test_path = os.path.join(args.data_dir, 'Friends/Friends_English/friends.test.json')
    elif args.emoset == 'friends_german':
        print('Creating Training/Val/Test Dataframes for Friends German Dataset')
        if not test_only:
            train_path = os.path.join(args.data_dir, 'Friends/Friends_German/friends_train_de.csv')
            val_path = os.path.join(args.data_dir, 'Friends/Friends_German/friends_val_de.csv')
        test_path = os.path.join(args.data_dir, 'Friends/Friends_German/friends_test_de.csv')

    elif  args.emoset == 'emotionpush':
        print('Creating Training/Val/Test Dataframes for EmotionPush Dataset')
        if not test_only:
            train_path = os.path.join(args.data_dir, 'Emotionpush/English/emotionpush.train.json')
            val_path =   os.path.join(args.data_dir, 'Emotionpush/English/emotionpush.dev.json')
        test_path =  os.path.join(args.data_dir, 'Emotionpush/English/emotionpush.test.json')

    elif args.emoset == 'emotionpush_german':
        print('Creating Training/Val/Test Dataframes for Emotionpush German Dataset')
        if not test_only:
            train_path = os.path.join(args.data_dir, 'Emotionpush/German/emotionpush_train_de.csv')
            val_path =   os.path.join(args.data_dir, 'Emotionpush/German/emotionpush_val_de.csv')
        test_path =  os.path.join(args.data_dir, 'Emotionpush/German/emotionpush_test_de.csv')

    elif args.emoset == 'meld_friends_german_aligned':
        print('Creating Training/Val/Test Dataframes for MELD Friends aligned German Dataset')
        if not test_only:
            train_path = os.path.join(args.data_dir, 'train_sent_emo_de_fixed_idx.csv')
            val_path =   os.path.join(args.data_dir, 'dev_sent_emo_de.csv')
        test_path =  os.path.join(args.data_dir, 'test_sent_emo_de_gold_fixed_idx.csv')

    elif args.emoset == 'meld_friends_german_deepl':
        print('Creating Training/Val/Test Dataframes for MELD Friends aligned German Dataset')
        if not test_only:
            train_path = os.path.join(args.data_dir, 'train_sent_emo_de_deepl.csv')
            val_path =   os.path.join(args.data_dir, 'dev_sent_emo_de_deepl.csv')
        test_path =  os.path.join(args.data_dir, 'test_sent_emo_de_gold_fixed_idx_deeplname.csv')

    elif args.emoset == 'meld_friends_english':
        print('Creating Training/Val/Test Dataframes for MELD Friends English Dataset')
        if not test_only:
            train_path = os.path.join(args.data_dir, 'train_sent_emo_fixed_idx.csv')
            val_path =   os.path.join(args.data_dir, 'dev_sent_emo.csv')
        test_path =  os.path.join(args.data_dir, 'test_sent_emo_gold_fixed_idx.csv')
    elif args.emoset == 'vam':
        print('Creating Test Dataframes for VAM')
        test_path =  os.path.join(args.data_dir, 'vam.csv')

    else:
        print('Creating Training/Val/Test Dataframes for Semeval Dataset')
        if not test_only:
            train_path = os.path.join(args.data_dir, 'clean_train.txt')
            val_path = os.path.join(args.data_dir, 'clean_val.txt')
        test_path = os.path.join(args.data_dir, 'clean_test.txt')

    if args.emoset in ['friends', 'emotionpush', 'emorynlp'] :
        with open(train_path, encoding='utf-8') as data_file:
            train_json = json.loads(data_file.read())

        with open(val_path, encoding='utf-8') as data_file:
            val_json = json.loads(data_file.read())


        with open(test_path, encoding='utf-8') as data_file:
            test_json = json.loads(data_file.read())

        if not test_only:
            df_train = create_df(train_json, args.emoset)
            df_val = create_df(val_json,  args.emoset)
        df_test = create_df(test_json,  args.emoset)
    elif args.emoset == 'semeval':
        if not test_only:
            df_train = pd.read_csv(train_path, delimiter='\t', index_col='id')
            df_val =  pd.read_csv(val_path, delimiter='\t', index_col='id')
        df_test = pd.read_csv(test_path, delimiter='\t', index_col='id')
        '''
        df_train = df_train.iloc[0:500, :]
        df_val = df_val.iloc[0:160, :]
        df_test = df_test.iloc[0:160, :]
        '''
        #file,narrow_transcription,transcription,valence_ewe,activation_ewe,dominance_ewe,valence_std,activation_std,dominance_std,speaker,gender,age
    elif args.emoset == 'vam':
        #utterance_string = "transcription"
        df_test = pd.read_csv(test_path)
        df_test = df_test.dropna()
        df_test = df_test.assign(dialogue_id=df_test['file'].str.extract(r'Satz(..)').astype(int)-1) # dialogue ids start at 1, set to 0
        df_test = df_test.assign(utterance_id=df_test['file'].str.extract(r'(...)\.wav').astype(int))
        df_test['utterance_len'] = df_test[['transcription']].applymap(lambda x: len(x.split()))
        return df_test
    else:
        col_dict = {'label': 'label'}  if args.emoset == 'meld_friends_german_aligned' else {'emotion': 'label'}   ## key→old name, value→new name

        utterance_string =  "utterance_de_deepl"
        if args.emoset == 'meld_friends_german_aligned':
            utterance_string = 'utterance_de'
        elif  args.emoset == 'meld_friends_english':
            utterance_string = 'Utterance'

        if not test_only:
            df_train = pd.read_csv(train_path)
            df_val =  pd.read_csv(val_path)
            df_train = df_train.dropna()
            df_val = df_val.dropna()
            df_train.columns = [col_dict.get(x, x) if x in col_dict.keys() else x  for x in df_train.columns]
            df_val.columns = [col_dict.get(x, x) if x in col_dict.keys() else x  for x in df_val.columns]
            df_train['utterance_len'] = df_train[[utterance_string]].applymap(lambda x: len(x.split()))
            df_val['utterance_len'] = df_val[[utterance_string]].applymap(lambda x: len(x.split()))
        df_test = pd.read_csv(test_path)
        df_test = df_test.dropna()
        df_test.columns = [col_dict.get(x, x) if x in col_dict.keys() else x  for x in df_test.columns]
        df_test['utterance_len'] = df_test[[utterance_string]].applymap(lambda x: len(x.split()))
        if test_only:
            return df_test
        else:
            return (df_train, df_val, df_test)

def shuffle_dataframe(df):
    dialogue_id_list = list((df['dialogue_id'].unique()))
    df_shuffled =  pd.DataFrame(columns = list(df.columns))
    for dialogue in dialogue_id_list:
        df_dialogue = df[(df.dialogue_id ==dialogue)].copy(deep = True)
        df_dialogue = df_dialogue.sample(frac=1)
        df_shuffled = df_shuffled.append(df_dialogue)
    df_shuffled =df_shuffled.reset_index(drop=True)
    return df_shuffled
