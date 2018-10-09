import pandas as pd
import numpy as np
pd.set_option('display.max_colwidth', 140)
pd.set_option('display.width', 2000)
import json
import os
import math
import hashlib
m = hashlib.md5()
import pickle
import argparse
from docqa.config import TRIVIA_QA, TRIVIA_QA_UNFILTERED, CORPUS_DIR
from os.path import relpath, join, exists
from docqa.triviaqa.build_span_corpus import TriviaQaWebDataset, TriviaQaOpenDataset, TriviaQaWikiDataset

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Evaluate a model on TriviaQA data')
parser.add_argument('datasets')
parser.add_argument("--sample_first", type=float, default=1.0,
                        help="Percentage to sample first dataset")
parser.add_argument("--limit_train_size", type=int, default=0,
                        help="Sample only this amount from training")
args = parser.parse_args()

if args.sample_first<1.0:
    datasets = args.datasets.split(',')
    datasets[0] += '_' + str(args.sample_first).replace('.','')
    model_name = '__'.join(datasets)
else:
    model_name = args.datasets.replace(',','__')

if args.limit_train_size!=0:
    model_name += '___' + str(args.limit_train_size)


all_train_questions = []
all_dev_questions = []
all_filemaps= {}
for ind,dataset in enumerate(args.datasets.split(',')):
    print('loading ' + dataset)
    source_dir = join(CORPUS_DIR, "triviaqa", "web-open", dataset)

    dataset = TriviaQaOpenDataset(source_dir)
    # just loading the pkl that was saved in build_span_corpus
    if args.sample_first==1.0 or ind == 0:
        all_dev_questions += dataset.get_dev()

    # sample_first assumes the first dataset in the list is our target dataset, to ablate we may whish
    # to take only a sample of it for training. sample_first is between (0,1]
    if args.sample_first<1.0 and ind == 0:
        all_train_questions += list(pd.Series(dataset.get_train()).sample(frac=args.sample_first))
    elif args.sample_first>1.0 and ind == 0:
        train = dataset.get_train()
        all_train_questions += train * math.floor(args.sample_first) + list(pd.Series(train).sample(frac=args.sample_first % 1))
    else:
        all_train_questions += dataset.get_train()

    with open(join(source_dir, "file_map.json"),'r') as f:
        all_filemaps.update(json.load(f))

if len(all_dev_questions) >= 8000:
    all_dev_questions = list(pd.Series(all_dev_questions).sample(n=8000))

# randomizing
all_train_questions = list(pd.Series(all_train_questions).sample(frac=1))

if args.limit_train_size!=0:
    all_train_questions = list(pd.Series(all_train_questions).sample(n=args.limit_train_size))


# Saving new training run:
print('saving new files')
target_dir = join(CORPUS_DIR, "triviaqa", "web-open", model_name)
if not os.path.isdir(target_dir):
    os.mkdir(target_dir)

with open(join(target_dir, "train.pkl"), "wb") as f:
    pickle.dump(all_train_questions, f)

with open(join(target_dir, "dev.pkl"), "wb") as f:
    pickle.dump(all_dev_questions, f)

with open(join(target_dir, "file_map.json"),'w') as f:
    json.dump(all_filemaps,f)



