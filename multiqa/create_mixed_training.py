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
parser.add_argument("--sample_rest", type=float, default=1.0,
                        help="Sample only this amount from training")
args = parser.parse_args()


model_name = args.datasets.replace(',','__')
if args.sample_first != 1.0:
    model_name += '___SF' + str(args.sample_first)

if args.sample_rest !=  1.0:
    model_name += '___SR' + str(args.sample_rest)


all_train_questions = []
all_dev_questions = []
all_filemaps= {}
for ind,dataset_name in enumerate(args.datasets.split(',')):
    print('loading ' + dataset_name)
    source_dir = join(CORPUS_DIR, "triviaqa", "web-open", dataset_name)

    dataset = TriviaQaOpenDataset(source_dir)
    # just loading the pkl that was saved in build_span_corpus
    if args.sample_first==1.0 or ind == 0:
        all_dev_questions += dataset.get_dev()

    num_of_contexts = (pd.Series(args.datasets.replace('-G','').replace('-O','').split(',')) == \
                       dataset_name.replace('-G','').replace('-O','')).sum()

    train = dataset.get_train()

    # Filtering cases with no answer:
    train_with_ans = []
    for question in train:
        if pd.Series([len(doc.answer_spans) for doc in question.all_docs]).sum()>0:
            train_with_ans.append(question)

    print("number of question with answer is %d" % (len(train_with_ans)))

    # sample_first assumes the first dataset in the list is our target dataset, to ablate we may whish
    # to take only a sample of it for training. sample_first is between (0,1]
    if args.sample_first<=1.0 and ind == 0:
        all_train_questions += list(pd.Series(train_with_ans).sample(frac=args.sample_first))
    elif args.sample_first>1.0 and ind == 0:
        # Greater than 100 means absolute number of samples, converting to frac
        if args.sample_first>100:
            args.sample_first = float(args.sample_first)/len(train_with_ans)
            print('sampling first dataset in frac = %f' % args.sample_first)

        oversampled_train = train_with_ans * math.floor(args.sample_first) + list(pd.Series(train_with_ans).sample(frac=args.sample_first % 1))
        print("dataset %s oversampled sampled train size %f" % (dataset_name,len(oversampled_train)))
        all_train_questions += oversampled_train
    else: # Rest of the datasets
        # Greater than 100 means absolute number of samples, converting to frac
        if args.sample_rest > 100:
            sample_rest_frac = float(args.sample_rest) / len(train_with_ans)
            sample_rest_frac /= num_of_contexts
            print('sampling rest dataset in frac = %f' % sample_rest_frac)
        else:
            sample_rest_frac /= num_of_contexts
            sample_rest_frac = args.sample_rest

        if args.sample_rest<=1.0 and ind > 0:
            all_train_questions += list(pd.Series(train_with_ans).sample(frac=sample_rest_frac))
        else:
            oversampled_train = train_with_ans * math.floor(sample_rest_frac) + list(pd.Series(train_with_ans).sample(frac=sample_rest_frac % 1))
            print("multiqa dataset %s oversampled sampled train size %f" % (dataset_name, len(oversampled_train)))
            all_train_questions += oversampled_train
            #all_train_questions += train_with_ans

    print("total train size %f" % (len(all_train_questions)))

    with open(join(source_dir, "file_map.json"),'r') as f:
        all_filemaps.update(json.load(f))

if len(all_dev_questions) >= 8000:
    all_dev_questions = list(pd.Series(all_dev_questions).sample(n=8000))

# randomizing
all_train_questions = list(pd.Series(all_train_questions).sample(frac=1))

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



