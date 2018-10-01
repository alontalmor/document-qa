
import pandas as pd
import numpy as np
pd.set_option('display.max_colwidth', 140)
pd.set_option('display.width', 2000)
import sys,os
import hashlib
m = hashlib.md5()
from subprocess import PIPE, Popen, call
import argparse
from docqa.config import TRIVIA_QA, TRIVIA_QA_UNFILTERED, CORPUS_DIR
from os.path import relpath, join, exists

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Evaluate a model on TriviaQA data')
parser.add_argument('datasets')
parser.add_argument('GPU')
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


target_dir = join(CORPUS_DIR, "triviaqa", "web-open", model_name)

print('creating mixed training')
command = 'python multiqa/create_mixed_training.py ' + args.datasets + ' --sample_first ' + str(args.sample_first) + \
+ ' --limit_train_size ' + str(args.limit_train_size)
print(command)
call(command , shell=True, preexec_fn=os.setsid)

# running build_span_corpus.py

# running the docqa training

source_dir = join(CORPUS_DIR, "triviaqa", "web-open", model_name)
print('running ablate_triviaqa_unfiltered')
command = 'export CUDA_VISIBLE_DEVICES=' + args.GPU + '; python docqa/scripts/ablate_triviaqa_unfiltered.py shared-norm ' + model_name + \
               ' --source_dir ' + source_dir
print(command)
call(command, shell=True, preexec_fn=os.setsid)

# running the docqa evaluation
source_dir = target_dir
print('running triviaqa_full_document_eval')
command = 'export CUDA_VISIBLE_DEVICES=' + args.GPU + '; python multiqa/eval_all_devsets.py models/' + model_name + ' CompWebQ-G,MSMARCO-G,MSMARCO-O,Squad-G,Squad-O,TriviaQA-G'
print(command)
call(command, shell=True, preexec_fn=os.setsid)