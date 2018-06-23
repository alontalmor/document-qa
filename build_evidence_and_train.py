
import pandas as pd
import numpy as np
pd.set_option('display.max_colwidth', 140)
pd.set_option('display.width', 2000)
import sys,os
import nltk
import io
import json
import time
from pandas import ExcelWriter
from ast import literal_eval
import hashlib
import unicodedata
import subprocess
import datetime
m = hashlib.md5()
import zipfile
import random
import requests
from subprocess import PIPE, Popen, call
import sys, traceback
from threading  import Thread
import time
import dropbox
import socket
import shutil
import argparse
import signal
from nltk.metrics import *
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
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
parser.add_argument('exp_name',default="test")
parser.add_argument('--build_evidence', type=str2bool, default=True, nargs='?', const=True , \
                    help="should we build the evidence as well?")
args = parser.parse_args()

# running evidence_corpus (This builds evidence)
# No need for source evidence dir, because each experiment name evidence is added as a different sub dir
# (evidence/web/exp_name/0 ... )
if args.build_evidence:
    print('running evidence_corpus')
    command = 'python docqa/triviaqa/evidence_corpus.py --n_processes 8 --source ' + \
                   join(TRIVIA_QA, "evidence",args.exp_name) + ' --output_dir ' + \
                   join(CORPUS_DIR, "triviaqa", "evidence","web", args.exp_name)
    print(command)
    call(command , shell=True, preexec_fn=os.setsid)

# running build_span_corpus.py
source_dir = join(TRIVIA_QA_UNFILTERED, args.exp_name)
target_dir = join(CORPUS_DIR, "triviaqa", "web-open", args.exp_name)
print('running build_span_corpus')
command = 'python docqa/triviaqa/build_span_corpus.py web-open --n_processes 8 --sets_to_build dev,train --source_dir ' + source_dir \
               + ' --target_dir ' + target_dir
print(command)
call(command, shell=True, preexec_fn=os.setsid)

# running the docqa evaluation
source_dir = target_dir
print('running ablate_triviaqa_unfiltered')
command = 'python docqa/scripts/ablate_triviaqa_unfiltered.py shared-norm ' + args.exp_name + \
               ' --source_dir ' + source_dir
print(command)
call(command, shell=True, preexec_fn=os.setsid)