
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
parser.add_argument('model')
parser.add_argument('datasets')

args = parser.parse_args()


# running the docqa evaluation
print('running triviaqa_full_document_eval')
command = 'python docqa/eval/triviaqa_full_document_eval.py --n_processes 8 -c open-dev --tokens 800 -o question-output.json -p paragraph-output.csv triviaqa_unfiltered_full-0719-211745 --source_dir /media/disk1/alont/document-qa/data/triviaqa/web-open/ ' + args.exp_name + \
               ' --source_dir ' + source_dir
print(command)
call(command, shell=True, preexec_fn=os.setsid)