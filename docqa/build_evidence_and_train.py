
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
from docqa.config import TRIVIA_QA, TRIVIA_QA_UNFILTERED
from os.path import relpath, join, exists

parser = argparse.ArgumentParser(description='Evaluate a model on TriviaQA data')
parser.add_argument('-m', '--model',default="../models-cpu/triviaqa-web-shared-norm")
parser.add_argument('-e', '--evidence',default=join(TRIVIA_QA, "evidence"))
args = parser.parse_args()

# running evidence_corpus (This builds evidence)
wa_proc = call('python triviaqa/evidence_corpus.py --n_processes 8 --source ' + args.evidence \
                , shell=True, preexec_fn=os.setsid)

# running build_span_corpus.py
wa_proc = call('python triviaqa/build_span_corpus.py web-open --n_processes 8', shell=True,
                preexec_fn=os.setsid)

# running the docqa evaluation
wa_proc = call('python docqa/scripts/ablate_triviaqa_unfiltered.py shared-norm ' + args.model, \
               shell=True, preexec_fn=os.setsid)