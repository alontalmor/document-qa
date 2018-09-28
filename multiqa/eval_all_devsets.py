
import sys,os
from os.path import relpath, join, exists
from subprocess import PIPE, Popen, call
import argparse
from docqa.config import TRIVIA_QA, TRIVIA_QA_UNFILTERED, CORPUS_DIR

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

models_dir = join(CORPUS_DIR, "triviaqa", "web-open/")
# running the docqa evaluation
for dataset in args.datasets.split(','):
    if args.model == '*':
        for dirname, dirnames, filenames in os.walk(models_dir):
            for filename in filenames:
                print('running triviaqa_full_document_eval')
                print(models_dir + dataset)
                print(models_dir + filename)
                command = 'python docqa/eval/triviaqa_full_document_eval.py --n_processes 8 -c open-dev --tokens 800 -o question-output.json -p paragraph-output.csv ' + models_dir + filename + ' --source_dir ' + models_dir + dataset
                print(command)
                call(command, shell=True, preexec_fn=os.setsid)
    else:
        print('running triviaqa_full_document_eval')
        command = 'python docqa/eval/triviaqa_full_document_eval.py --n_processes 8 -c open-dev --tokens 800 -o question-output.json -p paragraph-output.csv ' + args.model + ' --source_dir ' + models_dir + dataset
        print(command)
        call(command, shell=True, preexec_fn=os.setsid)