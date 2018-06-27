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
parser.add_argument('-m', '--model',
                        default="models-cpu/triviaqa-web-shared-norm")
parser.add_argument('--full_evidence', type=str2bool, default=False, nargs='?', const=True , \
                    help="should we add all evidence to SimpQA (300 results)?")
args = parser.parse_args()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def enqueue_output(out,err,q):
    for line in iter(out.readline, b''):
        q.put(line)
    for line in iter(err.readline, b''):
        q.put('error: ' + line)
    q.put('exit')
    out.close()

def append_goog_results_for_SimpQA(questions):
    #questions['google_results'] = questions['google_results'].astype(object)
    for ind,simp_qa_question in questions[questions['SimpQA'].notnull()].iterrows():
        search_results = []
        for q_part in questions.loc[questions['ID'] == simp_qa_question['ID'], 'google_results'].tolist():
            search_results += q_part

        questions.at[ind,'google_results'] = search_results

def build_evidence(questions,DELETE_PREV_EVIDENCE = True):

    questions_triviaqa_format = pd.DataFrame()
    questions_triviaqa_format['QuestionId'] = questions.index.astype(str)
    questions_triviaqa_format['Question'] = questions['question']
    if args.full_evidence:
        append_goog_results_for_SimpQA(questions)
    questions_triviaqa_format['SearchResults'] = questions['google_results']
    questions_triviaqa_format['EntityPages'] = [[] for x in range(len(questions_triviaqa_format))]
    questions_triviaqa_format['QuestionSource'] = ''
    all_answers = []
    for ind, q in questions.iterrows():
        if 'answers' not in q or q['answers'] != q['answers'] or q['answers'][0]['answer'] == None:
            all_answers.append(None)
        else:
            filtered_answer = []
            for answer in q['answers']:
                if len(answer['answer']) > 0:
                    filtered_answer.append(answer)

            triviaqa_formated_answers = {'Aliases': [], 'NormalizedAliases': [], \
                                         'NormalizedValue': '', \
                                         'Type': 'FreeForm', 'Value': ''}
            triviaqa_formated_answers['Value'] = filtered_answer[0]['answer']
            triviaqa_formated_answers['NormalizedValue'] = ' '.join(word_tokenize(filtered_answer[0]['answer'].lower()))
            for answer in filtered_answer:
                triviaqa_formated_answers['Aliases'] += answer['aliases']
                triviaqa_formated_answers['Aliases'].append(answer['answer'])

            triviaqa_formated_answers['NormalizedAliases'] = \
                [' '.join(word_tokenize(word.lower())) for word in triviaqa_formated_answers['Aliases']]
            all_answers.append(triviaqa_formated_answers)

    triviaqa_dict = {}
    triviaqa_dict['Domain'] = 'unfiltered-web'
    triviaqa_dict['Split'] = 'dev'
    triviaqa_dict['VerifiedEval'] = False
    triviaqa_dict['Version'] = 1.0

    # deleting prev evidence (only one at a time)
    if not os.path.isdir(TRIVIA_QA +'/evidence/batch_run'):
        os.mkdir(TRIVIA_QA + '/evidence/batch_run')

    if DELETE_PREV_EVIDENCE:
        shutil.rmtree(TRIVIA_QA + '/evidence/batch_run/')
        os.mkdir(TRIVIA_QA + '/evidence/batch_run')

    # create a query to file map
    WRITE_EVIDENCE = True
    train_file_ind = int(0)
    questions_triviaqa_format = questions_triviaqa_format.set_index('QuestionId')
    for questionID, question in questions_triviaqa_format.iterrows():
        # building 10 text files out of 100 snippets
        SearchResults = []
        files = []
        filenames = []
        file_ind = 0
        google_results = question['SearchResults']
        train_file_ind += 1

        if WRITE_EVIDENCE and not os.path.isdir(TRIVIA_QA + '/evidence/batch_run/' + str(int(train_file_ind / 100))):
            os.mkdir(TRIVIA_QA + '/evidence/batch_run/' + str(int(train_file_ind / 100)))
            if train_file_ind % 1000 == 0:
                print('evidence/batch_run/' + str(int(train_file_ind / 100)))

        for ind, g in enumerate(google_results):
            file_ind = file_ind % 30
            if len(files) <= file_ind:
                file_name = 'batch_run/' + str(int(train_file_ind / 100)
                                ) + "/" + questionID + '_' + str(file_ind) + '.txt'
                SearchResults.append(
                    {'Rank': ind, 'Description': g['snippet'], 'Title': g['title'], 'DisplayUrl': g['url'], \
                     'Url': g['url'] + file_name.replace('/', '_').replace('.txt', ''), 'Filename': file_name})
                files.append('')
                filenames.append(file_name)

            files[file_ind] += str(
                ind) + '. ' + g['title'] + '\n' + g['snippet'] + '\n'

            file_ind += 1

        # saving files
        if WRITE_EVIDENCE:
            for file_str, file_name in zip(files, filenames):
                with open(TRIVIA_QA + '/evidence/' + file_name, 'w') as outfile:
                    outfile.write(file_str)

        questions_triviaqa_format.at[questionID, 'SearchResults'] = SearchResults

    triviaqa_dict['Data'] = questions_triviaqa_format.reset_index().to_dict(orient='rows')

    #questions_triviaqa_format['Answer'] = all_answers
    #questions_triviaqa_format = questions_triviaqa_format[questions_triviaqa_format['Answer'].notnull()]
    if not os.path.isdir(TRIVIA_QA_UNFILTERED + '/batch_run'):
        os.mkdir(TRIVIA_QA_UNFILTERED + '/batch_run')
    with open(TRIVIA_QA_UNFILTERED + '/batch_run/unfiltered-web-dev.json','w') as f:
        f.write(json.dumps(triviaqa_dict, sort_keys=True, indent=4))

dirs_to_process = []
proc_running = []
dbx = dropbox.Dropbox('7j6m2s1jYC0AAAAAAAHy69fu0OxDAU3fPbIjjarqr_1zalj8Mvypf8U71BoLT-AD')
print('starting webkb run batch')
# searching inside the webanswer batch output
iter_count = 0
found_counter = 0
while True:
    time.sleep(1)
    #try:
    iter_count+=1
    for entry in dbx.files_list_folder('/docqa').entries:
        if entry.name.find('_done.json') > -1:
            print('fetching the next batch to work on')
            print('copying file to backup and backupdir : ' + entry.name )
            md, res = dbx.files_download('/docqa/' + entry.name)
            google_results = pd.DataFrame(json.loads(res.content))
            # moving the file to backup


            print('building traivia qa evidence from google results')
            build_evidence(google_results)

            print('running evidence_corpus')
            wa_proc = call('python docqa/triviaqa/evidence_corpus.py --n_processes 8 --source ' + \
                           join(TRIVIA_QA, "evidence", 'batch_run') + ' --output_dir ' + \
                           join(CORPUS_DIR, "triviaqa", "evidence", "web", 'batch_run'), shell=True,
                           preexec_fn=os.setsid)

            # running build_span_corpus.py
            source_dir = join(TRIVIA_QA_UNFILTERED, 'batch_run')
            target_dir = join(CORPUS_DIR, "triviaqa", "web-open", 'batch_run')
            print('running build_span_corpus')
            call('python docqa/triviaqa/build_span_corpus.py web-open --sets_to_build dev --n_processes 8 --source_dir ' + source_dir \
                 + ' --target_dir ' + target_dir, shell=True, preexec_fn=os.setsid)

            # running the docqa evaluation
            source_dir = target_dir
            wa_proc = call('python docqa/eval/triviaqa_full_document_eval.py --n_processes 8 --n_paragraphs 100  -c open-dev' + \
                           ' --tokens 800  ' + args.model + ' --source_dir ' + source_dir, shell=True,
                            preexec_fn=os.setsid)

            # storing results
            results = pd.read_csv('results.csv')
            google_results['results'] = None
            google_results['results'] = google_results['results'].astype(object)

            for ind,question in google_results.iterrows():
                q_results = results[results['question_id'] == ind][['text_answer','predicted_score']]
                q_results.columns = ['spans','scores']
                q_results = q_results.reset_index(drop=True)
                google_results.at[ind,'results'] = q_results.to_dict(orient='rows')

            if 'answers' in google_results.columns:
                del google_results['answers']

            for retry in range(0,10):
                try:
                    dbx.files_upload(json.dumps(google_results.to_dict(orient='rows'), indent=4, sort_keys=True).encode(), \
                             '/docqa_res/' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H_%M_%S') + \
                             '__' + entry.name)
                    break
                except:
                    print('upload failed')

            if retry == 9:
                raise ConnectionError('upload failed')

            dbx.files_move('/docqa/' + entry.name, '/cache/' \
                           + datetime.datetime.fromtimestamp(time.time()).strftime(
                '%Y-%m-%d_%H_%M_%S') + '__' + entry.name)



    #except:
    #    pass
