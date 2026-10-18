from prompt_toolkit import Application
import stanza
import json
import torch
import os
import shutil
from os import listdir 
from os.path import isfile, join
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from preprocess_utterances import *
import time

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--language', type=str, default='de',
        choices=['en', 'de', 'ja'],
        help=
        'choose one of the following three language: English, German, Japanese'
    )
    parser.add_argument(
        '--input_path', type=str, default='/home/ywang/linguistic_complexity/json_combined_con',
        help='load locally if specified'
    )
    parser.add_argument('--output', '-o', type=str, default='',
                        help='output file'
    )
    parser.add_argument('--filename', type=str, default='',
                        help='specify a specific JSON filename to process (e.g., combined_candor.json). If not specified, all JSON files will be processed'
    )
    return parser

def dependency_parsing(args):
    lang = args.language
    path = args.input_path
    if torch.cuda.is_available():
        parser = stanza.Pipeline(lang, use_gpu=True, processors='tokenize,pos,lemma,depparse')
    else:
        parser = stanza.Pipeline(lang, processors='tokenize, pos,lemma,depparse')
    return parser

## A recursive function which calculates the tree depth of a given utterance
def tree_depth(sentence, dep_parser):

    def compute_depth(node_id, children_dict):
        if node_id not in children_dict:
            return 1
        else:
            return 1 + max(compute_depth(child, children_dict) for child in children_dict[node_id])
            
    doc = dep_parser(sentence)
    dependencies = doc.sentences[0] if doc.sentences else 0
    children_dict = {}
    for word in dependencies.words:
        if word.head == 0:
            root_id = word.id
        else:
            if word.head not in children_dict:
                children_dict[word.head] = []
            children_dict[word.head].append(word.id)

    tree_depth = compute_depth(root_id, children_dict)
    return tree_depth


##dependency length is the sum of the distance from each node to its head
def dependency_length(text,dep_parser):
    doc = dep_parser(text)
    distances = [abs(word.id - word.head) for sentence in doc.sentences for word in sentence.words if word.head > 0]
    return np.mean(distances) if distances else 0


##syntactic complexity, as measured by length of utterance divided by heads (the average nodes that head controls plus the tree deoth
def calculate_complexity(sentence, dep_parser):
    doc = dep_parser(sentence)
    sen_len = len(sentence.split())
    #dependencies = doc.sentences[0] if doc.sentences else 0
    depth = tree_depth(sentence, dep_parser)
    heads = [word.head for sent in doc.sentences for word in sent.words]
    count_head = len(list(set(heads)))
    complexity = 0.5*(sen_len/(count_head)) + 0.5 * depth if count_head>0 else 0.5 * depth
    return complexity


## Long distance proportion, calculate the ratio of syhntactic structures which have long dependency
def long_dependency_ratio(text, dep_parser, threshold=3):
    doc = dep_parser(text)
    long_deps = sum(1 for sentence in doc.sentences for word in sentence.words if word.head > 0 and abs(word.id - word.head) > threshold)
    total_deps = sum(1 for sentence in doc.sentences for word in sentence.words if word.head > 0)
    return long_deps / total_deps if total_deps > 0 else 0


def clear_cache():
    for folder in ["~/.cache", "/tmp", "/var/tmp"]:
        path = os.path.expanduser(folder)
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)


def sentence_syncom_metrics(text, dep_parser):
    comp = calculate_complexity(text, dep_parser)
    length = dependency_length(text, dep_parser)
    depth = tree_depth(text, dep_parser)
    prop_dep = long_dependency_ratio(text, dep_parser, threshold=2)
    return [comp, length, depth, prop_dep]


def calculate_syncom(texts, dep_parser):
    # Build an N x 4 matrix (N sentences, four metrics each)
    metrics = []
    for text in texts:
        sent = str(text).strip()
        if not sent:
            continue
        metrics.append(sentence_syncom_metrics(sent, dep_parser))
    return np.array(metrics, dtype=float) if metrics else np.empty((0, 4))


def summarize_syncom(sentences, dep_parser):
    metrics = calculate_syncom(sentences, dep_parser)
    if metrics.size == 0:
        return None
    return metrics.mean(axis=0)


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    json_path = args.input_path
    dep_parser = dependency_parsing(args)

    ## STEP 1: Calculate syntactic complexity metrics for example sentence

    #Example sentences kept for manual sanity checks. Uncomment to reuse the
    #original toy comparisons before running the CSV analysis block below.
    sentence1 = "es gibt drei Runden"
    sentence2 = "Also ich denke ich habe alles Wichtige erklärt"
    sentence_ori_dephaine1 = "So sollen die Treibhaus-Gase in der EU bis zum Jahr 2030 um mindestens 55 Prozent verringert werden."
    sentence_sim_dephaine2 = "Bis zum Jahr 2030 soll es in der EU um 55 Prozent weniger Treibhaus-Gase geben."
    ori_dephane_list = ["So sollen die Treibhaus-Gase in der EU bis zum Jahr 2030 um mindestens 55 Prozent verringert werden.",
                        "In den letzten 29 Jahren schaffte man eine Verringerung von rund 25 Prozent.",
                        "In der Nacht auf Mittwoch ist im Hafen der italienischen Stadt Ancona ein großer Brand ausgebrochen.",
                        "Die österreichische Fußball-Mannschaft Rapid Wien hat am Dienstag in der Qualifikation für die Champions League gegen Gent aus Belgien gespielt.",
                        "Wegen der vielen Neu-Infektionen haben Deutschland und Belgien die Stadt Wien zum Corona-Risikogebiet erklärt.",
                        "Quarantäne bedeutet, dass man nicht mit anderen Menschen in Kontakt kommen darf. Man wird von allen anderen Menschen getrennt.",
                        "Als sexueller Missbrauch gelten alle sexuellen Handlungen an Menschen, die das gar nicht wollen.",
                        "Dann sind bei privaten Feiern und Veranstaltungen in Innenräumen nur noch 10 Personen erlaubt."]
    sim_dephane_list = ["Bis zum Jahr 2030 soll es in der EU um 55 Prozent weniger Treibhaus-Gase geben.",
                        "Das ist sehr viel, denn in den letzten 29 Jahren schaffte man nur 25 Prozent.",
                        "Im Hafen der Stadt Ancona in Italien hat es in der Nacht auf Mittwoch einen großen Brand gegeben.",
                        "Rapid Wien ist eine von den besten Fußball-Mannschaften aus Österreich. Am Dienstag hat Rapid gegen Gent aus Belgien gespielt.",
                        "In Wien infizieren sich derzeit viele Menschen mit dem Corona-Virus. Deshalb haben Deutschland und Belgien eine Reise-Warnung für Wien herausgegeben.",
                        "Quarantäne bedeutet, man wird von allen anderen Menschen getrennt.",
                        "Sexueller Missbrauch sind sexuelle Handlungen an Menschen, die das gar nicht wollen.",
                        "Bei privaten Feiern im Inneren dürfen dann nicht mehr als 10 Personen dabei sein."]
    for i in ori_dephane_list:
        print(i)
        res = calculate_syncom([i], dep_parser)
        print(res)
    print("====================================")
    for i in sim_dephane_list:
        print(i)
        res = calculate_syncom([i], dep_parser)
        print(res)

    res1 = calculate_syncom([sentence1], dep_parser)
    res2 = calculate_syncom([sentence2], dep_parser)
    res3 = calculate_syncom([sentence_ori_dephaine1], dep_parser)
    res4 = calculate_syncom([sentence_sim_dephaine2], dep_parser)
    print(res1, res2, res3, res4)

    #### STEP 2: get the average through parsing the two datasets.
    datasets = [
        ('train_deplain.csv', 'train'),
        ('test_deplain.csv', 'test'),
    ]
    target_columns = ['original', 'simplification']

    for csv_name, label in datasets:
        csv_path = Path(csv_name)
        if not csv_path.exists():
            print(f'[WARN] {csv_path} not found, skipping"".')
            continue

        df = pd.read_csv(csv_path)
        for column in target_columns:
            if column not in df.columns:
                print(f'[WARN] Column "{column}" missing in {csv_name}, skipping.')
                continue

            average_metrics = summarize_syncom(df[column].dropna(), dep_parser)
            if average_metrics is None:
                print(f'[WARN] No valid sentences found for {label}:{column}.')
                continue

            comp, dep_len, depth, long_ratio = average_metrics
            print(
                f'{label}:{column} -> '
                f'complexity={comp:.4f}, '
                f'dependency_length={dep_len:.4f}, '
                f'tree_depth={depth:.4f}, '
                f'long_dependency_ratio={long_ratio:.4f}'
            ) 


      #### STEP 3 : get syncom metric value and save them to txt file for each utterance in the given dataset.

    # # Determine which files to process
    if args.filename:
        # Process only the specified file
        files_to_process = [args.filename]
    else:
        # Process all JSON files in the directory
        files_to_process = [f for f in os.listdir(json_path) if f.endswith(".json")]
    
    for filename in files_to_process:
        print(filename)
        if filename.endswith(".json") or args.filename: 
            file_path = os.path.join(json_path, filename)
            if 'english' in file_path:
                args.language = 'en'
            elif 'japanese' in file_path:
                args.language = 'ja'
            elif 'german' in file_path:
                args.language ='de'
            elif 'candor' in file_path or 'maptask' in file_path:
                args.language = 'en'
            elif 'btsj' in file_path or 'Hazumi' in file_path:
                args.language = 'ja'
            elif 'a2' in file_path or 'texprax' in file_path or 'vm2' in file_path:
                args.language ='de'
                print(args.language)
                
                # Using a conditional for GPU to keep the line readable and avoid indentation slips
            if torch.cuda.is_available():
                dep_parser = stanza.Pipeline(args.language, use_gpu=True, processors='tokenize,pos,lemma,depparse')
            else:
                dep_parser = stanza.Pipeline(args.language, processors='tokenize,pos,lemma,depparse')

            with open(file_path, "r", encoding="utf-8") as f_in:
                data = json.load(f_in)
                list_of_lists = [[key, value] for key, value in data.items()]
                processed_data = []
                    
                res_file = './json_combined_con/' + os.path.splitext(os.path.basename(file_path))[0] + '_symcom.txt'
                columns = ["Index", "syn_value", "dependency_length", "tree_depth", "complex_dependency_ratio"]
                    
                with open(res_file, "w", encoding="utf-8") as f_out:
                    f_out.write("\t".join(columns) + "\n")
                    for index, sentences in list_of_lists:
                        print(index)
                        for sentence in sentences:
                                float_values = [index] + calculate_syncom([sentence], dep_parser)
                                f_out.write("\t".join(map(str, float_values)) + "\n")
