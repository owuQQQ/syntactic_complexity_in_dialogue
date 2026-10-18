import os
from os import listdir 
from os.path import isfile, join
from pathlib import Path
import nltk
import spacy
import emoji
import pandas as pd
import re
import json
import glob
from preprocess_utterances import remove_special_symbols

### the dictionary connective_ls contains the discourse markers from three language: English, Japanese and German

connectives_ls = {
    "en":['and, also, moreover, furthermore, besides, in addition,but, however, yet, nevertheless, still, though, although, even though, on the other hand, whereas, while,because, since, as, for, given that, seeing that, considering, so, therefore, thus, hence, consequently, as a result, accordingly, if, unless, provided that, as long as, in case, in the event that, supposing,when, while, once, after, before, until, then, next, eventually, finally, for example, for instance, such as, namely, that is, in other words, like, in conclusion, in sum, overall, all in all, to summarize, to conclude, in short, ultimately'],
    
    "de":['äh, Ähm,und,mhm,auch, außerdem, zudem, ferner, darüber hinaus,aber, jedoch, doch, allerdings, obwohl, obgleich, obschon, trotzdem, dennoch, wohingegen, während,weil, da, denn, zumal, angesichts,also, deshalb, deswegen, daher, somit, folglich, infolgedessen,wenn, falls, sofern, es sei denn, vorausgesetzt dass,als, wenn, während, nachdem, bevor, bis, dann, danach, schließlich, zuletzt,zum Beispiel, beispielsweise, etwa, das heißt, nämlich, mit anderen Worten,insgesamt, zusammenfassend, abschließend, schließlich, letztlich, kurz gesagt'],
    
    "ja":['そして, また, それから, しかも, さらに, おまけに, でも, しかし, けれども, けど, だが, なのに, 一方で, それなのに, ところが, だから, から, なぜなら, というのは,それで, したがって, その結果, よって, だからこそ, そういうわけで, たら, ならば, とすれば, としたら,とき, てから, その後, それから, やがて, ついに, 最後に, まず, 次に, 例えば, つまり, 要するに, すなわち, 言い換えれば, 具体的に言うと,全体として, 結論として, 要するに, 結局, まとめると, 最終的に']    
}


##input: a csv file with dialogue separated by the two speakers##
##output: a json file with dialogue merged, either with or without discourse markers being replaced


def norm(x):
    return re.sub(r'_(ee|er)$','', x)

def df_splitting(df_path):
    df = pd.read_csv(df_path)
    if 'texprax' in df_path:
        ids = df['id'].tolist()
        ls1,ls2,ls3 = [], [], []
        for item in ids:
            parts = item.split('_')
            ls1.append(parts[0])
            ls2.append(parts[1])
            ls3.append(parts[2])
        df['Dialogue_id'] = ls1
        df['turn_id'] = ls2
        df['sentence_id'] = ls3
        df.drop(columns=['id','label','subsplit'], inplace=True)
        df =df.rename(columns={"sentence": 'Utterance'})
        df = remove_special_symbols(df, language = 'de')
        return df 
        
    elif 'A2' in df_path:
        df.drop(columns=['speaker_name'], inplace=True)
        df = remove_special_symbols(df, language = 'de')
        return df
        
    elif 'vm2' in df_path:
        df =df.rename(columns={"Dialogue_ID": "Dialogue_id"})
        df = remove_special_symbols(df, language = 'de')
        return df
        
    elif 'combined_candor' in df_path:
        df =df.rename(columns={"id": "Dialogue_id", "speaker": 'Speaker',"utterance":'Utterance'})
        df = remove_special_symbols(df, language = 'en')
        df = df[['Speaker','Utterance',"Dialogue_id",'turn_id']]
        return df
        
    elif 'Hazumi' in df_path:
        df =df.rename(columns={"Speaker_name": 'Speaker'})
        df = remove_special_symbols(df, language = 'ja')
        return df
        
    elif 'btsj' in df_path:
        df = df.rename(columns={"Dialogue":'Utterance', "File_ID": "Dialogue_id"})
        df = remove_special_symbols(df, language = 'ja')
        return df

def parse_csv(path):
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    for path in csv_files:
        df = df_splitting(path)
        check_cols = ["Utterance"] 
        if "A2" in path:
            df['norm_id'] = df['id'].apply(norm)
            group_key = "norm_id"
        else:
            group_key = "Dialogue_id"
        valid_groups = []
        for gid, group in df.groupby(group_key):
            if not group[check_cols].isna().any().any():   # if NO NaN inside group
                valid_groups.append(group)
                
        if len(valid_groups) == 0:
            print(f"No valid groups in file: {path}")
            continue

        df = pd.concat(valid_groups)

        if "A2" in path:
            df_sorted = df.sort_values(['norm_id', 'Start'])
            out = {nid: group["Utterance"].tolist() for nid, group in df_sorted.groupby("norm_id")}
            out_path = './json_combined_con/A2.json'
        else:
            df_sorted = df.sort_values(['Dialogue_id'])
            out = {nid: group["Utterance"].tolist() for nid, group in df_sorted.groupby("Dialogue_id")}

            if "texprax" in path:
                out_path = './json_combined_con/texprax.json'
            elif "vm2" in path:
                out_path = './json_combined_con/vm2.json'
            elif "combined_candor" in path:
                out_path = './json_combined_con/combined_candor.json'
            elif "Hazumi" in path:
                out_path = './json_combined_con/Hazumi.json'
            elif "btsj" in path:
                out_path = './json_combined_con/btsj.json'

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
                
def parse_csv_no_connectives(path):
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    for file_path in csv_files:
        df = df_splitting(file_path)
        if "A2" in file_path:
            df['norm_id'] = df['id'].apply(norm)
            df_sorted = df.sort_values(['norm_id', 'Start'])
            grouped = df_sorted.groupby("norm_id")
            out_path = './json_combined_con/A2.json'
            lang = 'de'
        elif ("vm2" in file_path) or ("texprax" in file_path):
            df_sorted = df.sort_values(['Dialogue_id'])
            grouped = df_sorted.groupby("Dialogue_id")
            lang = 'de'
            if "texprax" in file_path:
                out_path = './json_combined_no_con/texprax.json'
            else:
                out_path = './json_combined_no_con/vm2.json'
        elif "combined_candor" in file_path:
            df_sorted = df.sort_values(['Dialogue_id'])
            grouped = df_sorted.groupby("Dialogue_id")
            out_path = './json_combined_no_con/combined_candor.json'
            lang = 'en'
        else:
            df_sorted = df.sort_values(['Dialogue_id'])
            grouped = df_sorted.groupby("Dialogue_id")
            lang = 'ja'
            if "Hazumi" in file_path:
                out_path = './json_combined_no_con/Hazumi.json'
            elif "btsj" in file_path:
                out_path = './json_combined_no_con/btsj.json'
            else:
                # No valid target → skip
                print("Unknown dataset:", file_path)
                continue
        out = {nid: group["Utterance"].tolist() for nid, group in grouped}

        out_replaced = {
            nid: [replace_connectives(utt, language=lang) for utt in utt_list]
            for nid, utt_list in out.items()
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_replaced, f, indent=2, ensure_ascii=False)

def replace_connectives(sentence, language):
    
    if not isinstance(sentence, str):
        sentence = "" if sentence is None else str(sentence)
        
    elif language=='en':
        to_check = [x.strip(",.!?;:").lower() for x in connectives_ls['en'][0].split(',')]
        to_check = [s.strip() for s in to_check]
        new_tokens = ["the" if word.strip(",.!?;:").lower() in to_check else word for word in sentence.split(' ')]
    elif language =='de':
        to_check = [x.strip(",.!?;:").lower() for x in connectives_ls['de'][0].split(',')]
        to_check = [s.strip() for s in to_check]
        new_tokens = ["das" if str(word).strip(",.!?;:").lower() in to_check else str(word)
                        for word in str(sentence).split()]   
    elif language =='ja':
        to_check = [x.strip(",.!?;:").lower() for x in connectives_ls['ja'][0].split(',')]
        to_check = [s.strip() for s in to_check]
        new_tokens = ["が" if word.strip(",.!?;:").lower() in to_check else word for word in sentence.split(' ')]
    new_sentence = " ".join(new_tokens)
    return new_sentence

if __name__ == '__main__':
    paths =  ['./German_data/','./Japanese_data/','./English_data/']
    for path in paths:
        parse_csv(path)
        #parse_csv_no_connectives(path)
            
            
        
    
    
        
    
    
        
        
    
    
    
