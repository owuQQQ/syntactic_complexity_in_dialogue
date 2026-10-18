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


data_path = ['./English_data', './German_data', './Japanese_data']


def convert_text(text):
    converted_text = re.sub(r'[「」]', '', text)
    converted_text = re.sub(r'の', ' の ', converted_text)
    converted_text = re.sub(r',+', '、', converted_text) 
    converted_text = ' '.join(converted_text.split()) 
    return converted_text


def replace_speaker_name(df):
    df_copy = df.copy()
    for dialogue_id in df_copy['Dialogue_id'].unique():
        dialogue_subset = df_copy[df_copy['Dialogue_id'] == dialogue_id]
        unique_speakers = dialogue_subset['Speaker'].unique()
        # if len(unique_speakers) != 2:
        #     print(unique_speakers)
        #     #raise ValueError(f"Dialogue {dialogue_id} does not have exactly 2 unique speakers.")
        for idx, row in dialogue_subset.iterrows():
            speaker = row['Speaker']
            utterance = row['Utterance']
            if speaker == unique_speakers[0]:
                updated_utterance = re.sub(rf'{re.escape(speaker)}姓', 'X', utterance)
                updated_utterance = re.sub(r'話者不明', 'X', updated_utterance)
            else:
                updated_utterance = re.sub(rf'{re.escape(speaker)}姓', 'Y', utterance)
            df_copy.at[idx, 'Utterance'] = updated_utterance
    
    return df_copy
    

def remove_special_symbols(df, language = 'en'):
        
    if language == 'en':
        df['Utterance'] = df['Utterance'].replace(r'<(?=[a-f0-9]*[a-f])(?=[a-f0-9]*\d)[a-f0-9]+>', '<NAME>', regex=True)
        return df
        
    elif language == 'ja':
        df['Utterance'] = df['Utterance'].replace(r'\[.*?\]', '', regex=True) 
        df['Utterance'] = df['Utterance'].replace(r'《.*?》', '', regex=True) 
        df['Utterance'] = df['Utterance'].replace(r'{.*?}', '', regex=True) 
        df['Utterance'] = df['Utterance'].replace(r'\(<笑い>\)', '', regex=True) 
        df['Utterance'] = df['Utterance'].replace(r'、<([^<>]*)>\{<\}', r'\1', regex=True) 
        df['Utterance'] = df['Utterance'].replace(r'<笑い>|<笑いながら>|<2人笑い>|(<笑い>)|<少し笑いながら>|<2人で笑い>|<ふふっ軽い笑い>|<ふ軽い笑い>|<い軽く笑いながら>|(軽い笑い)|ふっふふふ軽い笑い|<い軽く笑いながら>', '', regex=True)
        df['Utterance'] = df['Utterance'].replace(r'<([^<>]*)>', r'\1', regex=True)
        df['Utterance'] = df['Utterance'].replace(r'【【|】】', r'', regex=True)
        df['Utterance'] = df['Utterance'].str.replace(r'\|', '', regex=True) 
        df['Utterance'] = df['Utterance'].str.replace(r'\([A-Z] ([^)]+)\)', r'\1,', regex=True)
        df['Utterance'] = df['Utterance'].apply(convert_text)
        df = replace_speaker_name(df)
        return df
        
    elif language == 'de': 
        df['Utterance'] = df['Utterance'].str.replace(r'\-', '', regex=True) 
        df['Utterance'] = df['Utterance'].replace(r'<(?=[a-f0-9]*[a-f])(?=[a-f0-9]*\d)[a-f0-9]+>', '<NAME>', regex=True)
        df['Utterance'] = df['Utterance'].str.replace(r'<(.*?)>', r'\1', regex=True)
        return df     


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
        print(df.columns)
        df = remove_special_symbols(df, language = 'de')
        return df 
        
    elif 'A2' in df_path:
        df[['Dialogue_id', 'Speaker']] = df['id'].str.rsplit('_', n=1, expand=True)
        df.drop(columns=['id','speaker_name'], inplace=True)
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


def df_to_dict(grouped_df):
    dialogue_dict = {}
    for (dialogue_id, turn_id), sentences in grouped_df.items():
        if dialogue_id not in dialogue_dict:
            dialogue_dict[dialogue_id] = {}
        if dialogue_id in dialogue_dict:
            if 'prev_turn_id' not in locals():
                prev_turn_id = None
            if prev_turn_id is None:
                new_turn_id = 0
            else:
                new_turn_id = 0 if prev_turn_id == 1 else 1   
            prev_turn_id = new_turn_id
        else:
            new_turn_id = turn_id
        dialogue_dict[dialogue_id].setdefault(new_turn_id, []).extend(sentences)

    filtered_dialogue_dict_f = {dialogue_id:  sentences for dialogue_id, dialogues in dialogue_dict.items() for turn_id, sentences in dialogues.items() if turn_id == 0}
    filtered_dialogue_dict_i = {dialogue_id:  sentences for dialogue_id, dialogues in dialogue_dict.items() for turn_id, sentences in dialogues.items() if turn_id == 1}
    return filtered_dialogue_dict_i, filtered_dialogue_dict_f

def separation_de(df_path):
    df = df_splitting(df_path)
    grouped_df = df.groupby(['Dialogue_id', 'turn_id'])['Utterance'].apply(list) if 'texprax' in df_path else df.groupby(['Dialogue_id', 'Speaker'])['Utterance'].apply(list)
    return df_to_dict(grouped_df)    

def separation_ja(df_path):
    df = df_splitting(df_path)
    grouped_df = df.groupby(['Dialogue_id', 'Speaker'])['Utterance'].apply(list)    
    return df_to_dict(grouped_df)

def separation_en(df_path):
    df = df_splitting(df_path)
    grouped_df = df.groupby(['Dialogue_id', 'Speaker'])['Utterance'].apply(list)
    return df_to_dict(grouped_df)
    

def dict_to_json (path_name):
    sub ='./json/'
    if 'German' in path_name:
        dict1,dict2 = separation_de(path)
        file_name = os.path.splitext(os.path.basename(path))[0] 
        initiator_name = sub + file_name + '_initiator.json'
        follower_name = sub + file_name + '_follower.json'
        with open(initiator_name, "w") as f:
            json.dump(dict1, f)
        with open(follower_name, "w") as f:
            json.dump(dict2, f)
    elif 'English' in path_name:
        dict1, dict2 = separation_en(path)
        file_name = os.path.splitext(os.path.basename(path))[0] 
        initiator_name = sub + file_name + '_initiator.json'
        follower_name = sub + file_name + '_follower.json'
        with open(initiator_name, "w") as f:
            json.dump(dict1, f)
        with open(follower_name, "w") as f:
            json.dump(dict2, f)

    elif 'Japanese' in path_name:
        dict1, dict2 = separation_en(path)
        file_name = os.path.splitext(os.path.basename(path))[0]
        initiator_name = sub + file_name + '_initiator.json'
        follower_name = sub + file_name + '_follower.json'
        with open(initiator_name, "w") as f:
            json.dump(dict1, f)
        with open(follower_name, "w") as f:
            json.dump(dict2, f)

if __name__ == '__main__':
    paths = [
'./German_data/A2.csv','./German_data/texprax.csv','./German_data/vm2.csv','./Japanese_data/btsj.csv','./Japanese_data/Hazumi.csv','./English_data/combined_candor.csv']
    for path in paths:
        dict_to_json(path)
    
