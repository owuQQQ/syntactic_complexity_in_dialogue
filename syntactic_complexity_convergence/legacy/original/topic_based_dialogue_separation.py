import pandas as pd
import torch
from transformers import pipeline
from preprocess_utterances import remove_special_symbols
from tqdm import tqdm


INPUT_CSV = "./English_data/combined_candor.csv"
OUTPUT_CSV = "./English_data/combined_candor_segmented.csv"
device = 0 if torch.cuda.is_available() else -1 
BATCH_SIZE = 16

print("Using device:", "cuda" if device==0 else "cpu")

classifier = pipeline(
    "zero-shot-classification",
    model="joeddav/xlm-roberta-large-xnli",
    device=device,
    batch_size=16 
)

TOPIC_LABELS = [
    "greeting",  "instruction", "clarification", "confirmation",
    "planning", "action", "location", "description",
    "personal_information", "narrative", "other"
]

def classify_topics_batch(sentences):
    """
    Batch zero-shot classification. Returns list of (label, score)
    """
    results = []
    for i in range(0, len(sentences), BATCH_SIZE):
        batch = sentences[i:i+BATCH_SIZE]
        batch_results = classifier(batch, candidate_labels=TOPIC_LABELS, multi_label=False)
        if isinstance(batch_results, dict):
            batch_results = [batch_results]
        for res in batch_results:
            results.append((res['labels'][0], res['scores'][0]))
    return results

def segment_dialogue(sentences, confidence_drop=0.30):
    if len(sentences) == 0:
        return [], []

    segments = []
    current_segment = [sentences[0]]

    topic_preds = classify_topics_batch(sentences)
    prev_topic, prev_conf = topic_preds[0]

    for idx, (topic, conf) in enumerate(topic_preds[1:], start=1):
        if topic != prev_topic and (prev_conf - conf) > confidence_drop:
            segments.append(current_segment)
            current_segment = [sentences[idx]]
        else:
            current_segment.append(sentences[idx])
        prev_topic, prev_conf = topic, conf

    segments.append(current_segment)
    return segments, topic_preds


df = pd.read_csv(INPUT_CSV)

if 'candor' in INPUT_CSV.lower():
    df = df.rename(columns={"id": "Dialogue_id", "speaker": 'Speaker', "utterance": 'Utterance'})
    df = remove_special_symbols(df, language='en')
    df = df[['Speaker', 'Utterance', "Dialogue_id", 'turn_id']]

elif 'maptask' in INPUT_CSV.lower():
    df = df.rename(columns={"Dialogue_ID": "Dialogue_id"})
    df = remove_special_symbols(df, language='en')
    df = df[['Speaker', 'Utterance', "Dialogue_id"]]

elif 'btsj' in INPUT_CSV.lower():
    df = df.rename(columns={"Dialogue": 'Utterance', "File_ID": "Dialogue_id"})
    df = remove_special_symbols(df, language='ja')

elif 'hazumi' in INPUT_CSV.lower():
    df = df.rename(columns={"Speaker_name": 'Speaker'})
    df = remove_special_symbols(df, language='ja')

df = df.sort_values(by=["Dialogue_id"]).reset_index(drop=True)

segmented_rows = []

for dlg_id, subdf in tqdm(df.groupby("Dialogue_id"), desc="Segmenting dialogues"):
    sentences = subdf["Utterance"].tolist()
    speakers = subdf["Speaker"].tolist()

    segments,topic_preds = segment_dialogue(sentences)

    seg_id = 1
    idx = 0
    for seg in segments:
        for _ in seg:
            label, score = topic_preds[idx]
            segmented_rows.append({
                "Dialogue_id": dlg_id,
                "segment_id": seg_id,
                "topic":label,
                "Speaker": speakers[idx],
                "Utterance": sentences[idx]
            })
            idx += 1
        seg_id += 1


out_df = pd.DataFrame(segmented_rows)
out_df.to_csv(OUTPUT_CSV, index=False)

# print(f"Saved segmented dialogue dataset to {OUTPUT_CSV}")

# if __name__ == '__main__':
#     dataset = ['combined_candor','maptask','btsj','Hazumi']
#     TOPIC_LABELS = [
#     "greeting",
#     "task",
#     "instruction",
#     "clarification",
#     "confirmation",
#     "planning",
#     "action",
#     "location",
#     "description",
#     "emotion",
#     "personal_information",
#     "narrative",
#     "other"
# ]
#     for i in dataset:
#         if i=='combined_candor' or 'maptask':
#             INPUT_CSV = f'./English_data/{i}.csv'
#             OUTPUT_CSV = f'./English_data/{i}__segmented.csv'
            
            


