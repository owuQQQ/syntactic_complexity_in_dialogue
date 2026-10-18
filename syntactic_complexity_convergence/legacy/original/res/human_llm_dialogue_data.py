import os
import re
import datasets
from datasets import load_dataset
import pandas as pd
import json
import langid

os.environ["HF_HOME"] = "../DATA/huggingface_cache"
os.environ["HF_DATASETS_CACHE"] = "../DATA/huggingface_cache/datasets"
os.environ["TRANSFORMERS_CACHE"] = "../DATA/huggingface_cache/models"
os.environ["HF_MODULES_CACHE"] = "../DATA/huggingface_cache/modules"

langid.set_languages(["en", "de", "ja", "zh"])

allowed_langs = {"en", "de", "ja"}
blocked_langs = {"zh"} 


# user_token = “”

# user_metadata_fields = ['age', 'gender', 'location']
# conversation_metadata_fields = [
#     'language', 'redacted', 'toxic', 'rate', 'title',
#     'custom_instruction', 'status', 'redacted'
# ]

# ours = datasets.load_dataset("shachardon/ShareLM")["train"]
# lmsys_raw = datasets.load_dataset("lmsys/lmsys-chat-1m", token=user_token)["train"]

# def reorder_user_metadata(example):
#     meta = example["user_metadata"]
#     return {
#         **example,
#         "user_metadata": {
#             field: meta.get(field, "") for field in user_metadata_fields
#         }
#     }

# ours_fixed = ours.map(reorder_user_metadata)
# lmsys_small = lmsys_raw.shuffle(seed=42).select(range(10000))

# examples = []
# for data in lmsys_small:
#     user_metadata = {field: "" for field in user_metadata_fields}

#     conversation_metadata = {
#         field: data.get(field, "") if field in data else ""
#         for field in conversation_metadata_fields
#     }

#     example = {
#         "conversation_id": data["conversation_id"],
#         "conversation": data["conversation"],
#         "source": "https://huggingface.co/datasets/lmsys/lmsys-chat-1m",
#         "model_name": data["model"],
#         "user_id": "",
#         "user_metadata": user_metadata,
#         "timestamp": "",
#         "conversation_metadata": conversation_metadata
#     }
#     examples.append(example)

# lmsys_formatted_dataset = datasets.Dataset.from_pandas(pd.DataFrame(examples))
# lmsys_fixed = lmsys_formatted_dataset.map(reorder_user_metadata)


def detect_langs(text):
    """Return the set of languages detected line-by-line."""
    langs = set()
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        lang, _ = langid.classify(line)
        if lang in ["en", "de", "ja", "zh"]:
            langs.add(lang)
    return langs


def is_mixed_language(text):
    langs = detect_langs(text)

    if "zh" in langs:
        return True

    langs = langs.intersection(allowed_langs)
    return len(langs) != 1


def is_unstructured(text):
    """Detect code, garbage, repeated tokens, etc."""
    code_patterns = [
        r"<\?php", r"\?>", r"\$[A-Za-z0-9_]+",
        r"\{.*?\}", r";", r"\(\)", r"function "
    ]
    for pat in code_patterns:
        if re.search(pat, text):
            return True

    if len(re.findall(r"[{}$<>;]", text)) > 5:
        return True

    tokens = text.split()
    if len(tokens) > 0:
        most_freq = max(tokens.count(t) for t in set(tokens))
        if most_freq > 20:
            return True

    return False

ds1 = load_dataset("lmsys/lmsys-chat-1m", split="train[:50000]")
ds2 = load_dataset("lmsys/lmsys-chat-1m", split="train[50000:100000]") 

def extract_conversation_rows(ds):
    rows = []
    allowed_meta_langs = {"English", "German", "Japanese"}

    for item in ds:
        lang = item.get("language", "")
        if lang not in allowed_meta_langs:
            continue

        cid = item.get("conversation_id")
        conversation = item.get("conversation", [])

        for turn in conversation:
            rows.append({
                "conversation_id": cid,
                "role": turn.get("role"),
                "content": turn.get("content"),
                "language": lang
            })

    return rows


rows = extract_conversation_rows(ds1) + extract_conversation_rows(ds2)
df = pd.DataFrame(rows)
df = df.dropna(subset=["content", "conversation_id", "language", "role"])

# rows1 = extract_conversation_rows(ds1)
# rows2 = extract_conversation_rows(ds2)
# df1 = pd.DataFrame(rows1)
# df2 = pd.DataFrame(rows2)

#### TODO, dataset cleaning
## 1. filtering out invalid dialogues, and focusing only on the communication like dialogue
## 2. filtering out invalid utterance, e.g., labeled as German or Japanese, but later other language appears instead.

df["is_mixed"] = df["content"].apply(is_mixed_language)
df["is_garbage"] = df["content"].apply(is_unstructured)

bad_dialogues = set(
    df.loc[df["is_mixed"] | df["is_garbage"], "conversation_id"]
)

clean_df = df[~df["conversation_id"].isin(bad_dialogues)]
clean_df = clean_df.drop(columns=["is_mixed", "is_garbage"])


### Here we do the grouping on the clean dialogues
groups = clean_df.groupby("conversation_id")
languages = ["English", "German", "Japanese"]
joint = {lang: {} for lang in languages}
initiator = {lang: {} for lang in languages}
follower = {lang: {} for lang in languages}


for cid, g in groups:
    lang = g["language"].iloc[0]
    if lang not in languages:
        continue
    g = g.sort_index()
    joint[lang][cid] = g["content"].tolist()
    initiator[lang][cid] = g[g["role"] == "user"]["content"].tolist()
    follower[lang][cid] = g[g["role"] == "assistant"]["content"].tolist()


for lang in languages:
    with open(f"llm_{lang.lower()}_joint.json", "w", encoding="utf-8") as f:
        json.dump(joint[lang], f, ensure_ascii=False, indent=2)

    with open(f"llm_{lang.lower()}_initiator.json", "w", encoding="utf-8") as f:
        json.dump(initiator[lang], f, ensure_ascii=False, indent=2)

    with open(f"llm_{lang.lower()}_follower.json", "w", encoding="utf-8") as f:
        json.dump(follower[lang], f, ensure_ascii=False, indent=2)



    

