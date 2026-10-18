import os
import re
import json
import langid
import pandas as pd
from datasets import load_dataset

# --------------------------------------------
# Language setup
# --------------------------------------------
langid.set_languages(["en", "de", "ja", "zh"])
allowed_langs = {"en", "de", "ja"}

def detect_langs(text):
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

    # Chinese → reject entirely
    if "zh" in langs:
        return True

    # Allowed languages only
    langs = langs.intersection(allowed_langs)

    # Mixed (0 or >1)
    return len(langs) != 1

def is_unstructured(text):
    code_signals = [
        r"<\?php", r"\?>", r"\{.*?\}", r";", r"function ", r"\$[A-Za-z_]+"
    ]
    for pat in code_signals:
        if re.search(pat, text):
            return True

    if len(re.findall(r"[{}$<>;]", text)) > 5:
        return True

    tokens = text.split()
    if tokens:
        most_common = max(tokens.count(t) for t in set(tokens))
        if most_common > 20:
            return True

    return False

def get_major_language(text):
    langs = detect_langs(text)
    langs = langs.intersection(allowed_langs)
    if len(langs) == 1:
        return list(langs)[0]
    return None

ds = load_dataset("shachardon/ShareLM",  split="train[:200000]")
print("Loaded ShareLM rows:", len(ds))
rows = []

for item in ds:
    cid = item["conversation_id"]
    conversation = item.get("conversation", [])

    if not conversation:
        continue

    for turn in conversation:
        role = turn.get("role", "")
        content = (turn.get("content") or "").strip()
        if not content:
            continue

        rows.append({
            "conversation_id": cid,
            "role": role,
            "content": content
        })

df = pd.DataFrame(rows)
print("DF size after loading:", df.shape)

if df.empty:
    raise ValueError("ERROR: No valid conversation content found.")

conversation_lang = {}

for cid, g in df.groupby("conversation_id"):
    text = " ".join(g["content"].tolist())
    lang = get_major_language(text)
    conversation_lang[cid] = lang

df["language"] = df["conversation_id"].map(conversation_lang)

df = df[df["language"].notna()]
print("After language assignment:", df.shape)

df["is_mixed"] = df["content"].apply(is_mixed_language)
df["is_garbage"] = df["content"].apply(is_unstructured)

bad_cids = set(
    df.loc[df["is_mixed"] | df["is_garbage"], "conversation_id"]
)

df = df[~df["conversation_id"].isin(bad_cids)]
df = df.drop(columns=["is_mixed", "is_garbage"])

print("Final cleaned DF:", df.shape)
df = df[["conversation_id", "content", "role", "language"]]

languages = ["en", "de", "ja"]
joint = {lang: {} for lang in languages}
initiator = {lang: {} for lang in languages}
follower = {lang: {} for lang in languages}

for cid, g in df.groupby("conversation_id"):
    lang = g["language"].iloc[0]
    g = g.sort_index()

    joint[lang][cid] = g["content"].tolist()
    initiator[lang][cid] = g[g["role"] == "user"]["content"].tolist()
    follower[lang][cid] = g[g["role"] == "assistant"]["content"].tolist()

for lang in languages:
    with open(f"sharelm_{lang}_joint.json", "w", encoding="utf-8") as f:
        json.dump(joint[lang], f, ensure_ascii=False, indent=2)

    with open(f"sharelm_{lang}_initiator.json", "w", encoding="utf-8") as f:
        json.dump(initiator[lang], f, ensure_ascii=False, indent=2)

    with open(f"sharelm_{lang}_follower.json", "w", encoding="utf-8") as f:
        json.dump(follower[lang], f, ensure_ascii=False, indent=2)
