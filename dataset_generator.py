import os
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import tqdm
import string
from transformers import AutoTokenizer
import re
import copy

# <prefix<
# >suffix>
# {base}
# (root)


def make_dataset(morpho_path, configs = ["0-1-0", "1-1-0", "1-1-1", "0-1-1"]):
    df_temp = pd.ExcelFile(morpho_path)

    alphabet = string.ascii_lowercase

    rows = []
    bases = []

    for config in configs:
        df = pd.read_excel(df_temp, config)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        for _, row in df.iterrows():
            word = str(row["Word"]).lower().strip()
            if not all(c in alphabet for c in word):
                continue
            row["Word"] = word
            segm = row["MorphoLexSegm"]
            prefix = re.findall("<(.*?)<", segm)
            suffix = re.findall(">(.*?)>", segm)
            base = re.findall("\{(.*?)\}", segm)
            if base and "(" in base[0]: # searching for root
                root = re.findall("\((.*?)\)", segm)
            try:
                cleaned_base = re.sub("[{}()<>]", "", base[0]) # cleaning all special characters from base to extract the unmotivated words (this assumes that base == lemma)
            except:
                print("errore", word, segm)
                continue
            if prefix and suffix:
                if word.startswith(prefix[0]) and root[0] in word and word.endswith(suffix[0]):
                    # print(f"1-1-1 Word: {word}, prefix: {prefix[0]}, suffix: {suffix[0]}, base: {cleaned_base}, root: {root[0]}")
                    row["motivated"] = True
                    rows.append(row)
                    row["prefix"] = prefix[0]
                    row["suffix"] = suffix[0]
                    row["root"] = root[0]
            elif prefix and not suffix:
                if word.startswith(prefix[0]) and root[0] in word and word.endswith(cleaned_base):
                    # print(f"1-1-0 Word: {word}, prefix: {prefix[0]}, base: {cleaned_base}, root: {root[0]}")
                    row["motivated"] = True
                    row["prefix"] = prefix[0]
                    row["root"] = root[0]
                    rows.append(row)
                # else:
                #     print("excluding:", word)
            elif suffix and not prefix: 
                if root[0] in word and word.endswith(suffix[0]):
                    # print(f"0-1-1 Word: {word}, suffix: {suffix[0]}, base: {cleaned_base}, root: {root[0]}")
                    row["motivated"] = True
                    row["suffix"] = suffix[0]
                    row["root"] = root[0]
                    rows.append(row)
                # else:
                #     print("excluding:", word)
            else: #unmotivated
                if cleaned_base not in bases and word == cleaned_base:
                    bases.append(cleaned_base)
                    row["motivated"] = False
                    rows.append(row)

    final_df = pd.DataFrame(rows)
    print(final_df)
    print("Mot vs Unmot:", final_df["motivated"].value_counts())
    print(final_df.columns)
    print(final_df["prefix"].value_counts())
    print(final_df["suffix"].value_counts())
    print(final_df["root"].value_counts())
    final_df["to_drop"] = [x[0] in alphabet[0:-1:7] for x in final_df["Word"]] # making a column True/False for skipping words that starts with alphabet[0:-1:7] (ahov)
    df_skip_lemmas = final_df[final_df["to_drop"] == False].copy() # all words without the ones starting with alphabet[0:-1:7] (ahov) for the train split
    test_temp, train = train_test_split(df_skip_lemmas, test_size=200, random_state=42, stratify=df_skip_lemmas["motivated"])
    test_temp = pd.concat([test_temp, final_df[final_df["to_drop"] == True]]) # join the remaining skipped lemmas with the non skip ones to obtain a test and val that includes all words
    test, val = train_test_split(test_temp, test_size=200, random_state=42, stratify=test_temp["motivated"])
    print(f"Train: {train.shape}, {train['motivated'].value_counts()}")
    print(f"Val: {val.shape}, {val['motivated'].value_counts()}")
    print(f"Test: {test.shape}, {test['motivated'].value_counts()}")
    # train n_grams:
    n_grams_start_vocab, n_grams_middle_vocab, n_grams_end_vocab, n_grams_column = make_n_grams_vocab_and_col(train["Word"])
    train["n_grams"] = n_grams_column
    generate_examples(train, n_grams_start_vocab, n_grams_middle_vocab, n_grams_end_vocab, "train.csv", 300)
    # val n_grams
    n_grams_start_vocab, n_grams_middle_vocab, n_grams_end_vocab, n_grams_column = make_n_grams_vocab_and_col(val["Word"])
    val["n_grams"] = n_grams_column
    generate_examples(val, n_grams_start_vocab, n_grams_middle_vocab, n_grams_end_vocab, "val.csv", 300)
    # test n_grams
    n_grams_start_vocab, n_grams_middle_vocab, n_grams_end_vocab, n_grams_column = make_n_grams_vocab_and_col(test["Word"])
    test["n_grams"] = n_grams_column
    generate_examples(test, n_grams_start_vocab, n_grams_middle_vocab, n_grams_end_vocab, "test.csv", -1)

    return final_df


def make_n_grams_vocab_and_col(words):
    n_grams_start_vocab = {n: [] for n in range(1, 7)}
    n_grams_end_vocab = {n: [] for n in range(1, 7)}
    n_grams_middle_vocab = {n: [] for n in range(1, 7)}
    overall_number = 0
    n_grams_column = []
    for lemma in tqdm.tqdm(words, desc="n_grams"):
        n_grams = []
        for i in range(len(lemma)):
            for j in range(1, len(lemma) - i + 1):
                n_gram = lemma[i:i+j]
                if n_gram == lemma: # to avoid total identity 
                    continue
                if len(n_gram) <= 6 and len(n_gram) >= 1:
                    if i == 0:
                        n_gram = "_" + n_gram
                        n_grams_start_vocab[len(n_gram)-1].append(n_gram)
                    elif i != 0 and i+j == len(lemma):
                        n_gram = n_gram + "_"
                        n_grams_end_vocab[len(n_gram)-1].append(n_gram)
                    else:
                        n_grams_middle_vocab[len(n_gram)].append(n_gram)
                    n_grams.append(n_gram)
                    overall_number +=1
        n_grams_column.append(n_grams)

    n_grams_start_vocab = {k:list(set(v)) for k, v in n_grams_start_vocab.items()}
    n_grams_middle_vocab = {k:list(set(v)) for k, v in n_grams_middle_vocab.items()}
    n_grams_end_vocab = {k:list(set(v)) for k, v in n_grams_end_vocab.items()}
    
    return n_grams_start_vocab, n_grams_middle_vocab, n_grams_end_vocab, n_grams_column


def generate_examples(df, n_grams_start_vocab, n_grams_middle_vocab, n_grams_end_vocab, path, max_examples_per_class = 600):

    class_counters = {('start', n): 0 for n in range(1, 7)}
    class_counters.update({('end', n): 0 for n in range(1, 7)})
    class_counters.update({('middle', n): 0 for n in range(1, 7)})

    rows = []
    for _, row in tqdm.tqdm(df.iterrows(), desc="Making the dataset", total=len(df)):
        n_grams = row["n_grams"]
        for n_gram in n_grams: # we create a positive/negative instance for each n_gram
            # Determine the class of the n-gram
            n_gram_length = len(n_gram.replace("_", ""))
            if n_gram.startswith("_"):
                position = 'start'
            elif n_gram.endswith("_"):
                position = 'end'
            else:
                position = 'middle'
            class_key = (position, n_gram_length)

            # Check if we have reached the maximum examples for this class
            if max_examples_per_class != -1:
                if class_counters[class_key] >= max_examples_per_class:
                    continue  # Skip to the next n-gram

            #positive
            positive_row = copy.deepcopy(row)
            positive_row["target_n_gram"] = n_gram
            positive_row["source_text"] = f"Is [N_GRAM]{n_gram}[N_GRAM] inside {positive_row['Word']}?"
            positive_row["target_text"] = "yes"
            positive_row["n_gram_position"] = position


            if positive_row["motivated"]:
                if position == "start" and positive_row["prefix"] and n_gram.replace("_", "") == positive_row["prefix"]: # this means we have a prefix equal to starting n_gram
                    positive_row["is_motivated_n_gram"] = True
                elif position == "end" and positive_row["suffix"] and n_gram.replace("_", "") == positive_row["suffix"]: # this means we have a suffix equal to end n_gram
                    positive_row["is_motivated_n_gram"] = True
                elif position == "middle" and positive_row["root"] and n_gram == positive_row["root"]:
                    positive_row["is_motivated_n_gram"] = True
                else:
                    positive_row["is_motivated_n_gram"] = False
            else:
                positive_row["is_motivated_n_gram"] = False

            rows.append(positive_row) # this is a positve row

            # now we do the negative one
            negative_row = copy.deepcopy(row)
            if position == "start":
                negative_n_gram = random.choice(n_grams_start_vocab[len(n_gram)-1])
                while negative_row["Word"].startswith(negative_n_gram[1:]):
                    negative_n_gram = random.choice(n_grams_start_vocab[len(n_gram)-1])
            elif position == "end":
                negative_n_gram = random.choice(n_grams_end_vocab[len(n_gram)-1])
                while negative_row["Word"].endswith(negative_n_gram[:-1]):
                    negative_n_gram = random.choice(n_grams_end_vocab[len(n_gram)-1])
            else:
                negative_n_gram = random.choice(n_grams_middle_vocab[len(n_gram)])
                while negative_n_gram in negative_row["Word"][1:-1]:
                    negative_n_gram = random.choice(n_grams_middle_vocab[len(n_gram)])

            negative_row["target_n_gram"] = negative_n_gram
            negative_row["source_text"] = f"Is [N_GRAM]{negative_n_gram}[N_GRAM] inside {negative_row['Word']}?"
            negative_row["target_text"] = "no"
            negative_row["n_gram_position"] = position
            negative_row["is_motivated_n_gram"] = positive_row["is_motivated_n_gram"] # despite this cannot be true for negative examples we assign a yes label when the negative example is
            # generated from a positive motivated_ngram, this allows for having a balanced yes/no dataset for motivated n_grams
            rows.append(negative_row)

            class_counters[class_key] += 2  # Increment by 2 (positive and negative example)

    final_df = pd.DataFrame(rows)
    print(path, final_df.shape)
    print(final_df["motivated"].value_counts())
    print(final_df["is_motivated_n_gram"].value_counts())
    print(final_df["target_text"].value_counts())
    final_df.sample(frac=1).to_csv(path, index=False)

if __name__ == "__main__":
    random.seed = 42
    morpholex_path = "MorphoLEX_en.xlsx"
    make_dataset(morpholex_path)
