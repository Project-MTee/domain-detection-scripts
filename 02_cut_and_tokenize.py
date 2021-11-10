import os
import argparse
import jsonlines
from random import shuffle

class CustomLimit(argparse.Action):
    def __call__( self , parser, namespace,
                 values, option_string = None):
        setattr(namespace, self.dest, dict())
          
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = int(value)


_domains = ["general", "crisis", "legal", "military"]
_langs = ["et", "en", "ru", "de"]
_label_map = {"general": 0, "crisis": 1, "legal": 2, "military": 3}

def get_domain(path):
    for domain in _domains:
        if domain in path:
            #print("Found \"{}\" in {}".format(domain, path))
            return domain


def get_lang(path):
    for lang in _langs:
        if path[-3:] == ("." + lang):
            #print("Found \"{}\" in {}".format(lang, path))
            return lang


def get_split(path):
    for split in ["train", "test", "valid"]:
        if split in path:
            return split 


def get_domain_lang_limits(args):
    limits_dict = {}
    for domain in _domains:
        for lang in _langs:
            key = domain + "-" + lang
            if args.custom_limit and key in args.custom_limit:
                limits_dict[key] = args.custom_limit[key]
            else:
                limits_dict[key] = args.limit

    return limits_dict


def get_domain_lang_mono(mono_files, domain, lang):
    for file in mono_files:
        if domain in file and file[-3:] == lang:
            return file
    

def snip_from_file(filepath, domain, lang, limit):
    data = []
    
    with open(filepath, mode="r") as f:
        ix = 0
        line = f.readline()
        while line:
            data.append({"text":line.strip(),
                        "domain":domain,
                        "lang":lang,
                        "filepath": filepath})
            ix += 1
            if ix >= limit:
                break
            line = f.readline()

    return data

def take_all(filepath, domain, lang):
    data = []

    with open(filepath, mode="r") as f:
        line = f.readline()
        while line:
            data.append({"text":line.strip(),
                        "domain":domain,
                        "lang":lang,
                        "filepath": filepath})
            line = f.readline()

    return data

def tokenize_and_write_to_file(f, items, tokenizer):
    for item in items:
        tok = tokenizer(item["text"], max_length=256, truncation=True)
        item["input_ids"] = tok["input_ids"]
        item["attention_mask"] = tok["attention_mask"]
        item["label"] = _label_map[item["domain"]]
        f.write(item)

def cut_and_tokenize(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, is_split_into_words=False)
    print("Loaded tokenizer {}".format(args.tokenizer))

    os.makedirs(args.out_dir, exist_ok=True)

    limits_dict = get_domain_lang_limits(args)


    parallel_files = os.listdir(args.parallel_data_dir)
    mono_files = os.listdir(args.mono_data_dir)

    train_out = jsonlines.open(os.path.join(args.out_dir, "train.tok"), mode="w")
    test_out = jsonlines.open(os.path.join(args.out_dir, "test.tok"), mode="w")
    valid_out = jsonlines.open(os.path.join(args.out_dir, "valid.tok"), mode="w")

    for file in parallel_files:
        filepath = os.path.join(args.parallel_data_dir, file)
        split = get_split(filepath)
        domain = get_domain(filepath)
        lang = get_lang(filepath)
        limit = limits_dict[domain + "-" + lang]

        if split == "train":
            snip = snip_from_file(filepath, domain, lang, limit)
            print("Snipped {} sentences from {}.".format(len(snip), filepath))

            tokenize_and_write_to_file(train_out, snip, tokenizer)

            missing = limit - len(snip)

            if missing > 0:
                mono_file = get_domain_lang_mono(mono_files, domain, lang)
                print("Snipping {} from \"{}-{}\" data from mono file {}".format(missing, domain, lang, mono_file))
                        
                snip = snip_from_file(mono_file, domain, lang, missing)

                tokenize_and_write_to_file(train_out, snip, tokenizer)
        
        elif split == "test":
            snip = take_all(filepath, domain, lang)
            tokenize_and_write_to_file(test_out, snip,  tokenizer)
        
        elif split == "valid":
            snip = take_all(filepath, domain, lang)
            tokenize_and_write_to_file(valid_out, snip,  tokenizer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""This script takes the specified amount of sentences from each domain-lang pair for training data. It prefers parallel data sentences but if there's not enough it takes more from monolingual data. The sentences are tokenized and saved as jsonlines to the specified folder. NB! Note that: 1) The script doesn't do any shuffling of the data; 2) The limits don't apply to validation and test data, in case of that all data is taken.""")

    parser.add_argument("--parallel_data_dir", type=str, help="path to parallel data files")
    parser.add_argument("--mono_data_dir", type=str, help="path mono data files")
    parser.add_argument("--out_dir", type=str, help="folder where tokenized data is saved")
    parser.add_argument("--tokenizer", type=str, help="name of the tokenizer in HuggingFace")
    parser.add_argument("--limit", type=int, help="train sentences limit for domain-lang pair")
    parser.add_argument("--custom_limit", nargs="*", action = CustomLimit, help="key-value pairs of domain-langs and sentence limits, for example 'general-et=15000'")

    args = parser.parse_args()
    
    cut_and_tokenize(args)