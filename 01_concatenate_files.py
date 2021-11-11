import os
import argparse

_domains = ["general", "crisis", "legal", "military"]
_splits = ["train", "valid", "test"]


def get_domain(path):
    for domain in _domains:
        if domain in path:
            print("Found \"{}\" in {}".format(domain, path))
            return domain

def get_split(path):
    for split in _splits:
        if split in path:
            print("Found \"{}\" in {}".format(split, path))
            return split
    print("Couldn't found split from {}".format(path))
    return ""

def get_lang(path):
    return path[-2:]

def sentence_generator(path):
    for line in open(path, "r"):
        yield line

def iterate_data(args):
    for dirpath, _, filenames in os.walk(args.in_path):
        if args.type == "mono" or (args.type == "parallel" and args.required_subdir != None and args.required_subdir in dirpath):
            for filename in filenames:
                path = os.path.join(dirpath, filename)
                domain = get_domain(path)
                split = get_split(path)
                lang = get_lang(path)

                os.makedirs(args.out_path, exist_ok=True)

                if split:
                    out_file = ".".join([args.type, domain, split, lang])
                else:
                    out_file = ".".join([args.type, domain, lang])
                    
                out_file = os.path.join(args.out_path, out_file)

                print("Iterating {} and writing to {}.".format(path, out_file))
                with open(out_file, "a") as f:
                    for sentence in sentence_generator(path):
                        f.write(sentence)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""This script concatenates various monolingual and parallel data files by split, domain, and language, in order to make it easier to use this data for domain detection task. This script assumes that file path indicates the domain, language and split of the data (split is not required). Concatenated files follow naming convention <type>.<domain>.<split>.<lang>. It doesn't do any shuffling of the data.""")

    parser.add_argument("--in_path", type=str, help="Path to directory containing the data files (can be in subdirectories).")
    parser.add_argument("--out_path", type=str, help="Directory where concatenated files are saved.")
    parser.add_argument("--type", type=str, help="Must be either \"parallel\" or \"mono\"; used for naming files.")
    parser.add_argument("--required_subdir", type=str, default="v2", help="In case there are multiple versions of parallel data available, use this to specify from which subdirectory it should be taken.")


    args = parser.parse_args()

    print("subdir")
    print(args.required_subdir)

    assert args.type == "parallel" or args.type == "mono"
    
    iterate_data(args)