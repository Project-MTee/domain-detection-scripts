# domain-detection-scripts

### Usage

Run `main.sh`. It collects the data, tokenizes it, and trains the model by calling `02_cut_and_tokenize.py` and `03_train.py`.

The script assumes that the data is located in text files, one sentence per line. Filenames should indicate to which domain (`general|crisis|legal|military`), split (`train|valid|test`), and language (`et|en|ru|de`) its content belongs to, for example `crisis.test.ru`.

### The packages used to run the script were following:

    transformers==4.9.2
    datasets=1.11.0
    jsonlines==2.0.0

