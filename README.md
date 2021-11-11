# domain-detection-scripts

### Usage

Run `main.sh`. It collects necessary data, tokenizes it, and trains the model by calling `02_cut_and_tokenize.py` and `03_train.py`.

The script assumes that the data is located in text files, one domain and one language per file and one sentence per line. Files from monolingual corpus and parallel corpus should be in separate folders. Filenames should indicate to which domain (`general|crisis|legal|military`) and language (`et|en|ru|de`) its content belongs to. In case the data is from parallel corpus, split (`train|valid|test`) should also be specified. For example, you can use following folder structure and file naming:

    data
    |———parallel_data
    |       parallel.crisis.train.ru
    |       parallel.crisis.valid.ru
    |       parallel.crisis.test.ru
    |       parallel.legal.train.ru
    |       ...
    |———mono_data
    |       mono.crisis.ru
    |       mono.legal.ru
    |       ...
    main.sh

If you don't have such files prepared, you can use `01_concatenate_files.py` to generate the files from parallel and monolingual data folders. 


### The packages used to run the script were following:

`transformers==4.9.2`
`datasets=1.11.0`
`jsonlines==2.0.0`



