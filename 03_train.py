from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import argparse
import os

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall}

def train(args):
    from transformers import XLMRobertaForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
    from datasets import load_dataset

    dataset = load_dataset("json", data_files= {"valid": args.valid_file, "train": args.train_file}, cache_dir=args.cache)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    model = XLMRobertaForSequenceClassification.from_pretrained(args.model, num_labels=4)
    model.to("cuda")
    model.train()

    training_args = TrainingArguments(
        output_dir=os.path.join(args.out_path, args.run_name),             
        num_train_epochs=args.epochs,             
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_strategy='steps',
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        load_best_model_at_end=args.keep_best_model
    )

    if args.wandb and args.wandb == True:
        import wandb
        wandb.login()
        os.environ["WANDB_PROJECT"] = "Domain_detection"
        training_args.report_to = ["wandb"] #¯\_(ツ)_/¯
        training_args.run_name = args.run_name

        print(training_args.report_to)
        print(training_args.run_name)

    trainer = Trainer(
            model=model,                         
            args=training_args,                 
            train_dataset=dataset["train"],              
            eval_dataset=dataset["valid"],
            compute_metrics=compute_metrics,
            tokenizer=tokenizer
        )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script trains domain detection model.")

    parser.add_argument("--model", type=str, help="Name of the pretrained model in HuggingFace.")
    parser.add_argument("--tokenizer", type=str, help="Name of the tokenizer in HuggingFace.")
    parser.add_argument("--train_file", type=str, help="Path to train jsonlines file.")
    parser.add_argument("--valid_file", type=str, help="Path to validation jsonlines file.")
    parser.add_argument("--out_path", type=str, help="Directory where checkpoints are saved (in subdirectories).")
    parser.add_argument("--cache", type=str, default="cache", help="Directory where Datasets cache is saved.")
    parser.add_argument("--run_name", type=str, help="Name of the run. Will also be used as name of the directory where checkpoints are saves.")
    parser.add_argument("--wandb", type=bool, default=False, help="Whether to log to wandb.")
    
    subparsers = parser.add_subparsers()

    model_args_parser = subparsers.add_parser("train", help="Parameters required for initializing HuggingFace TrainingArguments class.")
    model_args_parser.add_argument("--epochs", type=int, default=5)
    model_args_parser.add_argument("--train_batch_size", type=int, default=16)
    model_args_parser.add_argument("--eval_batch_size", type=int, default=16)
    model_args_parser.add_argument("--learning_rate", type=float, default=5e-6)
    model_args_parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    model_args_parser.add_argument("--weight_decay", type=float, default=0.1)
    model_args_parser.add_argument("--warmup_ratio", type=float, default=0.06)
    model_args_parser.add_argument("--logging_steps", type=int, default=200)
    model_args_parser.add_argument("--save_steps", type=int, default=1000)
    model_args_parser.add_argument("--save_total_limit", type=int, default=10, help="Specifies how many checkpoints are kept in disk. Older checkpoints will be overwritten.")
    model_args_parser.add_argument("--eval_steps", type=int, default=1000)
    model_args_parser.add_argument("--keep_best_model", type=bool, default=True, help="If set True then best model will never be overwritten.")

    args = parser.parse_args()

    train(args)