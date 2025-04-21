from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from dataloader import load_dataset
from config import get_args

args = get_args()


model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)


#change the path to your dataset
dataset = load_dataset(args.train_dataset)
dataset = dataset.train_test_split(test_size=args.test_size, shuffle=True, seed=args.seed)

def tokenize_function(batch):
    inputs = ["multi-intent: " + x for x in batch["input"]]
    targets = batch["output"]
    # Add the prefix to the targets
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    num_train_epochs=args.epochs,
    logging_dir=args.logging_dir,
    save_strategy=args.save_strategy,
    evaluation_strategy=args.evaluation_strategy,
    warmup_steps=args.warmup_steps,
    logging_steps=args.logging_steps,
    fp16=args.fp16,
    report_to=args.report_to,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
)

trainer.train()
