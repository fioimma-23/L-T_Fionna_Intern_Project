import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

MODEL_NAME = "/home/llm-dev/models/flan-t5-base"    
DATA_PATH = "alpaca_qa_ollama.jsonl"
OUTPUT_DIR = "flan-t5-finetuned"
MAX_SOURCE_LENGTH = 256
MAX_TARGET_LENGTH = 256

dataset = load_dataset("json", data_files=DATA_PATH, split="train")

def preprocess(example):
    if example["input"]:
        prompt = f"{example['instruction']}\n{example['input']}"
    else:
        prompt = example["instruction"]
    return {
        "prompt": prompt,
        "target": example["output"]
    }

dataset = dataset.map(preprocess)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def tokenize(batch):
    model_inputs = tokenizer(
        batch["prompt"],
        max_length=MAX_SOURCE_LENGTH,
        padding="max_length",
        truncation=True,
    )
    labels = tokenizer(
        batch["target"],
        max_length=MAX_TARGET_LENGTH,
        padding="max_length",
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(tokenize, batched=True)

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    num_train_epochs=5,
    learning_rate=5e-5,
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    fp16=True,  
    report_to="none"
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)