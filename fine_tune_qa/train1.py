import os
import json
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import (
    prepare_model_for_kbit_training,
    get_peft_model,
    LoraConfig
)

os.environ["TRANSFORMERS_OFFLINE"]    = "1"
os.environ["HF_DATASETS_OFFLINE"]     = "1"
os.environ["HF_HUB_OFFLINE"]          = "1"
os.environ["HF_HOME"]                 = "./hf_cache"

LOCAL_MODEL_DIR = "/home/llm-dev/models/flan-t5-base"
JSONL_FILE      = "alpaca_qa_ollama.jsonl"
OUTPUT_DIR      = "flan-t5-finetuned"
MAX_LENGTH      = 512
BATCH_SIZE      = 2
ACCUM_STEPS     = 8
EPOCHS          = 3
LR              = 3e-4

class JSONLDataset(Dataset):
    def __init__(self, path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                instr = rec.get("instruction", "").strip()
                out   = rec.get("output", "").strip()
                self.samples.append((instr, out))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        instr, out = self.samples[idx]
        enc = self.tokenizer(
            instr,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        label = self.tokenizer(
            out,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": enc.input_ids.squeeze(),
            "attention_mask": enc.attention_mask.squeeze(),
            "labels": label.input_ids.squeeze()
        }

def main():
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_MODEL_DIR,
        local_files_only=True
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        LOCAL_MODEL_DIR,
        quantization_config=bnb_cfg,
        device_map="auto",
        local_files_only=True
    )

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q", "v"],  
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    dataset = JSONLDataset(JSONL_FILE, tokenizer, MAX_LENGTH)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=ACCUM_STEPS,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        fp16=True,
        logging_steps=50,
        save_total_limit=2,
        optim="paged_adamw_32bit"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(" Training complete — LoRA model saved to", OUTPUT_DIR)

if __name__ == "__main__":
    main()
