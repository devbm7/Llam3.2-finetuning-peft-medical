import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def load_and_process_data():
    dataset = load_dataset("Amod/mental_health_counseling_conversations")
    
    def format_conversation(examples):
        texts = [
            f"### Instruction: Act as a mental health counselor. Respond to the following message:\n\n{context}\n\n### Response: {response}"
            for context, response in zip(examples['Context'], examples['Response'])
        ]
        return {"text": texts}
    
    processed_dataset = dataset.map(
        format_conversation, 
        batched=True,
        remove_columns=dataset["train"].column_names,
        batch_size=64
    )
    return processed_dataset

def tokenize_data(dataset, tokenizer):
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=256,
            padding="max_length"
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        remove_columns=["text"],
        batched=True,
        batch_size=64
    )
    return tokenized_dataset

def main():
    model_name = "meta-llama/Llama-3.2-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    training_args = TrainingArguments(
        output_dir="./llama3-mental-health-counselor",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        warmup_steps=50,
        logging_steps=50,
        learning_rate=2e-5,
        weight_decay=0.01,
        fp16=True,
        logging_dir="./logs",
        save_strategy="epoch",
        save_total_limit=1,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        max_grad_norm=0.3,
        remove_unused_columns=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        device_map={"": 0},
        torch_dtype=torch.float16,
    )
    
    model.gradient_checkpointing_enable()
    
    dataset = load_and_process_data()
    tokenized_dataset = tokenize_data(dataset, tokenizer)
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    torch.cuda.empty_cache()
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator,
    )
    
    trainer.train()
    trainer.save_model("./llama3-mental-health-counselor-final")
    tokenizer.save_pretrained("./llama3-mental-health-counselor-final")

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main()