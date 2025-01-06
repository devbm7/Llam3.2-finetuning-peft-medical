import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import numpy as np
from typing import Dict, Sequence

def load_and_process_data():
    """Load and process the mental health counseling dataset."""
    dataset = load_dataset("Amod/mental_health_counseling_conversations")
    
    # Format conversations into instruction format
    def format_conversation(example):
        return {
            "text": f"### Instruction: Act as a mental health counselor. Respond to the following message:\n\n{example['Context']}\n\n### Response: {example['Response']}"
        }
    
    processed_dataset = dataset.map(format_conversation, remove_columns=dataset["train"].column_names)
    return processed_dataset

def tokenize_data(dataset, tokenizer):
    """Tokenize the dataset using the provided tokenizer."""
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding=False
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        remove_columns=["text"],
        num_proc=4
    )
    return tokenized_dataset

def main():
    # Initialize model and tokenizer
    model_name = "meta-llama/Llama-2-3b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load and process dataset
    dataset = load_and_process_data()
    tokenized_dataset = tokenize_data(dataset, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./llama2-mental-health-counselor",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        warmup_steps=100,
        logging_steps=100,
        learning_rate=2e-5,
        weight_decay=0.01,
        fp16=True,
        logging_dir="./logs",
        save_strategy="epoch",
        save_total_limit=2,
    )
    
    # Initialize data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator,
    )
    
    # Start training
    trainer.train()
    
    # Save the model
    trainer.save_model("./llama2-mental-health-counselor-final")
    tokenizer.save_pretrained("./llama2-mental-health-counselor-final")

if __name__ == "__main__":
    main()