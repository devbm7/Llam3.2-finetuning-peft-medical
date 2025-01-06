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
    model_name = "meta-llama/Llama-3.2-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Training arguments - specify device_map here
    training_args = TrainingArguments(
        output_dir="./llama3-mental-health-counselor",
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
        # Remove device mapping from training arguments
        no_cuda=False
    )
    
    # Load model with device mapping after training args are set
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        # Use specific device mapping instead of "auto"
        device_map={"": 0}  # Maps all layers to first GPU
    )
    
    # Load and process dataset
    dataset = load_and_process_data()
    tokenized_dataset = tokenize_data(dataset, tokenizer)
    
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
    trainer.save_model("./llama3-mental-health-counselor-final")
    tokenizer.save_pretrained("./llama3-mental-health-counselor-final")

if __name__ == "__main__":
    main()