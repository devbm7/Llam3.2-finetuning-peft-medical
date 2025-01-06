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
import os

# Set environment variables for memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def load_and_process_data():
    """Load and process the mental health counseling dataset."""
    dataset = load_dataset("Amod/mental_health_counseling_conversations")
    
    def format_conversation(example):
        return {
            "text": f"### Instruction: Act as a mental health counselor. Respond to the following message:\n\n{example['Context']}\n\n### Response: {example['Response']}"
        }
    
    processed_dataset = dataset.map(
        format_conversation, 
        remove_columns=dataset["train"].column_names,
        batched=True
    )
    return processed_dataset

def tokenize_data(dataset, tokenizer):
    """Tokenize the dataset using the provided tokenizer."""
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=256,  # Reduced from 512
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
    # Initialize model and tokenizer
    model_name = "meta-llama/Llama-3.2-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Training arguments with memory optimizations
    training_args = TrainingArguments(
        output_dir="./llama3-mental-health-counselor",
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Reduced batch size
        gradient_accumulation_steps=16,  # Increased gradient accumulation
        warmup_steps=50,
        logging_steps=50,
        learning_rate=2e-5,
        weight_decay=0.01,
        fp16=True,
        logging_dir="./logs",
        save_strategy="epoch",
        save_total_limit=1,  # Reduced number of checkpoints
        gradient_checkpointing=True,  # Enable gradient checkpointing
        optim="adamw_torch_fused",  # Use fused optimizer
        max_grad_norm=0.3,
        remove_unused_columns=True,
    )
    
    # Load model with 8-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,  # Enable 8-bit quantization
        device_map={"": 0},
        torch_dtype=torch.float16,
    )
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
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
    
    # Clear CUDA cache before training
    torch.cuda.empty_cache()
    
    # Start training
    trainer.train()
    
    # Save the model
    trainer.save_model("./llama3-mental-health-counselor-final")
    tokenizer.save_pretrained("./llama3-mental-health-counselor-final")

if __name__ == "__main__":
    # Set memory efficient options
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    main()