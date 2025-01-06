import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
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
    
    return dataset.map(
        format_conversation, 
        batched=True,
        remove_columns=dataset["train"].column_names,
        batch_size=64
    )

def tokenize_data(dataset, tokenizer):
    return dataset.map(
        lambda examples: tokenizer(
            examples["text"],
            truncation=True,
            max_length=256,
            padding="max_length"
        ),
        remove_columns=["text"],
        batched=True,
        batch_size=64
    )

def main():
    model_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        device_map={"": 0},
        torch_dtype=torch.float16,
    )

    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)

    training_args = TrainingArguments(
        output_dir="./llama3-mental-health-counselor",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        warmup_steps=50,
        learning_rate=2e-4,
        fp16=True,
        logging_dir="./logs",
        save_strategy="epoch",
        save_total_limit=1,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
    )

    dataset = load_and_process_data()
    tokenized_dataset = tokenize_data(dataset, tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    trainer.train()
    model.save_pretrained("./llama3-mental-health-counselor-final")
    tokenizer.save_pretrained("./llama3-mental-health-counselor-final")

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main()