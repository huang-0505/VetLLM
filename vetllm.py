import os
import torch
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling,
    pipeline
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from datasets import Dataset

# Hugging Face login
token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
if token:
    login(token=token)
else:
    print("⚠️ No Hugging Face token found in environment — attempting interactive login...")
    login()

# Load your text data
with open('combined.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
lines = [line.strip() for line in lines if line.strip()]
data_dict = {"text": lines}
dataset = Dataset.from_dict(data_dict)
print(dataset)

# Model quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16"
)

# Load model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Apply LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, lora_config)

# Tokenization
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
total_tokens = sum(len(x) for x in tokenized_dataset['input_ids'])
print(f"Total tokens: {total_tokens:,}")

# Train/test split
split = tokenized_dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=64,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    bf16=True,
    report_to="none",
    evaluation_strategy="steps",
    eval_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_strategy="steps",             
    logging_dir="./logs", 
    resume_from_checkpoint=True
)

# Trainer setup
early_stopping = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    callbacks=[early_stopping],
)

# Start training
trainer.train()
# === Step 1: Save LoRA adapter only ===
model.save_pretrained("./finetuned-mistral7b-vet-lora")
tokenizer.save_pretrained("./finetuned-mistral7b-vet-lora")

# === Step 2: Merge LoRA into base model ===
print("Merging LoRA weights into base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    device_map="auto",
    torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(base_model, "./finetuned-mistral7b-vet-lora")
merged_model = model.merge_and_unload()

# Save merged full model
print("Saving merged model...")
merged_model.save_pretrained("./finetuned-mistral7b-vet-merged")
tokenizer.save_pretrained("./finetuned-mistral7b-vet-merged")

# === Step 3: Reload and test generation ===
print("Reloading merged model for testing...")
model = AutoModelForCausalLM.from_pretrained(
    "./finetuned-mistral7b-vet-merged",
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained("./finetuned-mistral7b-vet-merged")

print("Running test prompt...")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
output = generator("What is Papillomaviral skin diseases?", max_new_tokens=250)
print(output[0]['generated_text'])