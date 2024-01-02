import huggingface_hub
import pandas as pd  # Only for splitting CSV dataset (if used)
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    TextDataset,
    DataCollatorForLanguageModeling,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    LineByLineTextDataset,
    Trainer,
    TrainingArguments,
)
import tempfile  # Import for temporary file creation

huggingface_hub.login(token="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")

# Choose model and tokenizer (uncomment either Llama or GPT)
# tokenizer = LlamaTokenizer.from_pretrained("facebook/opt-125m")
# model = LlamaForCausalLM.from_pretrained("facebook/opt-125m")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Set padding token
tokenizer.pad_token = tokenizer.eos_token

# Read text from TXT file
with open("MovieBooking.txt", "r") as file:
    text1 = file.read()

# Optionally preprocess text (e.g., clean, remove extra spaces)

# Create temporary file
with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
    temp_file.write(text1)

# Create dataset using TextDataset with the temporary file
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=temp_file.name,  # Pass the temporary file path
    block_size=128
)

# Rest of your code remains the same
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False  # Enable padding
)

# Training code
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=3,  # Adjust based on dataset size and needs
    per_device_train_batch_size=4,  # Adjust based on available memory
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

trainer.train()

trainer.save_model("./fine-tuned-gpt2")
tokenizer.save_pretrained("./fine-tuned-gpt2")
tokenizer.save_vocabulary("vocab.json")
