import huggingface_hub
import pandas as pd
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

huggingface_hub.login(token="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")

# Choose model and tokenizer (uncomment either Llama or GPT)
# tokenizer = LlamaTokenizer.from_pretrained("facebook/opt-125m")
# model = LlamaForCausalLM.from_pretrained("facebook/opt-125m")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Set padding token
tokenizer.pad_token = tokenizer.eos_token

# Split dataset
df = pd.read_csv("movie_ticket_booking.csv")
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

# Create datasets
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="train_data.csv",
    block_size=128
)
test_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="test_data.csv",
    block_size=128
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
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

# Save model and tokenizer
trainer.save_model("./fine-tuned-gpt2")
tokenizer.save_pretrained("./fine-tuned-gpt2")
tokenizer.save_vocabulary("vocab.json")
