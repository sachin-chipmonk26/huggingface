# from transformers import LlamaForCausalLM, LlamaTokenizer, TextDataset, DataCollatorForLanguageModeling

import huggingface_hub

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
# from transformers import TextDataset

huggingface_hub.login(token="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")

#for llama model
# tokenizer = LlamaTokenizer.from_pretrained("facebook/opt-125m")
# model = LlamaForCausalLM.from_pretrained("facebook/opt-125m")

#for gpt model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# **Set padding token (example using eos_token):**
tokenizer.pad_token = tokenizer.eos_token

#for CSV dataset
# train_dataset = TextDataset(
#     tokenizer=tokenizer,
#     file_path="bookings.csv",
#     block_size=128  # Adjust based on model and hardware
# )

#for json dataset
train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="bookings.json",
    block_size=128  # Adjust as needed
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False # Enable padding
)
#training code
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

#If you want to reuse the fine-tuned model later, save it to a directory using:

# model.save_pretrained("path/to/save/model")
# tokenizer.save_pretrained("path/to/save/model")
