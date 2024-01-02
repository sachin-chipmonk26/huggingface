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
import docx  # For DOCX handling

huggingface_hub.login(token="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")

# Choose model and tokenizer (uncomment either Llama or GPT)
# tokenizer = LlamaTokenizer.from_pretrained("facebook/opt-125m")
# model = LlamaForCausalLM.from_pretrained("facebook/opt-125m")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Set padding token
tokenizer.pad_token = tokenizer.eos_token

# Read text from DOCX file
doc = docx.Document("MovieBooking.docx")
text = '\n'.join([para.text for para in doc.paragraphs])

# Optionally preprocess text (e.g., clean, remove extra spaces)

# Create dataset
train_dataset = TextDataset(
    tokenizer=tokenizer,
    text=text,  # Pass the extracted text (singular)
    block_size=128
)


#for json dataset
# train_dataset = LineByLineTextDataset(
#     tokenizer=tokenizer,
#     file_path="bookings.json", #add any json file that is used for training
#     block_size=128  # Adjust as needed
# )

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