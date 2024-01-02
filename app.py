import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel,BertTokenizer,BertForMaskedLM  # Import GPT2 classes
# from transformers import AutoTokenizer
# from transformers import TFAutoModelForSequenceClassification
# from tensorflow.keras.optimizers import Adam

model = GPT2LMHeadModel.from_pretrained("fine-tuned-gpt2")  # Load fine-tuned model - reponame  - sachin26/autotrain
tokenizer = GPT2Tokenizer.from_pretrained("fine-tuned-gpt2") 

# model = BertForMaskedLM.from_pretrained("fine-tuned-gpt2")  # Load fine-tuned model - reponame  - sachin26/autotrain
# tokenizer = BertTokenizer.from_pretrained("fine-tuned-gpt2") 


# tokenizer = AutoTokenizer.from_pretrained("fine-tuned-gpt2")
# model = TFAutoModelForSequenceClassification.from_pretrained("fine-tuned-gpt2")
# model.compile(optimizer=Adam(3e-5))

  # Load fine-tuned tokenizer

def generate_response(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=128)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print("@@@",response)
    return response 

st.title("Movie Booking Chatbot")
user_input = st.text_input("Ask a question about movie bookings:")
if user_input:
    response = generate_response(user_input)
    print("###",response)
    st.write("Response:", response)
