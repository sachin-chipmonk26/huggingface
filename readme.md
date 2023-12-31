# prerequistes
# Install all the below libraries
 ```pip install transformers```
```pip install datasets```
```pip install huggingface_hub```
```pip install streamlit```

# get your own Hugging face token from the site below

https://huggingface.co/settings/tokens

Request access to "meta-llama/Llama-2-7b" as this is a gated repo

# in case there is a ImportError: cannot import name 'LlamaForCausalLM' from 'transformers', run the below command
pip install git+https://github.com/huggingface/transformers

# to fine-tune the pre-trained LLM model run the following command
```python train_chatbot.py``` (In our case we are running it on a CPU, GPU is usually recommended)
# This process will take few moments. Please wait until all the files are successfully downloaded completely before you run the main application

# Once the above files are downloaded run the following command to chat with hyour fine tuned custom bot 

```streamlit run app.py```