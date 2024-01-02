# prerequistes
# Install all the below libraries
 ```pip install transformers```
```pip install datasets```
```pip install huggingface_hub```
```pip install streamlit```

# get your own Hugging face token from the site below

https://huggingface.co/settings/tokens

Request access to "meta-llama/Llama-2-7b" as this is a gated repo (Optional)

# in case there is a ImportError: cannot import name 'LlamaForCausalLM' from 'transformers', run the below command
pip install git+https://github.com/huggingface/transformers

# If you want to train the model with csv file then follow the below steps
## To Create a csv dataset run the below cmd
```python movie_booking_script.py```
 1.This creates a csv with name "booking_prompts.csv" which is being used by "movie_booking_script.py"
 2.uncomment line 32-36 and comment line 39-43 in "train_chatbot.py"

# If you want to train the model with json file then follow the below steps
 1.open "convertToJson.py" file
 2.add the csv file name which needs to be converted to json in csvFilePath (line:19)
  run ```python convertToJson.py```
 3.this will generate a json file(with name convert.json) which can be used in "train_chatbot.py"(line:42)
 4.comment line 32-36 and uncomment line 39-43 in "train_chatbot.py" and add the .json file in the file path 

# to fine-tune the pre-trained LLM model run the following command
Note : delete "output","fine-tuned-gpt2" folders  and "cached_lm_GPT2Tokenizer_128_booking_prompts.csv" before running the below cmd (if those files already exist). Delete them everytime before training. 
```python train_chatbot.py``` (In our case we are running it on a CPU, GPU is usually recommended)
# This process will take few moments. Please wait until all the files are successfully downloaded completely before you run the main application

# Once the above files are downloaded run the following command to chat with hyour fine tuned custom bot 

```streamlit run app.py```

# Reference links for pretrained models

https://huggingface.co/transformers/v3.0.2/pretrained_models.html

https://huggingface.co/docs/transformers/model_doc/gpt2