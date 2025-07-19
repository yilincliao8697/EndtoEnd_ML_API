from transformers import AutoModelForSequenceClassification, AutoTokenizer 

# Define the model name 
model_name = "winegarj/distilbert-base-uncased-finetuned-sst2"
# Load the model and tokenizer 
model = AutoModelForSequenceClassification.from_pretrained(model_name) 
tokenizer = AutoTokenizer.from_pretrained(model_name) 
# Save the model and tokenizer locally 
model.save_pretrained("./distilbert-base-uncased-finetuned-sst2/") 
tokenizer.save_pretrained("./distilbert-base-uncased-finetuned-sst2/")