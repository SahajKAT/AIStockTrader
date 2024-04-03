# Import necessary libraries from the Hugging Face `transformers` package and PyTorch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
# Import the Tuple class from the typing module for type hinting (not used in this snippet)
from typing import Tuple

# Determine the computing device based on the availability of a GPU (CUDA) or defaulting to CPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Initialize the tokenizer with the pretrained 'finbert' model from ProsusAI, for tokenizing input text
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
# Initialize the model for sequence classification, also using the 'finbert' model, and move it to the determined device
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
# Define the labels for classification outcomes in the order that the model outputs
labels = ["positive", "negative", "neutral"]

# Define a function to estimate the sentiment of a given news text
def estimate_sentiment(news):
    # Check if the input news text is not empty
    if news:
        # Tokenize the input text, adjust for model input requirements, and move to the correct device
        tokens = tokenizer(news, return_tensors="pt", padding=True).to(device)

        # Pass tokenized input through the model to get raw output logits for each class
        result = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])["logits"]
        # Sum the logits across the batch, apply softmax to get probabilities for each class
        result = torch.nn.functional.softmax(torch.sum(result, 0), dim=-1)
        # Find the probability of the most likely sentiment
        probability = result[torch.argmax(result)]
        # Determine the sentiment label by finding the index with the highest probability
        sentiment = labels[torch.argmax(result)]
        # Return both the probability and sentiment label of the most likely sentiment
        return probability, sentiment
    else: 
        # If the input is empty, return 0 and the 'neutral' sentiment as defaults
        return 0, labels[-1]

# Main execution block to run if the script is executed directly
if __name__ == "__main__":
    # Call the estimate_sentiment function with a list of news texts (this should ideally be a single string)
    tensor, sentiment = estimate_sentiment(['markets responded negatively to the news!','traders were displeased!'])
    # Print the probability and sentiment of the analysis
    print(tensor, sentiment)
    # Print whether a CUDA-compatible GPU is available in the system
    print(torch.cuda.is_available())
