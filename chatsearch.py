import json
from sentence_transformers import SentenceTransformer,util
import torch

# Load Sentence Transformer model for similarity search
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

class ChatSearch:
    def __init__(self, dataset_path):
        # Load the JSON data
        with open(dataset_path, 'r') as f:
            self.data = json.load(f)
        
        # Ensure each entry is processed correctly
        self.sentences = []
        for dialog_entry in self.data:
            # Ensure dialog is properly accessed as a list of sentences
            if isinstance(dialog_entry['dialog'], list):
                self.sentences.extend(dialog_entry['dialog'])  # Add sentences to the list

        # Now, assuming you have a model for embedding sentences, proceed
        self.embeddings = embedding_model.encode(self.sentences, convert_to_tensor=True)
    
    def search(self, query, top_k=1):
        # Compute the query embedding and calculate similarity with sentences
        query_embedding = embedding_model.encode(query, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(query_embedding, self.embeddings)[0]
        # Get indices of top-k most similar sentences
        top_results = torch.topk(similarities, k=top_k)
        return [self.sentences[i] for i in top_results.indices]