from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from chatsearch import ChatSearch
import streamlit as st

# Load GPT-2 for response generation
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Handle padding
model = GPT2LMHeadModel.from_pretrained("gpt2")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# create function for gpt response
def generate_response(query, searcher, model, tokenizer, device):

    # Perform search to get top relevant sentences
    relevant_contexts = searcher.search(query, top_k=2)
    context = " ".join(relevant_contexts)

    # Combine query and context for input to LLM
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    
    # Generate response
    output = model.generate(inputs["input_ids"],
                            max_new_tokens=50,  # Limit the length of the generated response
                            num_return_sequences=1,  # Ensure only one response is returned
                            temperature=0.7,  # Balance randomness and coherence
                            top_k=50,  # Consider top 50 tokens for sampling
                            top_p=0.9,  # Consider tokens with a cumulative probability of 0.9
                            repetition_penalty=1.2,  # Penalize repetitive tokens
                            no_repeat_ngram_size=3  # Prevent repetition of n-grams
                            )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.split("Answer:")[-1].strip()

# Streamlit UI
st.title("Chatbot with Me Buddy :)")
st.write("Ask your question, and the chatbot will respond based on relevant context!")

# User input
user_input = st.text_input("You: ", placeholder="Type your question here...")
#     # Initialize searcher with dataset
dataset_path = "D:\Works\LLM_projects\project1\your_dataset.json"  
searcher = ChatSearch(dataset_path)
if user_input:
    with st.spinner("Generating response..."):
        bot_response = generate_response(user_input, searcher, model, tokenizer, device)
    st.text_area("Bot:", bot_response, height=200)

# Footer
st.write("\nPowered by Sentence Transformers and GPT-2.")

# ------------------------
# check the funtions are working
# if __name__ == "__main__":
#     # Initialize searcher with dataset
#     dataset_path = "D:\Works\LLM_projects\project1\your_dataset.json"  
#     searcher = ChatSearch(dataset_path)

#     print("Chatbot is ready! Type 'exit' to stop.")
#     while True:
#         user_query = input("You: ")
#         if user_query.lower() == "exit":
#             print("Goodbye!")
#             break
        
#         response = generate_response(user_query, searcher, model, tokenizer, device)
#         print(f"Bot: {response}")
