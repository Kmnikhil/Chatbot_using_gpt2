#  Retrieval-Augmented Chatbot

## Project Overview

This project is a context-aware chatbot that combines Sentence Transformers for semantic search and GPT-2 for natural language generation. 
The chatbot is built to handle casual conversations and respond with relevant, coherent answers by retrieving contextual sentences from a
provided dataset. Specifically, the all-MiniLM-L6-v2 model is used to find the most relevant sentences in the dataset, which are then 
passed as context to the GPT-2 model for response generation. The project utilizes Streamlit for an interactive user interface, enabling
users to ask questions and receive answers in real-time. Designed for efficiency, the bot balances coherence and diversity using parameters
like temperature, top-k, and repetition penalty. This project can serve as a foundation for building conversational AI systems with contextual understanding.

## Key Features:
1. **Sentence similarity search using Sentence Transformers (all-MiniLM-L6-v2).**
2. **Response generation using GPT-2 with a fine-tuned prompt mechanism.**
3. **Real-time interaction powered by Streamlit.**
4. **Configurable hyperparameters for optimal performance.**
5. **Python 3.8+**
6. **Libraries: torch, transformers, sentence-transformers, streamlit**

## Result 
![Screenshot 2024-12-13 202143](https://github.com/user-attachments/assets/7bced448-88a3-4b5f-970d-f55d2d17734e)
