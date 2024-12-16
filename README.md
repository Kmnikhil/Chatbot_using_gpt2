This project is a context-aware chatbot that combines Sentence Transformers for semantic search and GPT-2 for natural language generation. 
The chatbot is built to handle casual conversations and respond with relevant, coherent answers by retrieving contextual sentences from a
provided dataset. Specifically, the all-MiniLM-L6-v2 model is used to find the most relevant sentences in the dataset, which are then 
passed as context to the GPT-2 model for response generation. The project utilizes Streamlit for an interactive user interface, enabling
users to ask questions and receive answers in real-time. Designed for efficiency, the bot balances coherence and diversity using parameters
like temperature, top-k, and repetition penalty. This project can serve as a foundation for building conversational AI systems with contextual understanding.

Key Features:
*Sentence similarity search using Sentence Transformers (all-MiniLM-L6-v2).
*Response generation using GPT-2 with a fine-tuned prompt mechanism.
*Real-time interaction powered by Streamlit.
*Configurable hyperparameters for optimal performance.
*Prerequisites:
*Python 3.8+
*Libraries: torch, transformers, sentence-transformers, streamlit
