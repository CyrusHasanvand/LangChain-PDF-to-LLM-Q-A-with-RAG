# LangChain PDF-to-LLM Q-A with-RAG
A simple Retrieval-Augmented Generation (RAG) pipeline that allows you to upload PDF documents, then save it into small chunks in a database such as Chroma, FAISS, or MongoDB. When a user asks questions about their content, the model will respond to the user based on the similarity measure of the user's content and relevant information.
The system uses LangChain, HuggingFace Embeddings, and a database to store, search, and retrieve relevant document chunks for context-aware answers.

