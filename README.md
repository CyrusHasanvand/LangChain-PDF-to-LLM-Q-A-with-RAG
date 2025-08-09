# LangChain PDF-to-LLM Q-A with RAG
A simple Retrieval-Augmented Generation (RAG) pipeline that allows you to upload PDF documents, then save it into small chunks in a database such as Chroma, FAISS, or MongoDB. When a user asks questions about their content, the model will respond to the user based on the similarity measure of the user's content and relevant information.
The system uses LangChain, HuggingFace Embeddings, and a database to store, search, and retrieve relevant document chunks for context-aware answers.

# Code illustration
To construct our model, we have to follow the following sections

### -> Inserting packages
### -> Load PDF(s)
### -> Split PDF information into small chunks
### -> Create embedding to vectorize date before saving
### -> Store in a database such as Chroma/FAISS/MongoDB
### -> Take the user question
### -> Find the three most pertinent chunks to the user's question 
### -> Ask LLM to rank three chunks
### -> Answer the question based on the nearest choice (ranked first)



# Codes will be added as soon as possible
