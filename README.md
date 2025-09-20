# LangChain PDF-to-LLM Q-A with RAG
A simple Retrieval-Augmented Generation (RAG) pipeline that allows you to upload PDF documents, then save it into small chunks in a database such as Chroma, FAISS, or MongoDB. When a user asks questions about their content, the model will respond to the user based on the similarity measure of the user's content and relevant information.
The system uses LangChain, HuggingFace Embeddings, and a database to store, search, and retrieve relevant document chunks for context-aware answers.

## Packages
First, we need to import information from a PDF. Therefore, we have to load it at the first step as follows:
```python
Data_Path = r"D:\WorkPlace\Python\Training\May2025\Practice\PracLLM"
FAISS_Path = r"D:\WorkPlace\Python\Training\May2025\Practice\PracLLM\FAISS_DB\UFAREX"
```
then we can load **all PDF files** of the ```Data_Path``` as follows:
```python
loader = PyPDFDirectoryLoader(Data_Path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "]
)
chunks = text_splitter.split_documents(documents)
```
In this text splitter, ```chunk size``` sets the maximum threshold to define a chunk; on the other hand, ```chunk overlap``` allows a sequential of chunks to have a maximum similar data to help chunks have a deliverable meaning when they are requested by a similarity measure. In this situation, we don't lose the meaning behind the texts. In addition, separators help  
## Embedding and FIASS Vector store
After splitting the information, we use an embedding tool such as ```Sentence Transformer``` to vectorize the information as follows:
```python
embedding_model = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")

# Create FAISS vector store
vectorstore = FAISS.from_documents(chunks, embedding=embedding_model)
```
Now, we create our vectorstore. Then, we need to save this database.

## Save the FAISS
In comparison to ChromaDB, which can save the database automatically, we need to save FAISS database manually. Therefore, we have

```python
if os.path.exists(FAISS_Path):
    shutil.rmtree(FAISS_Path)       # Delete the folder and its contents
os.makedirs(FAISS_Path, exist_ok=True) # Re-create the folder

# Save index
index_path = os.path.join(FAISS_Path, "index.faiss")
faiss.write_index(vectorstore.index, index_path)

# Save documents and embeddings separately
store_path = os.path.join(FAISS_Path, "store.pkl")
with open(store_path, "wb") as f:
    pickle.dump(vectorstore, f)

print(f"Saved {len(chunks)} chunks to {FAISS_Path}")
```
Now, the database is created, and we can simply search through this database to find the relevant information by similarity search.

## Similarity Search in FAISS Database
When we created our database, which includes several chunks, we can search through the database by asking our question.
To do this, we have
```python
Query="definition of nonstationary data streams (NDS)"
results = vectorstore.similarity_search(Query, k=5)
```



















