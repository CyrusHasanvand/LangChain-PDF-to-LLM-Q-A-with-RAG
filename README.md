# LangChain PDF-to-LLM Q-A with RAG
This is a simple Retrieval-Augmented Generation (RAG) pipeline that allows you to upload PDF documents, then save them into small chunks after embedding in a database such as Chroma, FAISS, or MongoDB. When a user asks questions about their need, the query would be checked in the database to find the most relevant information with at least a number of ```k``` instances, and the model will respond to the user based on the similarity measure of the user's content and relevant information.
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
Results = vectorstore.similarity_search(Query, k=5)
```
where we search for the relevant information regarding the requested query. In this vein, we have ```k``` number of nearest meanings of the information in the database for the request query. In this example, ```Result``` gives a list of ```k=5``` members. Each list has its own document class that includes ```page_content```, ```metadata```, and so forth.

## LLM
In this section, we want to use the previous ```k=5``` responses to analyze them with an intelligent LLM (```llama3.1```).
This is because the proposed results may not be ordered in an accurate organization; therefore, an LLM can boost the results to deliver a better outcome.

So, we have to define a good ```prompt```, including the fast five result from the similarity search of FAISS database, and process the information to deliver a meaningful performance.

### Define local llama3.1 from Ollama
First, I used the following packages to call my local LLM (```llama3.1```)
```python
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage, AIMessage, StrOutputParser
from langchain_ollama import ChatOllama
```
then I define the ```LLM``` as follows:
```python
LLM = ChatOllama(model="llama3.1", Temperature=0.7,do_sample=True)
```
### Define the relevant Prompt
I've utilized the following steps to set the roles:
```python
prompt=ChatPromptTemplate.from_messages([
    ('system','You are an AI assistant, which refines the orders of similarity search from FAISS database and provides no more than 220 tokens when writing a response to a question'),
    ('user','{text}')
])
```
### Define the Chain
When we set our ```LLM``` model and its relevant ```prompt```, we can bring these information together as a ```chain``` by
```python
ChainLLama31=prompt|LLM|StrOutputParser()
```
where ```StrOutputParser()``` makes the result be expressed as a string.

### Define the task
Here, we create our question by using the previously retrieved information as follows: 
```python
Question = f'''
You are given the results of a FAISS similarity search. 
The user asked the following query:
Query: "{Query}"

Here are the retrieved results:
Result[0]: {Results[0].page_content}
Result[1]: {Results[1].page_content}
Result[2]: {Results[2].page_content}
Result[3]: {Results[3].page_content}
Result[4]: {Results[4].page_content}

Please re-order these results in order of **relevance to the user's query**,
starting with the most relevant. 
In your answer, you can refer to them by their original indices, e.g., Result[0],
Result[1], etc., and optionally explain why each result is in that position.
'''
```
This is because we have ```k=5``` question. 
### Invoke the Chain
Now, we invoke the model our question as
```python
Response_message=ChainLLama31.invoke({'text':Question})
```

The response is amazing. You can see the below explanation of ```llama3.1``` that sensationaly reorder the responses based on their relevancy.

```print(Response_message)```
```
After analyzing the results, I've reordered them based on relevance to the user's query "definition of nonstationary data streams (NDS)". Here are the results:

1. **Result[1]**: The core theme of NDS is concept drift. It results in conflict between the old and new trends... This result directly answers the user's question by defining a key aspect of NDS.
2. **Result[0]**: Online knowledge discovery from real-world non-stationary data streams (NDSs) presents profound challenges to the design of modern computational intelligence... While this result doesn't explicitly define NDS, it mentions it and discusses its challenges, making it highly relevant.
3. **Result[4]**: ...the sole dominant rule undertakes the prediction task... This result is less directly related to defining NDS, but it does discuss a concept drift within the context of NDS, which makes it somewhat relevant.
4. **Result[2]**: Thus, the sole dominant rule undertakes the prediction task... This result is similar to Result[4] and discusses the detection of recurrent drifts in NDS, but with less focus on defining the term itself.
5. **Result[3]**: C. Philosophy and Design... In quest of optimality and rule-base compactness, we hold interest in finding minimal ℜ local linear regions in NDS that can be sequentially extracted by TS rules... This result is the least directly related to defining NDS and appears to discuss a specific design aspect rather than providing a definition.

The ordering is based on how directly each result addresses the user's query. Result[1] provides a clear definition of concept drift within the context of NDS, making it the most relevant.
```
where ```llama3.1```suggests ```[second, first, fifth, third, fourth]```, highlighting similarity search would not provide an accurate decision, even though it caters a fast response.





















