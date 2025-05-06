from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.schema import Document
import os

# Load model and tokenizer
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")

llama_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    repetition_penalty=1.2
)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Function to load and split any text file
def load_and_split(filepath):
    if not os.path.exists(filepath):
        return []
    loader = TextLoader(filepath)
    docs = loader.load()
    split_docs = splitter.split_documents(docs)
    return split_docs

# Load persona and existing chat history
persona_docs = load_and_split("persona.txt")
history_docs = load_and_split("chat_history.txt")

# Combine persona and history for context
all_docs = persona_docs + history_docs

# Create FAISS vector database
db = FAISS.from_documents(all_docs, embedding_model)

# Wrap LLaMA pipeline as LangChain LLM
llm = HuggingFacePipeline(pipeline=llama_pipeline)

# Build RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    chain_type="stuff"
)

# Chat history file
history_file = "chat_history.txt"

# Chat loop
print("Type 'exit' or 'quit' to end the chat.\n")

while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    response = qa_chain.run(query)
    response = response.split("Helpful Answer:")[-1].strip()
    print(f"Bot: {response}")

    # Append query and response to chat history file
    with open(history_file, "a", encoding="utf-8") as f:
        f.write(f"You: {query}\nBot: {response}\n\n")
