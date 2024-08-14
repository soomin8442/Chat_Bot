# vector_store.py
import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_upstage import UpstageEmbeddings

class VectorStoreManager:
    def __init__(self, api_key, vector_index_path="./models/chat_faiss.json"):
        self.api_key = api_key
        self.vector_index_path = vector_index_path
        self.embedding_model = UpstageEmbeddings(
            api_key=api_key,
            model="solar-embedding-1-large"
        )
        self.loader = TextLoader("merged_script.txt", encoding='utf-8')
        self.text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1024, chunk_overlap=128)

    def load_documents(self):
        documents = self.loader.load()
        return self.text_splitter.split_documents(documents)

    def get_vector_index(self, documents):
        if os.path.exists(self.vector_index_path):
            print("Loading existing vector index...")
            vector_index = FAISS.load_local(
                folder_path=self.vector_index_path,
                embeddings=self.embedding_model,
                allow_dangerous_deserialization=True
            )
        else:
            print("Creating new vector index...")
            vector_index = FAISS.from_documents(documents, self.embedding_model)
            vector_index.save_local(self.vector_index_path)
        return vector_index
