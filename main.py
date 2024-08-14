# main.py
import os
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from vector_store import VectorStoreManager
from llm import get_llm

def main():
    # Load the API key from environment variables
    api_key = os.getenv('UPSTAGE_API_KEY')

    # Initialize vector store manager and load documents
    vector_store_manager = VectorStoreManager(api_key=api_key)
    documents = vector_store_manager.load_documents()
    vector_index = vector_store_manager.get_vector_index(documents)

    # Initialize the LLM (Large Language Model) and retriever
    llm = get_llm(api_key)
    retriever =YourRetrieverClass(vector_index=vector_index)

    # Set up the prompt template
    prompt_template = get_prompt_template()
    # Initialize the chat history
    chat_history = []

    # Build the chain
    def build_chain():
        return RunnableParallel({
            "context": retriever | merge_docs,
            "query": RunnablePassthrough(),
            "history": RunnableLambda(chat_history.memory.load_memory_variables) | itemgetter('history')
        }) | {
            "answer": prompt_template | llm | StrOutputParser(),
            "context": itemgetter("context"),
            "prompt": prompt_template
        }

    chain = build_chain()
    # Implement chat interface
if __name__ == "__main__":
    main()
