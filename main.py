# main.py
import os
import gradio as gr
from vector_store import VectorStoreManager
from llm import get_llm, get_prompt_template
from retriever import get_retriever, merge_docs
from chat_history import ChatHistory
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from dotenv import load_dotenv

def main():
    # Load the API key from environment variables
    load_dotenv()
    api_key = os.getenv('API_KEY')

    # Initialize vector store manager and load documents
    vector_store_manager = VectorStoreManager(api_key=api_key)
    documents = vector_store_manager.load_documents()
    vector_index = vector_store_manager.get_vector_index(documents)

    # Initialize LLM and retriever
    llm = get_llm(api_key)
    retriever = get_retriever(vector_index)
    prompt_template = get_prompt_template()
    chat_history = ChatHistory()

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

    # Build the chain
    def chat_with_bot(query, history):
        # Process the query and get the response
        result = chain.invoke(query)

        # Update the memory with the new interaction
        chat_history.save_interaction(query, result['answer'])

        # Add the new interaction to the display history (without labels)
        history.append((f'<div style="text-align: right;"><strong style="color: #eee;">상담자</strong><br><span style="background-color: #DCF8C6; color: black; padding: 8px 12px; border-radius: 10px; display: inline-block;">{query}</span></div>', None))
        history.append((None, f'<div style="text-align: left;"><strong style="color: #eee;">오은영 박사</strong><br><span style="background-color: #FFFFFF; color: black; padding: 8px 12px; border-radius: 10px; display: inline-block;">{result["answer"]}</span></div>'))

        return "", history

    # Gradio interface setup
    with gr.Blocks() as iface:
        gr.Markdown("<h1 style='text-align: center;'>오은영 박사님 상담챗봇</h1>")
        chatbot = gr.Chatbot()

        with gr.Row():
            with gr.Column(scale=8):
                txt = gr.Textbox(
                    show_label=False,
                    placeholder="Ask a question...",
                    lines=1
                )
            with gr.Column(scale=1):
                submit_btn = gr.Button("Send")

        # Clear the textbox after submitting and display conversation
        txt.submit(chat_with_bot, [txt, chatbot], [txt, chatbot])
        submit_btn.click(chat_with_bot, [txt, chatbot], [txt, chatbot])

    iface.launch()

if __name__ == "__main__":
    main()
