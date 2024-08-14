# chat_history.py
from langchain.memory import ConversationBufferWindowMemory

class ChatHistory:
    def __init__(self, k=10):
        self.memory = ConversationBufferWindowMemory(k=k)

    def save_interaction(self, query, answer):
        self.memory.save_context({'query': query}, {'answer': answer})

    def get_full_history(self):
        return self.memory.load_memory_variables({})['history']

    def format_history(self, full_history):
        return full_history.replace("Human:", "\n상담자:").replace("AI:", "\n오은영 박사:")
