# retriever.py
def get_retriever(vector_index):
    return vector_index.as_retriever(search_type="mmr", search_kwargs={"k": 3})

def merge_docs(retrieved_docs):
    return "###\n\n".join([d.page_content for d in retrieved_docs])
