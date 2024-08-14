from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from operator import itemgetter
from retriever import merge_docs  # merge_docs 함수를 임포트
from retriever import get_retriever  # 추가로 retriever를 가져오는 코드도 필요할 수 있음

# 대화 메모리 객체 생성
memory = ConversationBufferWindowMemory(k=10)

# 프롬프트 템플릿 설정
def create_prompt():
    template = """
    당신은 상담가로서, "아이"라는 단어 대신 "금쪽"이라는 단어를 사용하여 대화를 진행해야 합니다. 대화 내에서 "아이"라는 단어가 나와야 할 때는 항상 "금쪽"으로 바꾸어 표현하세요.
    예시:
    사용자: 제 아이가 요즘 많이 울어요.
    챗봇: 금쪽이가 요즘 많이 울고 있군요. 무슨 일이 있었나요?
    사용자: 제 아이가 새로운 친구를 사귀었어요.
    챗봇: 금쪽이가 새로운 친구를 사귀었군요! 정말 기쁜 소식이네요.
    위와 같이 대화를 진행하세요.
    이전 대화내용을 참고해서 대화를 지속하세요. 주어진 문맥과 이전 대화내용에서 알 수 없는 내용에 대해서는 절대로 추측해서 답변하지 말고 모른다고 하세요.
    ---
    주어진 문맥: {context}

    이전 대화내용: {history}

    상담자: {query}
    오은영 박사:
    """
    return ChatPromptTemplate.from_template(template)

# 체인 생성 함수
def create_chain(llm, retriever):
    prompt = create_prompt()
    return RunnableParallel({
        "context": retriever | merge_docs,  # merge_docs가 이제 올바르게 사용됩니다.
        "query": RunnablePassthrough(),
        "history": RunnableLambda(memory.load_memory_variables) | itemgetter('history')
    }) | {
        "answer": prompt | llm | StrOutputParser(),
        "context": itemgetter("context"),
        "prompt": prompt
    }

# 대화 히스토리 저장 및 포맷팅
def save_and_format_history(query, result):
    # 새로운 상호작용을 메모리에 저장합니다.
    memory.save_context({'query': query}, {'answer': result['answer']})

    # 전체 대화 히스토리를 가져옵니다.
    full_history = memory.load_memory_variables({})['history']

    # 히스토리 포맷을 '상담자'와 '오은영 박사'로 변경하여 출력
    formatted_history = full_history.replace("Human:", "\n상담자:").replace("AI:", "\n오은영 박사:")

    return formatted_history