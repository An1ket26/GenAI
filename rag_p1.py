from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda,
    RunnableMap,
)
from langchain.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.environ["GOOGLE_API_KEY"]

parser = StrOutputParser()

chatHistory = []
chatHistory.append(
    {
        "role": "System",
        "content": """You are an data engineer expert for question-answering tasks.
            Use the following pieces of retrieved context to answer the question. If you don't know the answer,
            just say that you don't know,  don't try to make up an answer.""",
    }
)

historyprompt = PromptTemplate(
    template="""Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is.
Question: {original_question}
chatHistory = {chatHistory}""",
    input_variables=["original_question", "chatHistory"],
)

prompt = PromptTemplate(
    template="""You are an data engineering  expert for question-answering tasks.
            Use the following pieces of retrieved context to answer the question. If you don't know the answer,
            just say that you don't know,  don't try to make up an answer.

    Context: {context}

    Question: {question}

    Chat History = {chatHistory}

    Answer: """,
    input_variables=["context", "question", "chatHistory"],
)


def format_chat_history(history):
    return "\n".join(
        [
            (
                f"{'You' if h['role'].lower() == 'user' else 'Assistant'}: {h['content']}"
                if h["role"].lower() != "system"
                else f"System Instructions: {h['content']}"
            )
            for h in history
        ]
    )


def makecontent(docs):
    content = "\n\n".join(doc.page_content for doc in docs)
    return content


embed_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", google_api_key=api_key
)
llm_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=api_key)


vectorstore = Chroma(
    embedding_function=embed_model,
    persist_directory="./my_chroma_db",
    collection_name="DataEng",
)

retriever = vectorstore.as_retriever(search="similarity", search_kwargs={"k": 2})

multiretriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}), llm=llm_model
)


historychain = historyprompt | llm_model | parser

rewrite_chain = RunnableMap(
    {
        "reformulated_question": historychain,
        "chatHistory": lambda inputs: inputs["chatHistory"],
    }
)

retrieve_chain = RunnableParallel(
    {
        "context": multiretriever | makecontent,
        "question": lambda inputs: inputs["reformulated_question"],
        "chatHistory": lambda inputs: inputs["chatHistory"],
    }
)

llm_chain = rewrite_chain | retrieve_chain | prompt | llm_model | parser


try:
    while True:
        userinput = input("User - > Enter your Query: ")

        if userinput.lower() == "exit":
            break

        chatHistory.append({"role": "User", "content": userinput})
        result = llm_chain.invoke(
            {
                "original_question": userinput,
                "chatHistory": format_chat_history(chatHistory),
            }
        )
        print(f"Assistant : {result}")
        chatHistory.append({"role": "Assistant", "content": result})

except KeyboardInterrupt:
    print("\nExiting chat.")
