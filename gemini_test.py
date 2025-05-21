from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter,Language
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv();
api_key = os.environ["GOOGLE_API_KEY"]

loader = DirectoryLoader(
    path="Books",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,chunk_overlap=100, add_start_index=True,
)
split_docs = text_splitter.split_documents(docs)

embed_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004",google_api_key=api_key)
llm_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash",api_key=api_key)


vectorstore = Chroma(
    collection_name="DataEng",
    embedding_function=embed_model,
    persist_directory="./my_chroma_db"
)

vectorstore.add_documents(
    documents=split_docs
)

# contextres = vectorstore.similarity_search_with_relevance_scores(
#     query="what is data engineering",
#     k=2
# )



# # test = vectorstore.get(include=["metadatas"])
# res = vectorstore.similarity_search_with_relevance_scores(
#     query="what is data engineering",
#     k=2
# )

# print(res)


# vectorstore = Chroma.from_documents(
#     collection_name="DataEng",
#     embedding=embed_model,
#     documents=docs,
#     persist_directory="./my_chroma_db"
# )

# retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# res= retriever.invoke("What is data lakehouse")

# print(res[0].page_content)


# # models/gemini-embedding-exp-03-07
# # models/text-embedding-004

# prompt = PromptTemplate(
# template="""You are an data engineer expert for question-answering tasks.
#             Use the following pieces of retrieved context to answer the question. If you don't know the answer,
#             just say that you don't know,  don't try to make up an answer.

#     Context: {context}

#     Question: {question}

#     Answer: """,
# input_variables=["context","question"]
# )

# prompt = prompt.invoke({"context":contextres,"question":"what is data engineering"})

# result = llm_model.invoke(prompt)

# print(result.content)
# print(prompt)