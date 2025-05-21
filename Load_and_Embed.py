from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.environ["GOOGLE_API_KEY"]

loader = DirectoryLoader(path="Books", glob="*.pdf", loader_cls=PyPDFLoader)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    add_start_index=True,
)
split_docs = text_splitter.split_documents(docs)

embed_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", google_api_key=api_key
)


vectorstore = Chroma(
    collection_name="DataEng",
    embedding_function=embed_model,
    persist_directory="./my_chroma_db",
)

vectorstore.add_documents(documents=split_docs)
