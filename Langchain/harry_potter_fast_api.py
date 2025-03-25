from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
parser=StrOutputParser()
from langchain_openai import ChatOpenAI
import getpass
import os
os.environ["USER_AGENT"] = "MyApp/1.0"

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
from langchain_core.prompts import PromptTemplate

loader = WebBaseLoader("https://en.wikipedia.org/wiki/Harry_Potter_(film_series)")
docs = loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1000,
    chunk_overlap=20,
    
)
texts = text_splitter.split_documents(docs)
from langchain_openai import OpenAIEmbeddings
embeddings= OpenAIEmbeddings()
l=[]
for i in texts:
    l.append(i.page_content)
from langchain_core.vectorstores import InMemoryVectorStore
vectorstore = InMemoryVectorStore.from_texts(
    l,
    embedding=embeddings,
)
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(model="gpt-3.5-turbo",)
from fastapi import FastAPI
app = FastAPI()
# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Harry Potter API!"}

# Endpoint for getting the answer based on the question
@app.get("/result")

@app.get("/result")
def read_item(question:str):
    prompt = PromptTemplate.from_template("Anser the {question} based on the {input} ")
   
    retrieved_documents = retriever.invoke(question)
    input=''
    for i in retrieved_documents[:3] :
        input=input+i.page_content
    chain = prompt | llm |parser
    answer=chain.invoke(
        {
        "question":question ,
        "input":input ,
        }
    )
    return answer


