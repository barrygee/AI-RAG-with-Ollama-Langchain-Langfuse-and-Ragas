from dotenv import load_dotenv
import os
import uuid

from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langfuse.callback import CallbackHandler

session_id = str(uuid.uuid4())

langfuse_handler = CallbackHandler(
  secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
  public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
  host=os.getenv("LANGFUSE_HOST"),
  session_id=session_id
)

def load_environment_variables():
  load_dotenv()
  model = os.getenv("MODEL_NAME")
  base_url = os.getenv("LANGFUSE_HOST")
  return model, base_url

def load_documents():
  current_directory = os.path.dirname(os.path.abspath(__file__))
  docs_directory_path = os.path.join(current_directory, "./docs")
  pdfs = [p for p in os.listdir(docs_directory_path) if p.endswith(".pdf")]
  documents = []

  for pdf in pdfs:
      file_path = os.path.join(docs_directory_path, pdf)
      loader = PyPDFLoader(file_path)
      pdf_docs = loader.load()
      documents.extend(pdf_docs)
      for doc in pdf_docs:
          documents.append(doc)

  return documents

def split_documents(documents):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
  docs = text_splitter.split_documents(documents)
  return docs

def create_and_store_embeddings(docs):
  embeddings = OllamaEmbeddings(model='nomic-embed-text')
  doc_vectors = embeddings.embed_documents([t.page_content for t in docs])

  connection_string = f"postgresql+psycopg2://{os.getenv('POSTGRES_USERNAME')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
  collection_name = os.getenv('PGVECTOR_COLLECTION_NAME')

  vectorstore = PGVector.from_documents(
      embedding=embeddings,
      documents=docs,
      connection=connection_string,
      collection_name=collection_name,
      use_jsonb=True,
      async_mode=False,
  )

  return vectorstore

def retrieve_relevant_documents(vectorstore, question):
  retriever = vectorstore.as_retriever(
      search_type="similarity_score_threshold",
      search_kwargs={'k': 5, 'score_threshold': 0.1}
  )

  relevant_docs = retriever.invoke(question)
  return relevant_docs

def initialize_model(model_name):
  model = Ollama(
      model=model_name,
      temperature=2
  )
  return model

def create_prompt_template():
  template = """Answer the question based on the following context: {context}
  If you are unable to find the answer within the context, please respond with 'I don't know'.

  Question: {question}
  """
  prompt_template = ChatPromptTemplate.from_messages(
      [
          ("system", template),
          ("human", "{question}")
      ]
  )
  return prompt_template

def create_chain(prompt_template, model):
  chain = prompt_template | model | StrOutputParser()
  return chain

def main(q="What is the Annual budget?"):
  model_name, base_url = load_environment_variables()
  question = q
  documents = load_documents()
  docs = split_documents(documents)
  vectorstore = create_and_store_embeddings(docs)
  relevant_docs = retrieve_relevant_documents(vectorstore, question)

  model = initialize_model(model_name)
  prompt_template = create_prompt_template()
  chain = create_chain(prompt_template, model)

  result = chain.invoke({"context": relevant_docs, "question": question}, config={"callbacks": [langfuse_handler]})
  print(result)

if __name__ == "__main__":
    main()
