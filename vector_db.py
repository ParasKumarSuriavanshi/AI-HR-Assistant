import os
from langchain_community.document_loaders import TextLoader         #TO LOAD FILE FROM SYSTEM INTO OUR WORK ENVI.
from langchain_text_splitters import RecursiveCharacterTextSplitter     #TO SPLIT THE TEXT INTO SMALLER CHUNKS
from langchain_ollama import OllamaEmbeddings                           #TO CONVERT THE WORD INTO VECTORS
from langchain_community.vectorstores import FAISS                  #TO CREATE AND STORE THE DATABASE

documents = []

for file in os.listdir("documents"):    #have three file in the document folder so loop will run three times
    path = os.path.join("documents" , file)
    loader = TextLoader(path)
    docs = loader.load()                #create the object for the file
    documents.extend(docs)              #stored the object into the list
print(f"loaded {len(documents)} documents.")


#text splitter
splitter = RecursiveCharacterTextSplitter(chunk_size = 500 , chunk_overlap = 50)

chunk = splitter.split_documents(documents)

print(f"Created {len(chunk)} chunks")


#embedding
embedding = OllamaEmbeddings(model = "nomic-embed-text")

vector_db = FAISS.from_documents(chunk , embedding)             #store the vector in the db
vector_db.save_local("hr_vector_db")            #it wont get lost even we exit the code

print("Vector database id created.")