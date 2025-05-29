import os

import torch

from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pinecone import ServerlessSpec

from src.helper import load_pdf_data, text_split
from src.helper import download_hugging_face_embeddings


# Patch torch.get_default_device if missing
if not hasattr(torch, "get_default_device"):
    torch.get_default_device = (
        lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

extracted_data = load_pdf_data(data='data/')
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

pc = Pinecone(
    api_key=PINECONE_API_KEY,
    environment="us-east1-aws"
)

index_name = "medicalbot"

pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

# Embed each chunk and upsert the embeddings into
# Pinecone index.
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings
)
