# Download embeddings from Hugging Face
import torch

from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings


# Extract Data from the PDF File
def load_pdf_data(data):
    loader = DirectoryLoader(data,
                             glob="*.pdf",
                             loader_cls=PyPDFLoader
                             )
    documents = loader.load()

    return documents


# Split Data into Text Chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


# Patch torch.get_default_device if missing
if not hasattr(torch, "get_default_device"):
    torch.get_default_device = (
        lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )


def download_hugging_face_embeddings():
    # for clustering semantic search
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/'
                                       'all-MiniLM-L6-v2')
    return embeddings
