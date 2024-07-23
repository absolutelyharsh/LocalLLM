from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()


def get_embedding_function():
    embed_model_name = os.getenv("EMBED_MODEL_NAME")
    model_kwargs = {"device":os.getenv("MODEL_RUN_DEVICE")}
    encode_kwargs = {"normalize_embeddings": True}
    hf = HuggingFaceBgeEmbeddings(
        model_name=embed_model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    return hf

