# %% [markdown]
# # Loading Text, Chunking, Embedding and Upserting into Pinecone Index
# 
# Got most of these from James Briggs' notebook: https://www.pinecone.io/learn/langchain-retrieval-augmentation/

# %% [markdown]
# ### 1. Load Text

# %%
doc_path = (r"hr_policy.txt")

# Open the file
with open(doc_path, 'r') as f:
    # Read the file
    contents = f.read()

# %%
# set up tokenizer
import tiktoken
tokenizer = tiktoken.get_encoding('p50k_base')


# create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

# sample
tiktoken_len("hello I am a chunk of text and using the tiktoken_len function "
             "we can find the length of this chunk of text in tokens")

# %% [markdown]
# ### 2. Create chunking function

# %%
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""]
)

chunks = text_splitter.split_text(contents)
chunks[0]

# %% [markdown]
# ### 3. Create Embeddings

# %%
# initialize embedding function
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

model_name = 'text-embedding-ada-002'

# set embeddings function
embed = OpenAIEmbeddings(model = model_name)

# %%
# create data format from chunked text for upserting into Pinecone index. Format: id, embeddings, metadata
from uuid import uuid4

vectors = [(str(uuid4()), embed.embed_documents([text])[0], {"text": text}) for text in chunks]


# %% [markdown]
# #### How the 'vectors' or embeddings look when printed. 
# There are 1536 elements to the vector representing each chunk of data.

# %% [markdown]
# ![vectors.png](attachment:vectors.png)

# %% [markdown]
# ### 4. Prep Pinecone Index

# %%
import pinecone

index_name = 'tk-policy'
dimension=1536

pinecone.init(
        api_key="275bf42f-7112-4c33-94b7-7e821c83bcc4",  # get yours from pinecone.io. there is a free tier.
        environment="us-west4-gcp-free"  
)

# delete index if it exists
if index_name in pinecone.list_indexes():
    pinecone.delete_index(index_name)

# create index
pinecone.create_index(
        name=index_name,
        metric='cosine',
        dimension=dimension       
)

# %% [markdown]
# ### 5. Upsert vectors to index

# %%
# connect to index
index = pinecone.Index(index_name)

# upsert vectors to pinecone
index.upsert(
    vectors=vectors,
    #namespace=index_name, 
    values=True, 
    include_metadata=True
    )

index.describe_index_stats()


