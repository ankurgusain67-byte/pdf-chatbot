from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)

vector = embeddings.embed_query('What is machine learning?')
print(len(vector))   # prints 384 — that's 384 numbers for one sentence
print(vector[:5])    # prints first 5 numbers so you can see what it looks like