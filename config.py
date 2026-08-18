# config — shared constants needed wherever the chatbot actually runs (desktop or PythonAnywhere)
DB_PATH = "assets/stellarsearch.db"
OUTPUT_NPZ = "movie_embeddings.npz"

OPENAI_EMBEDDING_MODEL = 'text-embedding-3-small'
EMBEDDING_DIMENSIONS = 1536  # model's full native dimension; truncating hurt retrieval quality

OPENAI_MODEL = "gpt-3.5-turbo"
TOP_K = 5
