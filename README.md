


# RAG (Retrieval-Augmented Generation) in development
Classic RAG pipeline plan: 
Query → Embedding → Vector Search → Retrieved Context → LLM Answer

## prod
pip install "fastapi[standard]"
fastapi dev
fastapi cloud env set ENVIRONMENT "production"
fastapi cloud env set --secret API_KEY "your-api-key"

## dev
pip install fastapi uvicorn
uvicorn main:app --reload

Tutorials on https://realpython.com/