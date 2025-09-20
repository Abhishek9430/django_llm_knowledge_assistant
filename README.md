this is a django based llm knowledge assistant in which you can provide a document and later ask questions regarding the same document.
- Upload and manage documents.
- Extract text from multiple formats.
- Chunking & semantic embeddings.
- Retrieval-Augmented Generation (RAG).
- Works with **DeepSeek API** (default) or OpenAI.
- Uses Sentence Transformers locally for embeddings (flexibility is there to use OPENAI for embedding)

- steps to run
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt

-inside .env change the API key
  DEEPSEEK_API_KEY=your_deepseek_api_key

-db setup
  python manage.py makemigrations
  python manage.py migrate

-**runserver**
`python manage.py runserver


-upload document
  curl -X POST http://127.0.0.1:8000/api/knowledge/upload-doc/ \
    -F "file=@/path/to/sample.pdf"

-ask question
  curl -X POST http://127.0.0.1:8000/api/knowledge/ask-question/ \
    -H "Content-Type: application/json" \
    -d '{"question": "Summarize the uploaded document"}'
