# knowledge/views.py
import os
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions
from .serializers import DocumentSerializer
from .utils.ingest import ingest_document
from .utils.rag import get_top_k_chunks, generate_answer

class UploadDocumentView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        """
        Accepts multipart-form 'file' upload and optional 'source' metadata.
        """
        uploaded = request.FILES.get("file")
        source = request.data.get("source", "")
        if not uploaded:
            return Response({"error": "file required"}, status=status.HTTP_400_BAD_REQUEST)
        raw = uploaded.read()
        doc = ingest_document(uploaded.name, raw, source=source, use_openai=False)
        return Response({"message": "ingested", "document_id": doc.id})

class AskQuestionView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        """
        Body JSON: {"question": "...", "top_k": 5}
        Returns: answer + sources (top chunks)
        """
        question = request.data.get("question")
        if not question:
            return Response({"error": "question required"}, status=status.HTTP_400_BAD_REQUEST)
        top_k = int(request.data.get("top_k", os.getenv("TOP_K", 5)))
        top_chunks = get_top_k_chunks(question, k=top_k, use_openai=False)
        if not top_chunks:
            return Response({"answer": "", "sources": []})

        # call generator (RAG)
        answer = generate_answer(question, top_chunks)
        return Response({"answer": answer, "sources": top_chunks})
