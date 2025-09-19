# knowledge/serializers.py
from rest_framework import serializers
from .models import Document, Chunk

class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = "__all__"

class ChunkSerializer(serializers.ModelSerializer):
    class Meta:
        model = Chunk
        fields = "__all__"
