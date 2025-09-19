# knowledge/admin.py
from django.contrib import admin
from .models import Document, Chunk

@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ("id", "filename", "uploaded_at", "source")

@admin.register(Chunk)
class ChunkAdmin(admin.ModelAdmin):
    list_display = ("id", "document", "start_offset", "end_offset", "created_at")
    readonly_fields = ("embedding",)
