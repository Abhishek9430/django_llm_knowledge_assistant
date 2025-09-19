# knowledge/models.py
from django.db import models

class Document(models.Model):
    filename = models.CharField(max_length=512)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    source = models.CharField(max_length=256, blank=True, null=True)  # optional metadata

    def __str__(self):
        return self.filename


class Chunk(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="chunks")
    chunk_text = models.TextField()
    start_offset = models.IntegerField(default=0)
    end_offset = models.IntegerField(default=0)
    # store numpy float32 bytes
    embedding = models.BinaryField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["created_at"]),
        ]
