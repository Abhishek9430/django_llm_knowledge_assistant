# knowledge/urls.py
from django.urls import path
from .views import UploadDocumentView, AskQuestionView

urlpatterns = [
    path("upload-doc/", UploadDocumentView.as_view(), name="upload-doc"),
    path("ask-question/", AskQuestionView.as_view(), name="ask-question"),
]
