from django.shortcuts import render
from django.views import View
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os
from .rag_service import RAGService

# Initialize Service once when the server starts
rag_service = RAGService()

class IngestDocumentView(APIView):
    """
    Handles PDF ingestion. Supports uploading multiple files at once.
    """
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        # 1. Get all files uploaded with key 'file' (or 'files')
        files = request.FILES.getlist('file')
        
        if not files:
            return Response({"error": "No files uploaded"}, status=status.HTTP_400_BAD_REQUEST)

        total_chunks = 0
        processed_files = []
        errors = []

        for file_obj in files:
            # 2. Save file temporarily
            # We use 'temp_' prefix so we can easily identify temp files if needed
            file_name = f"temp_{file_obj.name}"
            try:
                temp_path = default_storage.save(file_name, ContentFile(file_obj.read()))
                full_temp_path = os.path.join(default_storage.location, temp_path)

                # 3. Process the file (Appends to DB)
                chunks_count = rag_service.ingest_file(full_temp_path)
                total_chunks += chunks_count
                processed_files.append(file_obj.name)

            except Exception as e:
                errors.append(f"Error processing {file_obj.name}: {str(e)}")
            
            finally:
                # 4. Cleanup temp file immediately to save space
                if 'full_temp_path' in locals() and os.path.exists(full_temp_path):
                    os.remove(full_temp_path)

        response_data = {
            "message": "Processing complete",
            "files_processed": processed_files,
            "total_new_chunks": total_chunks
        }
        
        if errors:
            response_data["errors"] = errors
            
        return Response(response_data, status=status.HTTP_200_OK)


class AskQuestionView(APIView):
    """
    Handles asking questions to the RAG service.
    """
    def post(self, request, *args, **kwargs):
        question = request.data.get('question')
        if not question:
            return Response({"error": "Question field is required"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            result = rag_service.ask_question(question)
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class ClearDBView(APIView):
    """
    Optional endpoint to clear the database.
    Add path('api/clear/', ClearDBView.as_view()) to urls.py if needed.
    """
    def post(self, request, *args, **kwargs):
        rag_service.clear_database()
        return Response({"message": "Database cleared successfully"})

class HomeView(View):
    """
    Serves the main HTML UI.
    """
    def get(self, request):
        return render(request, 'index.html')