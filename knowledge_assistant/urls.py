from django.contrib import admin
from django.urls import path
# IMPORT HOMEVIEW HERE 👇
from api.views import IngestDocumentView, AskQuestionView, ClearDBView, HomeView

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # Main UI (Root URL)
    path('', HomeView.as_view(), name='home'),

    # API Endpoints
    path('api/ingest/', IngestDocumentView.as_view(), name='ingest'),
    path('api/ask/', AskQuestionView.as_view(), name='ask'),
    path('api/clear/', ClearDBView.as_view(), name='clear_db'),
]