# debug_rag.py
import os
import django
from django.conf import settings

# 1. Setup Django standalone (to access settings)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'knowledge_assistant.settings') # <--- CHECK THIS NAME
# If your project folder is named 'core' or something else, change the line above!
# Based on your previous errors, your project seems to be 'knowledge_assistant'.

if not settings.configured:
    django.setup()

# 2. Import your service
try:
    from api.rag_service import RAGService
    print("Initializing RAG Service...")
    service = RAGService()
    print("✅ Service Initialized.")

    # 3. Test Ingestion (Create a dummy file if needed or skip)
    # print("Testing Ingestion...")
    # service.ingest_file("science.pdf") 

    # 4. Test Question
    print("Testing Question...")
    response = service.ask_question("What is the summary?")
    print("\n✅ SUCCESS! Response:")
    print(response)

except Exception as e:
    print("\n❌ ERROR OCCURRED:")
    print(e)
    import traceback
    traceback.print_exc()