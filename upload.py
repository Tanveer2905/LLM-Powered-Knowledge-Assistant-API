import requests

# REPLACE 'science.pdf' WITH YOUR ACTUAL FILE NAME
filename = 'knowledge_assistant/motion.pdf' 

url = 'http://127.0.0.1:8000/api/ingest/'

with open(filename, 'rb') as f:
    files = {'file': f}
    response = requests.post(url, files=files)

print(response.json())