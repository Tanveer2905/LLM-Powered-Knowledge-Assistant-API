import os
import faiss
import pickle
import numpy as np
import re
import shutil
from django.conf import settings
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


MODEL_NAME = "google/flan-t5-large"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
MAX_NEW_TOKENS = 256
MODEL_MAX_TOKENS = 512
BUFFER = 50

BASE_DIR = settings.BASE_DIR
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")
os.makedirs(FAISS_INDEX_PATH, exist_ok=True)

class RAGService:
    def __init__(self):
        print("Loading models... (This may take a moment)")
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
            self.llm = pipeline(
                "text2text-generation",
                model=MODEL_NAME,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                model_kwargs={"local_files_only": True}
            )
        except Exception:
            print("⚠️ Local model not found. Downloading...")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.llm = pipeline(
                "text2text-generation",
                model=MODEL_NAME,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False
            )
        print("✅ Models loaded successfully.")

    def clear_database(self):
        if os.path.exists(FAISS_INDEX_PATH):
            shutil.rmtree(FAISS_INDEX_PATH)
            os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
            return True
        return False

    def ingest_file(self, file_path):
        """
        Processes PDF and saves text + source filename.
        """
        raw_filename = os.path.basename(file_path)
        clean_filename = raw_filename.replace("temp_", "")

        loader = PyPDFLoader(file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=450,
            chunk_overlap=150
        )
        docs = text_splitter.split_documents(documents)
        
        new_chunks_data = [
            {'text': doc.page_content, 'source': clean_filename} 
            for doc in docs
        ]
        
        if not new_chunks_data: return 0

        new_texts = [item['text'] for item in new_chunks_data]
        new_embeddings = self.embed_model.encode(new_texts)
        

        index_file = os.path.join(FAISS_INDEX_PATH, "index.bin")
        chunks_file = os.path.join(FAISS_INDEX_PATH, "chunks.pkl")

        if os.path.exists(index_file) and os.path.exists(chunks_file):
            index = faiss.read_index(index_file)
            with open(chunks_file, "rb") as f:
                all_chunks_data = pickle.load(f)
            all_chunks_data.extend(new_chunks_data)
        else:
            dimension = new_embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            all_chunks_data = new_chunks_data

        index.add(np.array(new_embeddings).astype('float32'))
        faiss.write_index(index, index_file)
        with open(chunks_file, "wb") as f:
            pickle.dump(all_chunks_data, f)

        return len(new_chunks_data)

    def retrieve_context(self, question, k=30):
        index_file = os.path.join(FAISS_INDEX_PATH, "index.bin")
        chunks_file = os.path.join(FAISS_INDEX_PATH, "chunks.pkl")

        if not os.path.exists(index_file) or not os.path.exists(chunks_file):
            return []

        index = faiss.read_index(index_file)
        with open(chunks_file, "rb") as f:
            chunks_data = pickle.load(f)

        q_emb = self.embed_model.encode([question])
        _, idx = index.search(np.array(q_emb).astype('float32'), k)
        
        valid_indices = [i for i in idx[0] if 0 <= i < len(chunks_data)]
        return [chunks_data[i] for i in valid_indices]

    def clean_text(self, text):
        text = re.sub(r'Activity\s+\d+\.\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Fig\.\s*\d+\.\d+', '', text, flags=re.IGNORECASE)
        text = text.replace("\n", " ").replace("  ", " ")
        return text

    def smart_rerank(self, chunks_data, question):
        """
        Reranks dictionary chunks based on their 'text' content.
        """
        question_lower = question.lower()
        stop_words = ["what", "are", "the", "of", "in", "is", "explain", "describe", "how", "to", "do", "does", "define", "write"]
        keywords = [w for w in question_lower.split() if w not in stop_words and len(w) > 2]

        scored_chunks = []
        for i, chunk_obj in enumerate(chunks_data):
            score = len(chunks_data) - i
            
            text_content = chunk_obj['text']
            chunk_clean = self.clean_text(text_content.lower())
            
            for word in keywords:
                if word in chunk_clean:
                    score += 15
            
            if question_lower in chunk_clean:
                score += 50

            if "take a beaker" in chunk_clean or "materials required" in chunk_clean:
                score -= 10
            
            scored_chunks.append((score, chunk_obj))
        
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored_chunks[:5]]

    def truncate_to_tokens(self, text, max_tokens):
        tokens = self.tokenizer.encode(text, truncation=True, max_length=max_tokens)
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def build_prompt(self, context_chunks, question):

        cleaned_texts = [self.clean_text(c['text']) for c in context_chunks]
        combined_text = "\n".join(cleaned_texts)
        
        fixed_text = f"Question: {question}\nAnswer:"
        fixed_tokens = len(self.tokenizer.encode(fixed_text))
        available_space = MODEL_MAX_TOKENS - fixed_tokens - MAX_NEW_TOKENS - BUFFER

        safe_context = self.truncate_to_tokens(combined_text, max(0, available_space))

        return f"""You are a science tutor.
Answer the question using ONLY the provided context.
If the answer contains multiple points, list them as bullet points.
Do not mention "Activities" or "Experiments".

Context:
{safe_context}

Question:
{question}

Answer:"""

    def ask_question(self, question):
        # 1. Retrieve (Get list of dictionaries)
        raw_chunks_data = self.retrieve_context(question, k=30)
        if not raw_chunks_data:
            return {"answer": "No relevant documents found.", "sources": []}

        # 2. Rerank
        best_chunks_data = self.smart_rerank(raw_chunks_data, question)
        
        # 3. Extract Sources (Get unique filenames from the best chunks)
        unique_sources = list(set([c['source'] for c in best_chunks_data]))
        
        # 4. Build Prompt
        prompt = self.build_prompt(best_chunks_data, question)

        # 5. Generate
        result = self.llm(prompt)[0]["generated_text"]

        return {
            "answer": result,
            "sources": unique_sources
        }