import os
import re
import json
import time
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
import numpy as np

from flask import Flask, request, jsonify, session
from flask_cors import CORS
from flask_session import Session
from dotenv import load_dotenv

# For Pinecone version 6.0.0
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
# Corrected import for create_stuff_documents_chain
# Correct import for LangChain 0.3.26
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import requests
from bs4 import BeautifulSoup
import PyPDF2
import io
import base64
# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

# Load environment variables
load_dotenv()

# Initialize Flask app with session support
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)
CORS(app, supports_credentials=True)

# --- Configuration ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "ai-qna-index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize services
pc = Pinecone(api_key=PINECONE_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# --- Advanced Text Processing Classes ---

class SmartTextSplitter:
    """Intelligent text splitter that considers document structure and semantics"""

    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """Split text using multiple strategies"""
        # Strategy 1: Split by sections (headers)
        section_chunks = self._split_by_sections(text)

        # Strategy 2: Split by semantic similarity
        semantic_chunks = []
        for chunk in section_chunks:
            if len(chunk) > self.chunk_size:
                semantic_chunks.extend(self._semantic_split(chunk))
            else:
                semantic_chunks.append(chunk)

        # Strategy 3: Ensure proper overlap
        final_chunks = self._add_smart_overlap(semantic_chunks)

        return final_chunks

    def _split_by_sections(self, text: str) -> List[str]:
        """Split text by headers and sections"""
        # Look for headers (lines starting with #, or all caps, or followed by ===)
        sections = []
        current_section = ""
        lines = text.split('\n')

        for line in lines:
            line = line.strip()
            if self._is_header(line):
                if current_section:
                    sections.append(current_section.strip())
                current_section = line + '\n'
            else:
                current_section += line + '\n'

        if current_section:
            sections.append(current_section.strip())

        return sections if sections else [text]

    def _is_header(self, line: str) -> bool:
        """Detect if a line is a header"""
        if not line:
            return False

        # Markdown headers
        if line.startswith('#'):
            return True

        # All caps headers (short lines)
        if line.isupper() and len(line) < 100:
            return True

        # Numbered sections
        if re.match(r'^\d+\.?\s+[A-Z]', line):
            return True

        return False

    def _semantic_split(self, text: str) -> List[str]:
        """Split text based on semantic similarity between sentences"""
        sentences = sent_tokenize(text)
        if len(sentences) <= 1:
            return [text]

        # Calculate embeddings for sentences
        try:
            sentence_embeddings = [embeddings.embed_query(sent) for sent in sentences]
        except:
            # Fallback to simple splitting
            return self._simple_split(text)

        # Find natural break points
        chunks = []
        current_chunk = []
        current_length = 0

        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)

            # If adding this sentence would exceed chunk size, check if we should split
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Check semantic similarity with next sentence
                if i < len(sentences) - 1:
                    current_embedding = sentence_embeddings[i]
                    next_embedding = sentence_embeddings[i + 1]
                    similarity = cosine_similarity([current_embedding], [next_embedding])[0][0]

                    # If similarity is low, it's a good place to split
                    if similarity < 0.5:
                        chunks.append(' '.join(current_chunk))
                        current_chunk = [sentence]
                        current_length = sentence_length
                        continue

                # If we haven't split yet and chunk is too long, force split
                if current_length > self.chunk_size:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_length
                    continue

            current_chunk.append(sentence)
            current_length += sentence_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def _simple_split(self, text: str) -> List[str]:
        """Fallback simple splitting"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        return splitter.split_text(text)

    def _add_smart_overlap(self, chunks: List[str]) -> List[str]:
        """Add intelligent overlap between chunks"""
        if len(chunks) <= 1:
            return chunks

        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
            else:
                # Add last few sentences from previous chunk
                prev_sentences = sent_tokenize(chunks[i-1])
                current_sentences = sent_tokenize(chunk)

                # Take last 2 sentences from previous chunk
                overlap_sentences = prev_sentences[-2:] if len(prev_sentences) >= 2 else prev_sentences

                # Combine with current chunk
                overlapped_chunk = ' '.join(overlap_sentences + current_sentences)
                overlapped_chunks.append(overlapped_chunk)

        return overlapped_chunks


class QueryProcessor:
    """Advanced query processing and expansion"""

    def __init__(self):
        self.synonyms_cache = {}

    def process_query(self, query: str) -> Dict[str, Any]:
        """Process and analyze the query"""
        return {
            'original': query,
            'cleaned': self._clean_query(query),
            'expanded': self._expand_query(query),
            'intent': self._detect_intent(query),
            'keywords': self._extract_keywords(query),
            'question_type': self._classify_question_type(query)
        }

    def _clean_query(self, query: str) -> str:
        """Clean and normalize the query"""
        # Remove extra whitespace
        query = ' '.join(query.split())

        # Fix common typos (basic)
        query = query.replace('whats', 'what is')
        query = query.replace('hows', 'how is')
        query = query.replace('wheres', 'where is')

        return query.strip()

    def _expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms and related terms"""
        expanded_queries = [query]

        # Extract key terms
        words = word_tokenize(query.lower())
        key_words = [word for word in words if word not in stop_words and len(word) > 2]

        # Simple synonym expansion (you can enhance this with word2vec or WordNet)
        synonym_map = {
            'fast': ['quick', 'rapid', 'speedy'],
            'big': ['large', 'huge', 'massive'],
            'small': ['tiny', 'little', 'mini'],
            'good': ['excellent', 'great', 'fine'],
            'bad': ['poor', 'terrible', 'awful'],
            'important': ['crucial', 'vital', 'significant'],
            'method': ['approach', 'technique', 'way'],
            'problem': ['issue', 'challenge', 'difficulty'],
            'solution': ['answer', 'resolution', 'fix'],
            'example': ['instance', 'case', 'sample'],
            'difference': ['distinction', 'contrast', 'variation'],
            'benefit': ['advantage', 'profit', 'gain'],
            'disadvantage': ['drawback', 'limitation', 'weakness']
        }

        # Create expanded versions
        for word in key_words:
            if word in synonym_map:
                for synonym in synonym_map[word]:
                    expanded_query = query.replace(word, synonym)
                    expanded_queries.append(expanded_query)

        return expanded_queries

    def _detect_intent(self, query: str) -> str:
        """Detect the intent of the query"""
        query_lower = query.lower()

        if any(word in query_lower for word in ['what', 'define', 'definition', 'meaning']):
            return 'definition'
        elif any(word in query_lower for word in ['how', 'steps', 'process', 'way']):
            return 'how_to'
        elif any(word in query_lower for word in ['why', 'reason', 'because']):
            return 'explanation'
        elif any(word in query_lower for word in ['where', 'location', 'place']):
            return 'location'
        elif any(word in query_lower for word in ['when', 'time', 'date']):
            return 'temporal'
        elif any(word in query_lower for word in ['compare', 'difference', 'vs', 'versus']):
            return 'comparison'
        elif any(word in query_lower for word in ['example', 'instance', 'sample']):
            return 'example'
        elif any(word in query_lower for word in ['list', 'enumerate', 'types']):
            return 'list'
        else:
            return 'general'

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query"""
        words = word_tokenize(query.lower())

        # Remove stop words and short words
        keywords = [word for word in words if word not in stop_words and len(word) > 2]

        # Lemmatize keywords
        keywords = [lemmatizer.lemmatize(word) for word in keywords]

        return keywords

    def _classify_question_type(self, query: str) -> str:
        """Classify the type of question"""
        query_lower = query.lower()

        if query_lower.startswith(('what', 'who', 'when', 'where', 'why', 'how')):
            return 'wh_question'
        elif query_lower.startswith(('is', 'are', 'can', 'could', 'would', 'should', 'do', 'does')):
            return 'yes_no_question'
        elif '?' in query:
            return 'question'
        else:
            return 'statement'


class ConversationManager:
    """Manage conversation history and context"""

    def __init__(self):
        self.conversations = defaultdict(list)
        self.context_window = 5  # Keep last 5 exchanges

    def add_exchange(self, session_id: str, question: str, answer: str):
        """Add a question-answer exchange to conversation history"""
        exchange = {
            'question': question,
            'answer': answer,
            'timestamp': datetime.now().isoformat()
        }

        self.conversations[session_id].append(exchange)

        # Keep only recent exchanges
        if len(self.conversations[session_id]) > self.context_window:
            self.conversations[session_id] = self.conversations[session_id][-self.context_window:]

    def get_context(self, session_id: str) -> str:
        """Get conversation context for better responses"""
        if session_id not in self.conversations:
            return ""

        context_parts = []
        for exchange in self.conversations[session_id][-3:]:  # Last 3 exchanges
            context_parts.append(f"Previous Q: {exchange['question']}")
            context_parts.append(f"Previous A: {exchange['answer'][:200]}...")

        return "\n".join(context_parts)

    def is_follow_up(self, session_id: str, question: str) -> bool:
        """Check if this is a follow-up question"""
        if session_id not in self.conversations or not self.conversations[session_id]:
            return False

        # Simple heuristics for follow-up detection
        follow_up_indicators = [
            'what about', 'how about', 'and', 'also', 'furthermore',
            'can you explain', 'tell me more', 'what else', 'continue',
            'expand on', 'elaborate', 'more details'
        ]

        question_lower = question.lower()
        return any(indicator in question_lower for indicator in follow_up_indicators)


class AdvancedRetriever:
    """Advanced retrieval with hybrid search and reranking"""

    def __init__(self, index, embeddings):
        self.index = index
        self.embeddings = embeddings
        self.documents = []  # Store original documents for BM25
        self.bm25_retriever = None

    def add_documents(self, documents: List[str]):
        """Add documents for BM25 retrieval"""
        self.documents.extend(documents)

        # Create BM25 retriever
        doc_objects = [Document(page_content=doc) for doc in documents]
        self.bm25_retriever = BM25Retriever.from_documents(doc_objects)
        self.bm25_retriever.k = 5

    def hybrid_search(self, query: str, k: int = 5) -> List[Dict]:
        """Perform hybrid search combining semantic and keyword matching"""
        # Semantic search
        query_embedding = self.embeddings.embed_query(query)
        semantic_results = self.index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True
        )

        # Keyword search using BM25
        keyword_results = []
        if self.bm25_retriever:
            try:
                bm25_docs = self.bm25_retriever.get_relevant_documents(query)
                keyword_results = [
                    {
                        'score': 0.7,  # Default score for BM25 results
                        'metadata': {'text': doc.page_content},
                        'id': f'bm25_{i}'
                    }
                    for i, doc in enumerate(bm25_docs[:k])
                ]
            except:
                pass

        # Combine and deduplicate results
        combined_results = self._combine_results(semantic_results['matches'], keyword_results)

        # Rerank results
        reranked_results = self._rerank_results(query, combined_results)

        return reranked_results[:k]

    def _combine_results(self, semantic_results: List, keyword_results: List) -> List:
        """Combine semantic and keyword search results"""
        combined = {}

        # Add semantic results
        for result in semantic_results:
            text = result['metadata']['text']
            combined[text] = {
                'score': result['score'],
                'metadata': result['metadata'],
                'id': result['id'],
                'source': 'semantic'
            }

        # Add keyword results (boost score if already exists)
        for result in keyword_results:
            text = result['metadata']['text']
            if text in combined:
                # Boost score for items found in both searches
                combined[text]['score'] = min(1.0, combined[text]['score'] + 0.2)
                combined[text]['source'] = 'both'
            else:
                combined[text] = {
                    'score': result['score'],
                    'metadata': result['metadata'],
                    'id': result['id'],
                    'source': 'keyword'
                }

        return list(combined.values())

    def _rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """Rerank results based on relevance"""
        query_words = set(word_tokenize(query.lower()))

        for result in results:
            text = result['metadata']['text'].lower()
            text_words = set(word_tokenize(text))

            # Calculate additional relevance factors
            word_overlap = len(query_words.intersection(text_words)) / len(query_words) if query_words else 0
            length_penalty = 1.0 if len(text) > 100 else 0.8  # Prefer longer texts

            # Adjust score
            relevance_boost = word_overlap * 0.3 + length_penalty * 0.1
            result['score'] = min(1.0, result['score'] + relevance_boost)

        # Sort by score
        return sorted(results, key=lambda x: x['score'], reverse=True)


class AnswerGenerator:
    """Generate high-quality answers with different strategies"""

    def __init__(self, llm=None):
        self.llm = llm
        self.query_processor = QueryProcessor()

    def generate_answer(self, query: str, documents: List[Document],
                        conversation_context: str = "", intent: str = "general") -> str:
        """Generate answer using multiple strategies"""

        if self.llm:
            return self._generate_with_llm(query, documents, conversation_context, intent)
        else:
            return self._generate_extractive_answer(query, documents, intent)

    def _generate_with_llm(self, query: str, documents: List[Document],
                           context: str, intent: str) -> str:
        """Generate answer using LLM with advanced prompting"""

        # Create context-aware prompt
        prompt_template = self._get_prompt_template(intent)

        context_text = "\n\n".join([doc.page_content for doc in documents])

        # Add conversation context if available
        if context:
            context_text = f"Conversation Context:\n{context}\n\nRelevant Documents:\n{context_text}"

        try:
            chain = create_stuff_documents_chain(self.llm, prompt_template)
            response = chain.invoke({
                "context": documents,
                "input": query
            })
            return response
        except Exception as e:
            print(f"LLM generation failed: {e}")
            return self._generate_extractive_answer(query, documents, intent)

    def _get_prompt_template(self, intent: str) -> ChatPromptTemplate:
        """Get appropriate prompt template based on intent"""

        base_prompt = """You are an intelligent assistant. Use the provided context to answer the question accurately and comprehensively.

Context: {context}

Question: {input}

Instructions:
- Provide a detailed, accurate answer based on the context
- If the context doesn't contain enough information, clearly state what's missing
- Structure your answer logically with clear explanations
- Use examples from the context when relevant
- Be concise but thorough

"""

        intent_prompts = {
            'definition': base_prompt + "Focus on providing a clear, comprehensive definition with examples.",
            'how_to': base_prompt + "Provide step-by-step instructions or process explanations.",
            'explanation': base_prompt + "Explain the reasoning, causes, or mechanisms behind the topic.",
            'comparison': base_prompt + "Compare and contrast the different aspects mentioned in the question.",
            'example': base_prompt + "Provide specific examples and illustrations from the context.",
            'list': base_prompt + "Organize your answer as a clear, structured list or enumeration."
        }

        prompt_text = intent_prompts.get(intent, base_prompt)
        return ChatPromptTemplate.from_template(prompt_text)

    def _generate_extractive_answer(self, query: str, documents: List[Document], intent: str) -> str:
        """Generate answer using extractive methods when LLM is not available"""

        if not documents:
            return "I don't have enough information to answer this question."

        # Process query
        query_info = self.query_processor.process_query(query)
        keywords = query_info['keywords']

        # Score sentences based on keyword overlap
        all_sentences = []
        for doc in documents:
            sentences = sent_tokenize(doc.page_content)
            for sentence in sentences:
                if len(sentence.strip()) > 20:  # Filter out very short sentences
                    score = self._score_sentence(sentence, keywords)
                    all_sentences.append((sentence, score))

        # Sort by score and select top sentences
        all_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [sent for sent, score in all_sentences[:3] if score > 0.1]

        if not top_sentences:
            # Fallback to first few sentences
            context_text = "\n\n".join([doc.page_content for doc in documents])
            return f"Based on the available information: {context_text[:500]}..."

        # Generate answer based on intent
        if intent == 'definition':
            return f"Based on the context: {' '.join(top_sentences)}"
        elif intent == 'how_to':
            return f"Here's how to proceed: {' '.join(top_sentences)}"
        elif intent == 'explanation':
            return f"The explanation is: {' '.join(top_sentences)}"
        elif intent == 'list':
            # Try to extract list items
            list_items = self._extract_list_items(top_sentences)
            if list_items:
                return "Here are the key points:\n" + "\n".join([f"• {item}" for item in list_items])
            else:
                return f"Key information: {' '.join(top_sentences)}"
        else:
            return f"Based on the available information: {' '.join(top_sentences)}"

    def _score_sentence(self, sentence: str, keywords: List[str]) -> float:
        """Score a sentence based on keyword overlap"""
        sentence_words = set(word_tokenize(sentence.lower()))
        keyword_matches = sum(1 for keyword in keywords if keyword in sentence_words)
        return keyword_matches / len(keywords) if keywords else 0

    def _extract_list_items(self, sentences: List[str]) -> List[str]:
        """Extract list items from sentences"""
        items = []
        for sentence in sentences:
            # Look for numbered items
            numbered_items = re.findall(r'\d+\.\s*([^.]+)', sentence)
            items.extend(numbered_items)

            # Look for bullet points
            bullet_items = re.findall(r'[•\-\*]\s*([^.]+)', sentence)
            items.extend(bullet_items)

        return [item.strip() for item in items if len(item.strip()) > 10]


class DocumentProcessor:
    """Advanced document processing with multiple format support"""

    def __init__(self):
        self.text_splitter = SmartTextSplitter()

    def process_document(self, content: str, doc_type: str = 'text') -> List[str]:
        """Process document based on type"""

        if doc_type == 'pdf':
            text = self._extract_pdf_text(content)
        elif doc_type == 'html':
            text = self._extract_html_text(content)
        elif doc_type == 'url':
            text = self._extract_url_content(content)
        else:
            text = content

        # Clean and preprocess text
        text = self._clean_text(text)

        # Split into chunks
        chunks = self.text_splitter.split_text(text)

        # Post-process chunks
        processed_chunks = []
        for chunk in chunks:
            if len(chunk.strip()) > 50:  # Filter out very short chunks
                processed_chunks.append(self._enhance_chunk(chunk))

        return processed_chunks

    def _extract_pdf_text(self, pdf_content: str) -> str:
        """Extract text from PDF content"""
        try:
            # Decode base64 if needed
            if pdf_content.startswith('data:application/pdf;base64,'):
                pdf_content = pdf_content.split(',')[1]

            pdf_bytes = base64.b64decode(pdf_content)
            pdf_file = io.BytesIO(pdf_bytes)

            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"

            return text
        except Exception as e:
            print(f"PDF extraction error: {e}")
            return ""

    def _extract_html_text(self, html_content: str) -> str:
        """Extract text from HTML content"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text and clean it
            text = soup.get_text()
            return text
        except Exception as e:
            print(f"HTML extraction error: {e}")
            return html_content

    def _extract_url_content(self, url: str) -> str:
        """Extract content from URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            return self._extract_html_text(response.text)
        except Exception as e:
            print(f"URL extraction error: {e}")
            return ""

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\[\]{}"\']', '', text)

        # Fix common issues
        text = text.replace('\n', ' ')
        text = text.replace('\t', ' ')

        return text.strip()

    def _enhance_chunk(self, chunk: str) -> str:
        """Enhance chunk with additional context"""
        # Add metadata markers for better retrieval
        sentences = sent_tokenize(chunk)

        # Add sentence count info
        if len(sentences) > 3:
            chunk = f"[Multi-sentence content] {chunk}"

        # Identify and mark important content
        if any(word in chunk.lower() for word in ['definition', 'define', 'means', 'is defined as']):
            chunk = f"[Definition] {chunk}"
        elif any(word in chunk.lower() for word in ['step', 'process', 'method', 'procedure']):
            chunk = f"[Process] {chunk}"
        elif any(word in chunk.lower() for word in ['example', 'for instance', 'such as']):
            chunk = f"[Example] {chunk}"

        return chunk


# --- Initialize Advanced Components ---
query_processor = QueryProcessor()
conversation_manager = ConversationManager()
document_processor = DocumentProcessor()
answer_generator = AnswerGenerator()

# Initialize Pinecone index
try:
    existing_indexes = [index.name for index in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    index = pc.Index(INDEX_NAME)
    advanced_retriever = AdvancedRetriever(index, embeddings)
except Exception as e:
    print(f"Pinecone initialization error: {e}")
    index = None
    advanced_retriever = None

# Initialize LLM (try multiple options)
llm = None
models_to_try = [
    "microsoft/DialoGPT-medium",
    "google/flan-t5-base",
    "HuggingFaceH4/zephyr-7b-beta"
]

for model_name in models_to_try:
    try:
        llm = HuggingFaceEndpoint(
            repo_id=model_name,
            temperature=0.7,
            max_new_tokens=512,
            huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN")
        )
        answer_generator.llm = llm
        print(f"Successfully initialized LLM: {model_name}")
        break
    except Exception as e:
        print(f"Failed to initialize {model_name}: {e}")
        continue

# --- Flask Routes ---

@app.route("/process", methods=["POST"])
def process_document():
    """Enhanced document processing endpoint"""
    try:
        data = request.get_json()
        content = data.get('text', '')
        doc_type = data.get('type', 'text')

        if not content:
            return jsonify({"error": "No content provided"}), 400

        # Process document using advanced processor
        chunks = document_processor.process_document(content, doc_type)

        if not chunks:
            return jsonify({"error": "No valid content extracted"}), 400

        # Create embeddings and store in vector database
        documents_added = 0
        if index and advanced_retriever:
            for i, chunk in enumerate(chunks):
                try:
                    # Generate embedding
                    embedding = embeddings.embed_query(chunk)

                    # Create unique ID
                    doc_id = f"doc_{int(time.time())}_{i}"

                    # Store in Pinecone
                    index.upsert([(doc_id, embedding, {"text": chunk})])
                    documents_added += 1

                except Exception as e:
                    print(f"Error processing chunk {i}: {e}")
                    continue

            # Add to BM25 retriever
            try:
                advanced_retriever.add_documents(chunks)
            except Exception as e:
                print(f"BM25 setup error: {e}")

        return jsonify({
            "message": f"Document processed successfully",
            "chunks_created": len(chunks),
            "documents_stored": documents_added,
            "preview": chunks[0][:200] + "..." if chunks else ""
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/query", methods=["POST"])
def enhanced_query():
    """Enhanced query processing endpoint"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        session_id = data.get('session_id', 'default')

        if not query:
            return jsonify({"error": "No query provided"}), 400

        # Process query
        query_info = query_processor.process_query(query)

        # Get conversation context
        context = conversation_manager.get_context(session_id)
        is_follow_up = conversation_manager.is_follow_up(session_id, query)

        # Retrieve relevant documents
        relevant_docs = []
        if advanced_retriever:
            try:
                results = advanced_retriever.hybrid_search(query, k=5)
                relevant_docs = [
                    Document(page_content=result['metadata']['text'])
                    for result in results
                ]
            except Exception as e:
                print(f"Retrieval error: {e}")

        # Generate answer
        if relevant_docs:
            answer = answer_generator.generate_answer(
                query=query,
                documents=relevant_docs,
                conversation_context=context,
                intent=query_info['intent']
            )
        else:
            answer = "I don't have enough information to answer your question. Please provide relevant documents first."

        # Store conversation
        conversation_manager.add_exchange(session_id, query, answer)

        return jsonify({
            "answer": answer,
            "query_info": query_info,
            "context_used": len(relevant_docs),
            "is_follow_up": is_follow_up,
            "sources": [doc.page_content[:100] + "..." for doc in relevant_docs[:3]]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/conversation/<session_id>", methods=["GET"])
def get_conversation_history(session_id):
    """Get conversation history for a session"""
    try:
        history = conversation_manager.conversations.get(session_id, [])
        return jsonify({
            "session_id": session_id,
            "conversation_count": len(history),
            "history": history
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/conversation/<session_id>", methods=["DELETE"])
def clear_conversation(session_id):
    """Clear conversation history for a session"""
    try:
        if session_id in conversation_manager.conversations:
            del conversation_manager.conversations[session_id]
        return jsonify({"message": f"Conversation {session_id} cleared"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/search", methods=["POST"])
def semantic_search():
    """Direct semantic search endpoint"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        k = data.get('k', 5)

        if not query or not advanced_retriever:
            return jsonify({"error": "Invalid query or retrieval system not initialized"}), 400

        # Perform hybrid search
        results = advanced_retriever.hybrid_search(query, k=k)

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "content": result['metadata']['text'],
                "score": result['score'],
                "source": result.get('source', 'unknown'),
                "preview": result['metadata']['text'][:200] + "..."
            })

        return jsonify({
            "query": query,
            "results": formatted_results,
            "total_found": len(formatted_results)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze", methods=["POST"])
def analyze_query():
    """Analyze query without generating answer"""
    try:
        data = request.get_json()
        query = data.get('query', '')

        if not query:
            return jsonify({"error": "No query provided"}), 400

        # Process and analyze query
        query_info = query_processor.process_query(query)

        return jsonify({
            "original_query": query,
            "analysis": query_info,
            "suggestions": {
                "expanded_queries": query_info['expanded'],
                "recommended_intent": query_info['intent'],
                "key_terms": query_info['keywords']
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    try:
        status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "embeddings": embeddings is not None,
                "vector_db": index is not None,
                "retriever": advanced_retriever is not None,
                "llm": llm is not None,
                "query_processor": query_processor is not None,
                "conversation_manager": conversation_manager is not None
            }
        }

        # Check vector database connection
        if index:
            try:
                stats = index.describe_index_stats()
                status["vector_db_stats"] = {
                    "total_vectors": stats.total_vector_count,
                    "dimension": stats.dimension
                }
            except Exception as e:
                status["vector_db_error"] = str(e)

        return jsonify(status)

    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


@app.route("/", methods=["GET"])
def index_page():
    """Main page with API documentation"""
    return jsonify({
        "message": "Advanced RAG System API",
        "version": "2.0",
        "endpoints": {
            "POST /process": "Process and store documents",
            "POST /query": "Ask questions with conversation context",
            "POST /search": "Direct semantic search",
            "POST /analyze": "Analyze query without answering",
            "GET /conversation/<session_id>": "Get conversation history",
            "DELETE /conversation/<session_id>": "Clear conversation history",
            "GET /health": "System health check"
        },
        "features": [
            "Smart text splitting with semantic awareness",
            "Hybrid search (semantic + keyword)",
            "Conversation context management",
            "Multi-format document support",
            "Query intent detection",
            "Advanced answer generation"
        ]
    })


# --- Utility Functions ---

def initialize_system():
    """Initialize system components"""
    print("Initializing Advanced RAG System...")

    # Check required environment variables
    required_vars = ['PINECONE_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print(f"Warning: Missing environment variables: {missing_vars}")

    # Initialize components
    components_status = {
        "embeddings": embeddings is not None,
        "vector_db": index is not None,
        "retriever": advanced_retriever is not None,
        "llm": llm is not None,
        "processors": all([
            query_processor is not None,
            conversation_manager is not None,
            document_processor is not None,
            answer_generator is not None
        ])
    }

    print("Component Status:")
    for component, status in components_status.items():
        print(f"  {component}: {'✓' if status else '✗'}")

    return all(components_status.values())


def cleanup_resources():
    """Cleanup resources on shutdown"""
    print("Cleaning up resources...")
    # Add any cleanup logic here
    pass


# --- Error Handlers ---

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


@app.errorhandler(400)
def bad_request(error):
    return jsonify({"error": "Bad request"}), 400


# --- Main Execution ---

if __name__ == "__main__":
    # Initialize system
    system_ready = initialize_system()

    if not system_ready:
        print("Warning: System not fully initialized. Some features may not work.")

    # Register cleanup function
    import atexit
    atexit.register(cleanup_resources)

    # Run the Flask app
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5000)),
        debug=os.getenv("DEBUG", "False").lower() == "true"
    )