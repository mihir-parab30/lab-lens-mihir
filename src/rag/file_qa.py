#!/usr/bin/env python3
"""
File-based Q&A System using RAG
Allows users to upload text, PDF, or image files and ask questions about them
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional, List
import random

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.rag.rag_system import RAGSystem
from src.rag.document_processor import DocumentProcessor
from src.rag.vector_db import VectorDatabase, CHROMADB_AVAILABLE
from src.training.gemini_inference import GeminiInference
from src.utils.logging_config import get_logger
from src.utils.error_handling import ErrorHandler, safe_execute

logger = get_logger(__name__)

# Try to import medical utilities
try:
    from src.utils.medical_utils import get_medical_simplifier
    MEDICAL_UTILS_AVAILABLE = True
except ImportError:
    MEDICAL_UTILS_AVAILABLE = False
    logger.debug("Medical utilities not available")

# Try to import MedicalSummarizer for document summarization
try:
    import sys
    from pathlib import Path
    # Add project root to path to import summarizer
    project_root = Path(__file__).parent.parent.parent
    model_deployment_path = project_root / "model-deployment"
    
    if model_deployment_path.exists():
        # Add to path for import
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from model_deployment.deployment_pipeline.summarizer import MedicalSummarizer
        SUMMARIZER_AVAILABLE = True
        logger.debug("MedicalSummarizer available")
    else:
        SUMMARIZER_AVAILABLE = False
        logger.debug(f"MedicalSummarizer path not found: {model_deployment_path}")
except ImportError as e:
    SUMMARIZER_AVAILABLE = False
    logger.debug(f"MedicalSummarizer not available: {e}")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-generativeai not available for direct API calls")


class FileQA:
    """
    Q&A system that uses RAG to answer questions about uploaded documents
    Supports text files, PDFs, and images
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        use_biobert: bool = False,
        gemini_model: str = "gemini-2.0-flash-exp",
        gemini_api_key: Optional[str] = None,
        rag_k: int = 5,
        use_vector_db: bool = True,
        user_id: Optional[str] = None,
        collection_name: str = "file_qa_documents",
        simplify_medical_terms: bool = True
    ):
        """
        Initialize File Q&A system
        
        Args:
            embedding_model: Sentence transformer model for embeddings
            use_biobert: If True, use BioBERT for medical text (better for medical documents)
            gemini_model: Gemini model for answer generation
            gemini_api_key: Optional API key for Gemini (for image analysis)
            rag_k: Number of chunks to retrieve for context
            use_vector_db: Whether to use ChromaDB for persistent storage (default: True)
            user_id: Optional user ID for user-specific collections
            collection_name: Name of the ChromaDB collection
            simplify_medical_terms: If True, simplify medical terms in answers
        """
        self.error_handler = ErrorHandler(logger)
        self.rag_k = rag_k
        self.use_vector_db = use_vector_db and CHROMADB_AVAILABLE
        self.user_id = user_id
        self.simplify_medical_terms = simplify_medical_terms and MEDICAL_UTILS_AVAILABLE
        
        logger.info("Initializing File Q&A system...")
        
        # Initialize medical term simplifier if enabled
        self.term_simplifier = None
        if self.simplify_medical_terms:
            try:
                self.term_simplifier = get_medical_simplifier()
                logger.info("Medical term simplifier enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize medical term simplifier: {e}")
                self.simplify_medical_terms = False
        
        # Initialize document processor
        self.doc_processor = DocumentProcessor(gemini_api_key=gemini_api_key)
        
        # Initialize RAG system (without loading data)
        logger.info(f"Initializing RAG system (use_biobert={use_biobert})...")
        self.rag = RAGSystem(
            embedding_model=embedding_model,
            use_biobert=use_biobert
        )
        
        # Initialize vector database if enabled
        self.vector_db = None
        if self.use_vector_db:
            try:
                # Create embedding function wrapper for ChromaDB
                def embedding_fn(texts):
                    if not self.rag.embedding_model:
                        raise ValueError("Embedding model not initialized")
                    return self.rag.embedding_model.encode(texts, convert_to_numpy=True).tolist()
                
                # Use user-specific collection if user_id provided
                collection = f"{collection_name}_{user_id}" if user_id else collection_name
                
                self.vector_db = VectorDatabase(
                    persist_directory="models/vector_db",
                    collection_name=collection,
                    embedding_function=embedding_fn
                )
                logger.info(f"Vector database initialized: {collection}")
            except Exception as e:
                logger.warning(f"Failed to initialize vector database: {e}. Using in-memory storage.")
                self.use_vector_db = False
        
        # Initialize Gemini for answer generation
        logger.info("Initializing Gemini model for answer generation...")
        try:
            self.gemini = GeminiInference(model_name=gemini_model)
            logger.info("Gemini model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            self.gemini = None
        
        # Initialize MedicalSummarizer for document summarization (lazy loading)
        self.summarizer = None
        self.summarizer_available = SUMMARIZER_AVAILABLE
        if self.summarizer_available:
            logger.info("MedicalSummarizer will be loaded on first use")
    
    def load_file(self, file_path: str) -> Dict[str, any]:
        """
        Load and process a file (text, PDF, or image)
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing file: {file_path}")
        
        # Process the file
        document = self.doc_processor.process_file(file_path)
        
        # Load into RAG system
        self.rag.load_custom_documents([document])
        
        logger.info(f"File loaded and ready for Q&A: {document['file_name']}")
        
        return {
            'success': True,
            'file_name': document['file_name'],
            'file_type': document['file_type'],
            'text_length': len(document['text']),
            'num_chunks': len(self.rag.chunks),
            'preview': document['text'][:200] + "..." if len(document['text']) > 200 else document['text']
        }
    
    def load_text(self, text: str, source_name: str = "user_input") -> Dict[str, any]:
        """
        Load raw text content
        
        Args:
            text: Raw text content
            source_name: Name/identifier for the source
            
        Returns:
            Dictionary with loading results
        """
        logger.info(f"Processing text input: {source_name}")
        
        # Process the text
        document = self.doc_processor.process_text(text, source_name)
        
        # Load into RAG system
        self.rag.load_custom_documents([document])
        
        # Persist to vector database if enabled
        if self.use_vector_db and self.vector_db:
            try:
                if self.rag.embedding_model and self.rag.chunks:
                    # Use the embedding model directly
                    embeddings = self.rag.embedding_model.encode(
                        self.rag.chunks,
                        convert_to_numpy=True
                    )
                    embeddings_list = embeddings.tolist()
                    
                    self.vector_db.add_documents(
                        texts=self.rag.chunks,
                        embeddings=embeddings_list,
                        metadatas=self.rag.metadata
                    )
                    logger.info(f"Persisted {len(self.rag.chunks)} chunks to vector database")
            except Exception as e:
                logger.warning(f"Failed to persist to vector database: {e}")
        
        logger.info(f"Text loaded and ready for Q&A: {source_name}")
        
        return {
            'success': True,
            'source_name': source_name,
            'text_length': len(text),
            'num_chunks': len(self.rag.chunks),
            'preview': text[:200] + "..." if len(text) > 200 else text,
            'persisted': self.use_vector_db and self.vector_db is not None
        }
    
    def load_multiple_files(self, file_paths: List[str]) -> Dict[str, any]:
        """
        Load and process multiple files
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing {len(file_paths)} files")
        
        # Process all files and collect errors
        documents = []
        errors = []
        
        for file_path in file_paths:
            try:
                result = self.doc_processor.process_file(file_path)
                if result and result.get('text'):
                    documents.append(result)
                else:
                    errors.append(f"{Path(file_path).name}: No text extracted")
            except Exception as e:
                error_msg = f"{Path(file_path).name}: {str(e)}"
                errors.append(error_msg)
                logger.error(f"Failed to process {file_path}: {e}", exc_info=True)
        
        if not documents:
            error_summary = "; ".join(errors) if errors else "Unknown error"
            return {
                'success': False,
                'error': f'No documents could be processed. Errors: {error_summary}'
            }
        
        # Load into RAG system
        try:
            self.rag.load_custom_documents(documents)
        except Exception as e:
            logger.error(f"Failed to load documents into RAG: {e}", exc_info=True)
            return {
                'success': False,
                'error': f'Failed to load documents into RAG system: {str(e)}'
            }
        
        # Persist to vector database if enabled
        if self.use_vector_db and self.vector_db:
            try:
                # Generate embeddings for all chunks
                if self.rag.embedding_model and self.rag.chunks:
                    # Use the embedding model directly
                    embeddings = self.rag.embedding_model.encode(
                        self.rag.chunks,
                        convert_to_numpy=True
                    )
                    
                    # Convert embeddings to list of lists for ChromaDB
                    embeddings_list = embeddings.tolist()
                    
                    # Add to vector database
                    self.vector_db.add_documents(
                        texts=self.rag.chunks,
                        embeddings=embeddings_list,
                        metadatas=self.rag.metadata
                    )
                    logger.info(f"Persisted {len(self.rag.chunks)} chunks to vector database")
            except Exception as e:
                logger.warning(f"Failed to persist to vector database: {e}. Continuing with in-memory storage.")
        
        logger.info(f"Loaded {len(documents)} files with {len(self.rag.chunks)} total chunks")
        
        result = {
            'success': True,
            'num_files': len(documents),
            'num_chunks': len(self.rag.chunks),
            'files': [doc['file_name'] for doc in documents],
            'persisted': self.use_vector_db and self.vector_db is not None
        }
        
        if errors:
            result['warnings'] = errors
        
        return result
    
    @safe_execute("ask_question", logger, ErrorHandler(logger))
    def ask_question(self, question: str) -> Dict:
        """
        Answer a question about the loaded document(s)
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with answer and metadata
        """
        if not self.gemini:
            return {
                'answer': 'Sorry, the answer generation system is not available.',
                'error': 'Gemini model not initialized',
                'question': question
            }
        
        if not self.rag.chunks:
            return {
                'answer': 'No documents loaded. Please load a file first.',
                'error': 'No documents loaded',
                'question': question
            }
        
        logger.info(f"Processing question: {question[:100]}...")
        
        # Retrieve relevant chunks
        logger.info(f"Retrieving relevant information... (Total chunks available: {len(self.rag.chunks)})")
        
        # Try with lower threshold first, then progressively lower it
        retrieved_chunks = []
        thresholds = [0.3, 0.2, 0.1, 0.0]
        
        for threshold in thresholds:
            retrieved_chunks = self.rag.retrieve(
                query=question,
                k=self.rag_k,
                min_score=threshold
            )
            if retrieved_chunks:
                logger.info(f"Retrieved {len(retrieved_chunks)} chunks with threshold {threshold}")
                break
        
        # Determine if we have good document matches
        has_good_matches = False
        document_context = ""
        
        if retrieved_chunks:
            # Check if we have good quality matches (score > 0.2)
            good_chunks = [c for c in retrieved_chunks if c.get('score', 0) > 0.2]
            if good_chunks:
                has_good_matches = True
                # Build context from retrieved chunks
                context_parts = []
                for i, chunk_data in enumerate(good_chunks, 1):
                    chunk = chunk_data['chunk']
                    metadata = chunk_data['metadata']
                    source_info = f"[Source {i}: {metadata.get('document_name', 'Document')}]"
                    context_parts.append(f"{source_info}\n{chunk}")
                document_context = "\n\n".join(context_parts)
            else:
                # Low quality matches - use them but also allow general knowledge
                context_parts = []
                for i, chunk_data in enumerate(retrieved_chunks[:3], 1):
                    chunk = chunk_data['chunk']
                    metadata = chunk_data['metadata']
                    source_info = f"[Source {i}: {metadata.get('document_name', 'Document')}]"
                    context_parts.append(f"{source_info}\n{chunk}")
                document_context = "\n\n".join(context_parts)
        
        # If no chunks found, try to get any chunks for context
        if not retrieved_chunks and len(self.rag.chunks) > 0:
            # Get a few random chunks for context about the document
            import random
            sample_size = min(3, len(self.rag.chunks))
            sample_indices = random.sample(range(len(self.rag.chunks)), sample_size)
            context_parts = []
            for idx in sample_indices:
                chunk = self.rag.chunks[idx]
                metadata = self.rag.metadata[idx]
                source_info = f"[From document: {metadata.get('document_name', 'Document')}]"
                context_parts.append(f"{source_info}\n{chunk}")
            document_context = "\n\n".join(context_parts)
            logger.info("Using sample document chunks for context (no direct matches found)")
        
        # Generate answer using Gemini with hybrid approach
        logger.info("Generating answer using Gemini (hybrid: document + general knowledge)...")
        try:
            # Create a prompt that uses document if available, but allows general knowledge
            if document_context and has_good_matches:
                # Strong document match - prioritize document
                prompt = f"""You are a helpful medical assistant. Answer the patient's question based on the information from their medical report/document below. If the document doesn't contain enough information, you may supplement with your general medical knowledge, but clearly indicate what comes from the document vs. general knowledge.

DOCUMENT CONTENT:
{document_context}

PATIENT'S QUESTION: {question}

ANSWER:"""
            elif document_context:
                # Weak document match - use document as context but allow general knowledge
                prompt = f"""You are a helpful medical assistant. The patient has asked a question about their medical report. The document content below may or may not directly answer the question. Please:
1. First try to answer based on the document if relevant
2. If the document doesn't contain the answer, use your general medical knowledge to help
3. Clearly indicate when you're using general knowledge vs. document information

DOCUMENT CONTENT (may be partially relevant):
{document_context}

PATIENT'S QUESTION: {question}

ANSWER:"""
            else:
                # No document context - use general knowledge
                prompt = f"""You are a helpful medical assistant. The patient has asked a medical question. Please answer using your general medical knowledge. Be clear, accurate, and helpful.

PATIENT'S QUESTION: {question}

ANSWER:"""
            
            # Generate response using Gemini
            if GEMINI_AVAILABLE:
                # Use direct API call for more control
                model = genai.GenerativeModel(self.gemini.model.model_name)
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,  # Slightly higher for more natural responses
                        max_output_tokens=2048,
                    )
                )
                answer = response.text.strip()
            else:
                # Fallback to existing method
                if document_context:
                    answer = self.gemini.model.answer_question(question, document_context)
                else:
                    # No context - create a simple prompt
                    simple_prompt = f"Answer this medical question: {question}"
                    model = self.gemini.model.model
                    response = model.generate_content(simple_prompt)
                    answer = response.text.strip()
            
            logger.info("Answer generated successfully")
            
            # Apply medical term simplification if enabled
            if self.simplify_medical_terms and self.term_simplifier and answer:
                try:
                    original_answer = answer
                    answer = self.term_simplifier.simplify_text(answer, aggressive=False)
                    logger.debug("Applied medical term simplification to answer")
                except Exception as e:
                    logger.warning(f"Failed to simplify medical terms: {e}")
                    # Continue with original answer
            
            # Determine answer source
            if has_good_matches:
                answer_source = "document"
            elif retrieved_chunks:
                answer_source = "document_and_knowledge"
            else:
                answer_source = "general_knowledge"
            
            return {
                'answer': answer,
                'sources': retrieved_chunks if retrieved_chunks else [],
                'question': question,
                'num_sources': len(retrieved_chunks) if retrieved_chunks else 0,
                'answer_source': answer_source  # Indicates where answer came from
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                'answer': f"I encountered an error while generating an answer: {e}. Please try rephrasing your question.",
                'error': str(e),
                'sources': retrieved_chunks,
                'question': question
            }
    
    def _get_summarizer(self):
        """Lazy load MedicalSummarizer"""
        if not self.summarizer_available:
            return None
        
        if self.summarizer is None:
            try:
                logger.info("Loading MedicalSummarizer for document summarization...")
                self.summarizer = MedicalSummarizer(use_gpu=False)  # Use CPU for web app compatibility
                logger.info("MedicalSummarizer loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load MedicalSummarizer: {e}")
                self.summarizer_available = False
                return None
        
        return self.summarizer
    
    def summarize_document(self, text: Optional[str] = None) -> Dict:
        """
        Generate a summary of the loaded document(s) using MedicalSummarizer
        
        Args:
            text: Optional text to summarize. If None, uses all loaded document chunks.
            
        Returns:
            Dictionary with summary and metadata
        """
        summarizer = self._get_summarizer()
        if not summarizer:
            return {
                'success': False,
                'error': 'MedicalSummarizer not available',
                'summary': None
            }
        
        # Get text to summarize
        if text is None:
            if not self.rag.chunks:
                return {
                    'success': False,
                    'error': 'No documents loaded to summarize',
                    'summary': None
                }
            # Combine all chunks
            text = "\n\n".join(self.rag.chunks)
        
        if not text or not text.strip():
            return {
                'success': False,
                'error': 'No text content to summarize',
                'summary': None
            }
        
        try:
            logger.info("Generating document summary using MedicalSummarizer...")
            result = summarizer.generate_summary(text)
            
            return {
                'success': True,
                'summary': result.get('final_summary', ''),
                'raw_summary': result.get('raw_bart_summary', ''),
                'extracted_data': result.get('extracted_data', {}),
                'error': None
            }
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {
                'success': False,
                'error': str(e),
                'summary': None
            }

