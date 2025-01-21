import os
from typing import List
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma  # Updated import
from langchain_community.embeddings import HuggingFaceEmbeddings  # Update this import as well


# Configuration
GROQ_API_KEY = "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"
RESUME_DIR = "Resumes"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

class ResumeMatchingSystem:
    def __init__(self):     
        self.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="mixtral-8x7b-32768"
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        self.text_splitter = CharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        self.vector_store = None
        
    def load_resumes(self, resume_dir: str) -> List[str]:
        """Load and process all PDF resumes from the directory"""
        documents = []
        for filename in os.listdir(resume_dir):
            if filename.endswith('.pdf'):
                file_path = os.path.join(resume_dir, filename)
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
                
        # Split documents into chunks
        texts = self.text_splitter.split_documents(documents)
        return texts

    def create_vector_store(self, documents):
        """Create and persist the vector store"""
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        self.vector_store.persist()

    def setup_qa_chain(self):
        """Set up the retrieval QA chain with custom prompt"""
        template = """
        You are an AI recruiter analyzing resumes. Given the following job description, 
        identify the best candidate from the provided resume context.
        
        Job Description: {question}
        
        Context from resumes: {context}
        
        Please analyze the resumes and provide:
        1. The name of the best-matching candidate
        2. A detailed explanation of why they are the best fit
        3. Key qualifications that match the job requirements
        
        Response:
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 5}
            ),
            chain_type_kwargs={"prompt": prompt}
        )
        
        return chain

    def find_best_candidate(self, job_description: str):
        """Find the best candidate for a given job description"""
        qa_chain = self.setup_qa_chain()
        response = qa_chain.run(job_description)
        return response

def main():
    # Initialize the system
    matcher = ResumeMatchingSystem()
    
    # Load and process resumes
    print("Loading and processing resumes...")
    documents = matcher.load_resumes(RESUME_DIR)
    
    # Create vector store
    print("Creating vector store...")
    matcher.create_vector_store(documents)
    
    # Example usage
    job_description = """
Position: NLP Engineer
Location: [Insert Location]
Employment Type: [Full-time/Part-time/Contract]

Position Overview
We are seeking a talented and passionate NLP Engineer to join our team. The ideal candidate will have expertise in building, fine-tuning, and deploying natural language models and algorithms. You will work on cutting-edge projects involving text analytics, machine learning, and AI to develop innovative solutions that enhance our products and services.

Key Responsibilities
Design, develop, and implement NLP models and algorithms for text analysis, classification, and generation.
Work with large-scale datasets to preprocess and clean text data for model training.
Fine-tune pre-trained language models (e.g., BERT, GPT) to address specific use cases.
Create pipelines for text processing, entity recognition, sentiment analysis, and topic modeling.
Evaluate model performance and optimize for accuracy, efficiency, and scalability.
Collaborate with data scientists, engineers, and product teams to integrate NLP models into production systems.
Stay updated with the latest advancements in NLP and machine learning techniques.
Required Skills and Qualifications
Strong understanding of natural language processing concepts and techniques.
Proficiency in programming languages such as Python (e.g., NumPy, pandas) or R.
Experience with NLP libraries and frameworks (e.g., spaCy, NLTK, Hugging Face, TensorFlow, PyTorch).
Knowledge of machine learning and deep learning algorithms.
Familiarity with pre-trained models like BERT, GPT, T5, or similar.
Hands-on experience with text preprocessing (tokenization, stemming, lemmatization).
Strong analytical and problem-solving skills.
Preferred Qualifications
Experience in deploying NLP models in production environments.
Knowledge of cloud platforms (e.g., AWS, Google Cloud, Azure).
Understanding of linguistics or computational linguistics principles.
Familiarity with large language models and prompt engineering.
Background in speech-to-text or conversational AI systems.
Education and Experience
Bachelor’s or Master’s degree in Computer Science, Computational Linguistics, Data Science, or a related field.
1+ years of experience working with NLP projects or related fields.
Benefits
Competitive salary and bonuses.
Comprehensive health and wellness plans.
Remote work opportunities and flexible hours.
Professional development and training programs.
Inclusive and innovative work culture.


    """
    
    print("Finding best candidate...")
    result = matcher.find_best_candidate(job_description)
    print("\nBest Candidate Match:\n")
    print(result)

if __name__ == "__main__":
    main()