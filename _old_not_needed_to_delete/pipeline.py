import ollama
import random
from datasets import load_dataset
from typing import List, Dict, Tuple, Optional
import json
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

class OpenTopicRAGPipeline:
    def __init__(self, model_name="qwen3:1.7b", embedding_model="Qwen/Qwen3-Embedding-0.6B", use_ollama=True):
        """
        Initialize the Open Topic RAG pipeline with Qwen3 models.
        
        Args:
            model_name: The Ollama model to use for generation (e.g., "qwen3:1.7b")
            embedding_model: The HuggingFace model to use for embeddings
            use_ollama: Whether to use Ollama (set False for testing without LLM)
        """
        self.model_name = model_name
        self.embedding_model_name = embedding_model
        self.documents = []
        self.document_embeddings = []
        self.use_ollama = use_ollama
        self.user_preferences = ""
        self.discovered_topics = []  # Store all discovered topics
        
        # Initialize Qwen embedding model
        print(f"Loading embedding model: {embedding_model}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(embedding_model, trust_remote_code=True)
            self.embedding_model = AutoModel.from_pretrained(
                embedding_model, 
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            # Move to GPU if available, otherwise CPU
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.embedding_model = self.embedding_model.to(self.device)
            self.embedding_model.eval()
            print(f"✓ Embedding model loaded successfully on {self.device}")
        except Exception as e:
            print(f"⚠️ Error loading embedding model: {e}")
            print("Falling back to TF-IDF for embeddings")
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.embedding_model = None
            self.vectorizer = TfidfVectorizer(max_features=1000)
        
        # Check if Ollama is available
        if use_ollama:
            try:
                ollama.list()
                print(f"✓ Ollama is connected. Using model: {model_name}")
                self.use_ollama = True
            except Exception as e:
                print(f"⚠️ Error connecting to Ollama: {e}")
                print("\n" + "="*60)
                print("OLLAMA SETUP INSTRUCTIONS:")
                print("="*60)
                print("1. Install Ollama:")
                print("   Mac: brew install ollama")
                print("   Linux: curl -fsSL https://ollama.ai/install.sh | sh")
                print("   Windows: Download from https://ollama.ai/download")
                print("\n2. Start Ollama (in a new terminal):")
                print("   ollama serve")
                print("\n3. Pull a model (in another terminal):")
                print("   ollama pull qwen3:1.7b")
                print("="*60)
                print("\n⚠️ Running in DEMO MODE without Ollama")
                self.use_ollama = False
        else:
            print("📌 Running in DEMO MODE without Ollama")
            self.use_ollama = False
    
    def get_user_preferences(self):
        """
        Get user preferences for topic generation at runtime.
        """
        print("\n" + "="*60)
        print("USER PREFERENCE CONFIGURATION")
        print("="*60)
        print("Please specify your preferences for topic generation.")
        print("Examples:")
        print("  - 'Avoid topics about politics and wars'")
        print("  - 'Focus on technology and innovation'")
        print("  - 'No celebrity or entertainment topics'")
        print("  - 'Prefer business and entrepreneurship topics'")
        print("  - Press Enter for no specific preferences")
        print("-"*60)
        
        self.user_preferences = input("Enter your preference: ").strip()
        
        if not self.user_preferences:
            self.user_preferences = "No specific restrictions - explore all topics freely"
            print("✓ No preferences set - will explore all topics")
        else:
            print(f"✓ Preference set: {self.user_preferences}")
        
        return self.user_preferences
    
    def load_dataset_flexible(self, source="tweet", sample_size=200):
        """
        Load dataset from various sources without assuming labels.
        
        Args:
            source: Type of dataset ("tweet", "news", "custom", etc.)
            sample_size: Number of samples to use
        """
        print(f"\nLoading dataset (source={source}, sample_size={sample_size})...")
        
        if source == "tweet":
            try:
                # Try to load tweet dataset
                data_files = {
                    "train": "https://huggingface.co/datasets/cardiffnlp/tweet_topic_single/resolve/main/tweet_topic_single_train.parquet",
                    "test": "https://huggingface.co/datasets/cardiffnlp/tweet_topic_single/resolve/main/tweet_topic_single_test.parquet",
                }
                dataset = load_dataset("parquet", data_files=data_files["test"], split="train")
                
                # Load without labels - just use text
                indices = random.sample(range(len(dataset)), min(sample_size, len(dataset)))
                self.documents = []
                for idx in indices:
                    self.documents.append({
                        'text': dataset[idx]['text'],
                        'id': idx,
                        'discovered_topic': None  # Will be filled by LLM
                    })
                    
            except Exception as e:
                print(f"Could not load tweet dataset: {e}")
                print("Using sample data...")
                self._create_sample_data(sample_size)
        else:
            self._create_sample_data(sample_size)
        
        print(f"✓ Loaded {len(self.documents)} documents")
        return self.documents
    
    def _create_sample_data(self, sample_size):
        """Create diverse sample data without predefined labels."""
        sample_texts = [
            "Just launched my new startup! The journey of building something from scratch is incredible. #entrepreneur",
            "The new AI models are getting scary good at understanding context. We're living in the future!",
            "Climate change protests happening downtown. We need action now, not more empty promises.",
            "Made the most amazing homemade pasta today. The secret is in the sauce!",
            "Bitcoin just hit a new milestone. Is this the future of finance or another bubble?",
            "Working from home has completely changed my productivity. Best decision ever!",
            "The new Marvel movie was absolutely mind-blowing. That ending though!",
            "Just finished a 10K run. Feeling stronger every day! #fitness",
            "Teaching my kids about saving money. It's never too early to start financial literacy.",
            "The stock market volatility is insane right now. Time to reassess portfolios.",
            "Space exploration is accelerating. Can't wait to see humans on Mars!",
            "Local farmers market has the best organic produce. Supporting local businesses feels good.",
            "Machine learning is revolutionizing healthcare diagnostics. The accuracy is unprecedented.",
            "Fashion week highlights: sustainable fashion is finally taking center stage.",
            "Gaming industry revenues now exceed Hollywood. The cultural shift is real.",
            "Remote team management requires completely different skills. Learning every day.",
            "Quantum computing breakthrough announced. This changes everything!",
            "Mental health awareness is so important. Take care of yourselves, friends.",
            "Supply chain innovations are making logistics faster and greener.",
            "The housing market is absolutely wild right now. Prices just keep climbing.",
        ]
        
        # Duplicate and shuffle to reach sample size
        all_texts = sample_texts * (sample_size // len(sample_texts) + 1)
        random.shuffle(all_texts)
        
        self.documents = []
        for i, text in enumerate(all_texts[:sample_size]):
            self.documents.append({
                'text': text,
                'id': i,
                'discovered_topic': None
            })
    
    def generate_and_extract_topic(self, num_documents=5) -> Tuple[str, List[Dict]]:
        """
        Step 1: Pick random documents and generate a topic based on user preferences.
        The LLM will discover patterns and extract topics freely.
        """
        print("\n" + "="*50)
        print("STEP 1: Open Topic Discovery & Generation")
        print("="*50)
        
        with tqdm(total=4, desc="Topic Discovery", bar_format='{l_bar}{bar:30}{r_bar}') as pbar:
            
            # Pick random documents
            pbar.set_description("Selecting random documents")
            selected_docs = random.sample(self.documents, min(num_documents, len(self.documents)))
            time.sleep(0.5)
            pbar.update(1)
            
            # Create prompt for open topic discovery
            pbar.set_description("Preparing for topic discovery")
            doc_texts = "\n".join([f"Document {i+1}: {doc['text']}" 
                                  for i, doc in enumerate(selected_docs)])
            pbar.update(1)
            
            if self.use_ollama:
                pbar.set_description("Discovering topics with LLM")
                
                # First, discover what topics are present
                discovery_prompt = f"""Analyze these documents and identify the main topics or themes present. 
Be creative and specific in your topic identification.

Documents:
{doc_texts}

User Preference: {self.user_preferences}

First, identify ALL potential topics you can find in these documents (be comprehensive).
Then, select ONE specific topic that:
1. Appears in multiple documents or is most interesting
2. Respects the user preference
3. Is specific enough to be meaningful (not too broad like "life" or "things")

Output format:
DISCOVERED TOPICS: [List all topics found]
SELECTED TOPIC: [The one specific topic you choose]
REASON: [Why you selected this topic]
"""
                
                response = ollama.generate(model=self.model_name, prompt=discovery_prompt)
                full_response = response['response'].strip()
                
                # Extract the selected topic
                topic = self._extract_topic_from_response(full_response)
                
                # Store discovered topics for analysis
                self._store_discovered_topics(full_response)
                
            else:
                # Demo mode: generate a random topic
                pbar.set_description("Generating topic (Demo mode)")
                demo_topics = [
                    "artificial intelligence in everyday life",
                    "sustainable living practices",
                    "remote work culture transformation",
                    "personal finance strategies",
                    "health and wellness trends",
                    "innovative startup ideas",
                    "digital transformation in traditional industries"
                ]
                topic = random.choice(demo_topics)
                time.sleep(1)
            
            pbar.update(1)
            
            # Now label the selected documents with the discovered topic
            pbar.set_description("Labeling documents with discovered topic")
            for doc in selected_docs:
                doc['discovered_topic'] = topic
            pbar.update(1)
            
            pbar.set_description("Topic discovery complete")
        
        print(f"\n✓ Discovered and selected topic: '{topic}'")
        print(f"  Applied to {len(selected_docs)} documents")
        
        return topic, selected_docs
    
    def _extract_topic_from_response(self, response: str) -> str:
        """Extract the selected topic from LLM response."""
        lines = response.split('\n')
        for line in lines:
            if 'SELECTED TOPIC:' in line:
                return line.split('SELECTED TOPIC:')[1].strip()
        
        # Fallback: try to find any topic-like phrase
        for line in lines:
            line = line.strip()
            if len(line) > 5 and len(line) < 100 and not line.startswith('Document'):
                return line
        
        return "general discussion topics"
    
    def _store_discovered_topics(self, response: str):
        """Store all discovered topics for later analysis."""
        lines = response.split('\n')
        for line in lines:
            if 'DISCOVERED TOPICS:' in line:
                topics_str = line.split('DISCOVERED TOPICS:')[1].strip()
                # Parse the list of topics
                topics = [t.strip() for t in topics_str.split(',')]
                self.discovered_topics.extend(topics)
                break
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for texts using Qwen3 embedding model.
        """
        if self.embedding_model is not None:
            # Use Qwen3 embedding model
            embeddings = []
            
            # Process in batches for efficiency with progress bar
            batch_size = 8
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            with tqdm(total=total_batches, desc="Creating embeddings", leave=False, 
                     bar_format='{l_bar}{bar:30}{r_bar}') as pbar:
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    
                    # Tokenize and encode
                    with torch.no_grad():
                        encoded_input = self.tokenizer(
                            batch_texts, 
                            padding=True, 
                            truncation=True, 
                            max_length=512,
                            return_tensors='pt'
                        ).to(self.device)
                        
                        # Get embeddings
                        model_output = self.embedding_model(**encoded_input)
                        
                        # Use mean pooling to get sentence embeddings
                        attention_mask = encoded_input['attention_mask']
                        token_embeddings = model_output.last_hidden_state
                        
                        # Expand attention mask for broadcasting
                        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                        
                        # Apply mask and mean pooling
                        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                        batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
                        
                        embeddings.extend(batch_embeddings)
                    
                    pbar.update(1)
            
            return np.array(embeddings)
        else:
            # Fallback to TF-IDF
            if len(self.document_embeddings) == 0:
                all_texts = [doc['text'] for doc in self.documents]
                self.document_embeddings = self.vectorizer.fit_transform(all_texts)
            return self.vectorizer.transform(texts)
    
    def retrieve_relevant_documents(self, topic: str, selected_docs: List[Dict], top_k=10) -> List[Dict]:
        """
        Step 2: RAG retrieval - find documents relevant to the discovered topic.
        """
        print("\n" + "="*50)
        print("STEP 2: RAG Retrieval for Discovered Topic")
        print("="*50)
        
        with tqdm(total=4, desc="RAG Retrieval", bar_format='{l_bar}{bar:30}{r_bar}') as pbar:
            
            # Create embeddings for all documents if not already done
            pbar.set_description("Checking document embeddings")
            if len(self.document_embeddings) == 0:
                pbar.set_description("Creating document embeddings")
                all_texts = [doc['text'] for doc in self.documents]
                self.document_embeddings = self.create_embeddings(all_texts)
                print(f"  ✓ Created embeddings with shape: {self.document_embeddings.shape}")
            pbar.update(1)
            
            # Create embedding for the topic
            pbar.set_description(f"Embedding topic: '{topic[:30]}...'")
            topic_embedding = self.create_embeddings([topic])
            pbar.update(1)
            
            # Calculate similarities
            pbar.set_description("Calculating similarities")
            similarities = cosine_similarity(topic_embedding, self.document_embeddings)[0]
            pbar.update(1)
            
            # Get top-k most similar documents
            pbar.set_description("Selecting top documents")
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            retrieved_docs = []
            for idx in top_indices:
                if similarities[idx] > 0:
                    doc = self.documents[idx].copy()
                    doc['similarity_score'] = float(similarities[idx])
                    retrieved_docs.append(doc)
            
            # Include originally selected documents
            selected_ids = {doc['id'] for doc in selected_docs}
            for doc in selected_docs:
                if doc['id'] not in {d['id'] for d in retrieved_docs}:
                    doc_copy = doc.copy()
                    doc_copy['similarity_score'] = 1.0
                    retrieved_docs.append(doc_copy)
            
            pbar.update(1)
            pbar.set_description("Retrieval complete")
        
        print(f"✓ Retrieved {len(retrieved_docs)} relevant documents")
        top_scores = [round(d['similarity_score'], 3) for d in retrieved_docs[:3]]
        print(f"  Top similarity scores: {top_scores}")
        return retrieved_docs
    
    def analyze_and_label_documents(self, topic: str, retrieved_docs: List[Dict], selected_docs: List[Dict]) -> Dict:
        """
        Step 3: Analyze retrieved documents and assign discovered topics/themes.
        No predefined labels - let LLM discover and assign topics.
        """
        print("\n" + "="*50)
        print("STEP 3: Open Analysis & Topic Assignment")
        print("="*50)
        
        # Prepare documents to analyze
        docs_to_analyze = retrieved_docs[:5]
        
        results = []
        
        with tqdm(total=len(docs_to_analyze), desc="Analyzing documents", 
                 bar_format='{l_bar}{bar:30}{r_bar}') as pbar:
            
            for doc in docs_to_analyze:
                pbar.set_description(f"Analyzing document {pbar.n+1}/{len(docs_to_analyze)}")
                
                if self.use_ollama:
                    # Let LLM freely analyze and assign topics
                    analysis_prompt = f"""Analyze this document and identify its main topics/themes.

Context: We're exploring documents related to "{topic}"
User Preference: {self.user_preferences}

Document: {doc['text']}

Provide:
1. MAIN TOPIC: The primary topic of this document
2. RELATED THEMES: Other themes present (comma-separated)
3. RELEVANCE SCORE: How relevant is this to "{topic}" (0-10)
4. SENTIMENT: The overall sentiment (positive/negative/neutral)
5. KEY ENTITIES: Important entities mentioned (people, companies, concepts)

Format your response exactly as:
MAIN TOPIC: [topic]
RELATED THEMES: [theme1, theme2, ...]
RELEVANCE SCORE: [0-10]
SENTIMENT: [positive/negative/neutral]
KEY ENTITIES: [entity1, entity2, ...]
"""
                    
                    response = ollama.generate(model=self.model_name, prompt=analysis_prompt)
                    analysis = self._parse_analysis(response['response'])
                    
                else:
                    # Demo mode: simple analysis
                    analysis = {
                        'main_topic': topic if random.random() > 0.5 else "general discussion",
                        'related_themes': ["technology", "innovation", "society"],
                        'relevance_score': random.uniform(3, 10),
                        'sentiment': random.choice(["positive", "negative", "neutral"]),
                        'key_entities': ["AI", "startup", "community"]
                    }
                    time.sleep(0.5)
                
                results.append({
                    'text': doc['text'][:200] + "...",
                    'discovered_topic': analysis.get('main_topic', 'unclassified'),
                    'related_themes': analysis.get('related_themes', []),
                    'relevance_score': analysis.get('relevance_score', 0),
                    'sentiment': analysis.get('sentiment', 'neutral'),
                    'key_entities': analysis.get('key_entities', []),
                    'similarity_score': doc.get('similarity_score', 0),
                    'is_selected': doc['id'] in {d['id'] for d in selected_docs}
                })
                
                pbar.update(1)
            
            pbar.set_description("Analysis complete")
        
        return {
            'primary_topic': topic,
            'user_preferences': self.user_preferences,
            'results': results
        }
    
    def _parse_analysis(self, response: str) -> Dict:
        """Parse the LLM's analysis response."""
        analysis = {
            'main_topic': 'unclassified',
            'related_themes': [],
            'relevance_score': 0,
            'sentiment': 'neutral',
            'key_entities': []
        }
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if 'MAIN TOPIC:' in line:
                analysis['main_topic'] = line.split('MAIN TOPIC:')[1].strip()
            elif 'RELATED THEMES:' in line:
                themes = line.split('RELATED THEMES:')[1].strip()
                analysis['related_themes'] = [t.strip() for t in themes.split(',')]
            elif 'RELEVANCE SCORE:' in line:
                try:
                    score = line.split('RELEVANCE SCORE:')[1].strip()
                    analysis['relevance_score'] = float(score)
                except:
                    analysis['relevance_score'] = 5
            elif 'SENTIMENT:' in line:
                analysis['sentiment'] = line.split('SENTIMENT:')[1].strip().lower()
            elif 'KEY ENTITIES:' in line:
                entities = line.split('KEY ENTITIES:')[1].strip()
                analysis['key_entities'] = [e.strip() for e in entities.split(',')]
        
        return analysis
    
    def cleanup_dataset(self, num_to_remove=10):
        """
        Step 4: Remove processed documents from the pool.
        """
        print("\n" + "="*50)
        print("STEP 4: Cleanup - Removing Processed Documents")
        print("="*50)
        
        with tqdm(total=1, desc="Cleanup", bar_format='{l_bar}{bar:30}{r_bar}') as pbar:
            pbar.set_description("Removing processed documents")
            
            if len(self.documents) > num_to_remove:
                indices_to_remove = random.sample(range(len(self.documents)), num_to_remove)
                indices_to_remove.sort(reverse=True)
                
                for idx in indices_to_remove:
                    self.documents.pop(idx)
                
                self.document_embeddings = []
                
                pbar.update(1)
                pbar.set_description("Cleanup complete")
                print(f"✓ Removed {num_to_remove} documents. {len(self.documents)} documents remaining.")
            else:
                pbar.update(1)
                print(f"⚠️ Not enough documents to remove. {len(self.documents)} documents remaining.")
    
    def run_pipeline(self, iterations=2):
        """
        Run the complete open topic discovery pipeline.
        """
        print("\n" + "🚀 STARTING OPEN TOPIC RAG PIPELINE" + "\n" + "="*50)
        
        # Get user preferences once at the start
        self.get_user_preferences()
        
        all_results = []
        
        for i in range(iterations):
            print(f"\n📍 ITERATION {i+1}/{iterations}")
            
            # Step 1: Generate and extract topic
            topic, selected_docs = self.generate_and_extract_topic()
            
            # Step 2: Retrieve relevant documents
            retrieved_docs = self.retrieve_relevant_documents(topic, selected_docs)
            
            # Step 3: Analyze and label documents
            iteration_results = self.analyze_and_label_documents(topic, retrieved_docs, selected_docs)
            all_results.append(iteration_results)
            
            # Step 4: Cleanup
            self.cleanup_dataset()
            
            # Print results for this iteration
            print("\n📊 Results for this iteration:")
            print(f"Primary Topic Discovered: {iteration_results['primary_topic']}")
            print(f"User Preferences Applied: {iteration_results['user_preferences']}")
            print("\nDocument Analysis:")
            for j, result in enumerate(iteration_results['results'][:3], 1):
                print(f"\n{j}. Document excerpt: {result['text'][:80]}...")
                print(f"   Discovered Topic: {result['discovered_topic']}")
                print(f"   Related Themes: {', '.join(result['related_themes'][:3])}")
                print(f"   Relevance Score: {result['relevance_score']:.1f}/10")
                print(f"   Sentiment: {result['sentiment']}")
                print(f"   Key Entities: {', '.join(result['key_entities'][:3])}")
        
        return all_results
    
    def summarize_discoveries(self, results: List[Dict]):
        """
        Summarize all discovered topics and patterns across iterations.
        """
        print("\n" + "="*50)
        print("📈 TOPIC DISCOVERY SUMMARY")
        print("="*50)
        
        all_topics = []
        all_themes = []
        sentiments = []
        
        for iteration in results:
            all_topics.append(iteration['primary_topic'])
            for result in iteration['results']:
                all_themes.extend(result['related_themes'])
                sentiments.append(result['sentiment'])
        
        # Count frequencies
        from collections import Counter
        topic_counts = Counter(all_topics)
        theme_counts = Counter(all_themes)
        sentiment_counts = Counter(sentiments)
        
        print("\n🎯 Primary Topics Discovered:")
        for topic, count in topic_counts.most_common():
            print(f"  - {topic}: {count} iteration(s)")
        
        print("\n🔍 Top Related Themes:")
        for theme, count in theme_counts.most_common(10):
            print(f"  - {theme}: {count} occurrences")
        
        print("\n😊 Sentiment Distribution:")
        total_sentiments = sum(sentiment_counts.values())
        for sentiment, count in sentiment_counts.items():
            percentage = (count / total_sentiments) * 100
            print(f"  - {sentiment}: {percentage:.1f}%")


def main():
    """
    Main function to run the open topic discovery pipeline.
    """
    # Initialize pipeline
    print("="*60)
    print("🔮 OPEN TOPIC RAG PIPELINE WITH QWEN3")
    print("="*60)
    print("\nThis pipeline will:")
    print("1. Accept your topic preferences")
    print("2. Discover topics from documents without predefined labels")
    print("3. Use RAG to find related documents")
    print("4. Analyze and extract themes freely")
    print("\n📋 Requirements:")
    print("   pip install ollama datasets transformers torch scikit-learn tqdm")
    print("\n   Make sure Ollama is running: ollama serve")
    print("   Pull Qwen3 model: ollama pull qwen3:1.7b")
    print("-" * 60)
    
    # Initialize pipeline
    pipeline = OpenTopicRAGPipeline(
        model_name="qwen3:1.7b",  # Qwen3 1.7b model for generation via Ollama
        embedding_model="Qwen/Qwen3-Embedding-0.6B"  # Qwen3 embedding model
    )
    
    # Load dataset without assuming labels
    pipeline.load_dataset_flexible(source="tweet", sample_size=400)
    
    # Run pipeline
    results = pipeline.run_pipeline(iterations=2)
    
    # Summarize discoveries
    pipeline.summarize_discoveries(results)
    
    # Final summary
    print("\n" + "="*50)
    print("🏁 PIPELINE COMPLETE!")
    print("="*50)
    print(f"Processed {len(results)} iterations")
    print(f"Discovered {len(set([r['primary_topic'] for r in results]))} unique primary topics")
    
    # Show all unique discovered topics
    unique_topics = set()
    for result in results:
        unique_topics.add(result['primary_topic'])
        for doc_result in result['results']:
            unique_topics.add(doc_result['discovered_topic'])
    
    print(f"\n🌟 All Unique Topics Discovered:")
    for topic in sorted(unique_topics):
        if topic and topic != 'unclassified':
            print(f"  - {topic}")


if __name__ == "__main__":
    main()