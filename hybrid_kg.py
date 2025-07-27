from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to Neo4j
graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="RezinixKnowledgeGraph"  # Change this to your Neo4j password
)

# Step 1: Hybrid chunking function
def hybrid_chunk(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    chunks = splitter.split_text(text)
    chunks = [c for c in chunks if len(c.split()) > 5]
    return chunks

# 📄 Step 2: Sample documents
documents = [
    {
        "title": "AI in Healthcare",
        "text": """
AI helps doctors analyze medical images.
AI systems assist in diagnosis.
Machine learning predicts patient risk.
Robots support surgeons in operations.
Natural language processing extracts data from notes.
AI reduces human error in medication.
Wearables track patient health in real time.
AI chatbots answer patient questions.
Predictive analytics optimizes hospital resource use.
Healthcare AI saves lives and reduces cost.
"""
    },
    {
        "title": "AI in Finance",
        "text": """
AI detects fraudulent transactions.
Algorithmic trading makes split-second decisions.
Credit scoring models evaluate risk.
Chatbots handle customer inquiries.
Robo-advisors build personalized portfolios.
AI forecasts market trends.
Document processing automates compliance checks.
Sentiment analysis tracks news impact on stocks.
AI helps reduce operational costs.
Banks innovate with AI to serve customers better.
"""
    },
    {
        "title": "AI in Education",
        "text": """
AI tutors give personalized lessons.
Automated grading saves teachers time.
Chatbots answer student questions instantly.
Adaptive learning adjusts to student pace.
AI analyzes student performance data.
Plagiarism detection keeps work original.
Virtual classrooms use AI for engagement.
Speech recognition helps language learners.
AI supports special needs education.
Education AI makes learning more accessible.
"""
    },
    {
        "title": "AI in Retail",
        "text": """
AI predicts product demand.
Recommendation engines suggest products to customers.
Chatbots answer shopper questions.
Computer vision manages inventory in real time.
AI personalizes marketing campaigns.
Price optimization tools adjust dynamically.
Voice assistants help customers shop hands-free.
Sentiment analysis monitors reviews.
Supply chain optimization reduces stockouts.
Retail AI increases customer loyalty.
"""
    },
    
    {
        "title": "AI in Healthcare - Diagnosis",
        "text": """
AI models help detect diseases from X-rays and MRIs.
Early detection improves patient outcomes.
Deep learning outperforms traditional methods.
Doctors use AI as decision support, not replacement.
"""
    },
    {
        "title": "AI in Healthcare - Patient Monitoring",
        "text": """
Wearables and IoT devices track heart rate and glucose.
AI alerts doctors to abnormal patterns.
Remote monitoring reduces hospital visits.
Patients get more personalized care.
"""
    },
    {
        "title": "AI in Healthcare - Drug Discovery",
        "text": """
AI analyzes molecular structures to suggest new drugs.
Simulations speed up research and cut costs.
Machine learning predicts drug side effects early.
This reduces failure rates in clinical trials.
"""
    },
    {
        "title": "AI in Finance - Fraud Detection",
        "text": """
AI models spot unusual spending behaviors.
Banks stop fraud faster than manual checks.
Real-time detection reduces customer impact.
Continuous learning adapts to new fraud techniques.
"""
    },
    {
        "title": "AI in Finance - Algorithmic Trading",
        "text": """
Trading bots analyze market data in milliseconds.
AI executes trades automatically at the best price.
High-frequency trading increases market liquidity.
Regulations aim to keep AI trading fair.
"""
    },
    {
        "title": "AI in Finance - Customer Service",
        "text": """
Chatbots handle balance inquiries and FAQs.
AI helps reduce call center workload.
Customers get answers 24/7 without waiting.
Complex cases still go to human agents.
"""
    },
    {
        "title": "AI in Education - Adaptive Learning",
        "text": """
Platforms adjust content based on student performance.
Struggling students get extra practice.
Advanced students move ahead quickly.
AI personalizes lessons for different learning styles.
"""
    },
    {
        "title": "AI in Education - Automated Grading",
        "text": """
AI grades multiple-choice and short answers instantly.
Teachers spend more time on creative tasks.
Feedback is faster, helping students learn better.
Essay grading AI is improving but still needs oversight.
"""
    },
    {
        "title": "AI in Retail - Personalization",
        "text": """
Recommendation engines suggest products to shoppers.
Personalized emails increase sales.
AI predicts what customers will need next.
Retailers tailor ads based on browsing history.
"""
    },
    {
        "title": "AI in Retail - Inventory Management",
        "text": """
AI forecasts demand for different products.
Stores reduce overstock and stockouts.
Supply chains become more efficient.
Real-time data keeps inventory accurate.
"""
    },
    {
        "title": "AI in Manufacturing - Predictive Maintenance",
        "text": """
Sensors detect early signs of equipment failure.
AI schedules repairs before breakdowns happen.
Unplanned downtime drops significantly.
Factories save money on maintenance.
"""
    },
    {
        "title": "AI in Manufacturing - Quality Control",
        "text": """
Computer vision inspects products on assembly lines.
AI spots defects humans might miss.
Quality improves and scrap rates fall.
Faster inspections increase production speed.
"""
    },
    {
        "title": "AI in Transportation - Self-Driving Cars",
        "text": """
AI combines sensor data to navigate roads.
Safety is still the biggest challenge.
Cars learn from millions of miles of driving data.
Regulations vary by country and state.
"""
    },
    {
        "title": "AI in Transportation - Traffic Optimization",
        "text": """
AI adjusts traffic lights to reduce congestion.
Sensors track vehicle flow in real time.
Reduced idling lowers emissions.
Cities become safer and more efficient.
"""
    },
    {
        "title": "AI in Customer Service - Chatbots",
        "text": """
Bots answer common customer questions instantly.
They handle thousands of chats at once.
Customers get faster service at lower cost.
Complex issues still need human help.
"""
    },
    {
        "title": "AI Ethics - Bias and Fairness",
        "text": """
AI can reflect biases in training data.
Teams must audit models for fairness.
Bias can harm certain groups unintentionally.
Transparency helps users trust AI decisions.
"""
    },
    {
        "title": "AI Ethics - Privacy",
        "text": """
AI systems collect large amounts of data.
Data must be protected from misuse.
Users should know how their data is used.
Privacy laws guide responsible AI design.
"""
    },
    {
        "title": "Challenges in AI Adoption",
        "text": """
Data quality often limits AI accuracy.
Employees may resist new technology.
High upfront costs deter smaller businesses.
Ethical concerns slow AI rollout in sensitive areas.
"""
    },
    {
        "title": "Benefits of AI",
        "text": """
AI automates repetitive tasks.
Faster decisions improve business efficiency.
Predictive models help reduce risk.
AI can create new products and services.
"""
    },
    {
        "title": "Best Practices for AI Projects",
        "text": """
Start with clear business goals.
Use high-quality and diverse data.
Monitor AI models for drift over time.
Involve domain experts to improve outcomes.
"""
    }
]


# 🔗 Step 3: Build the Neo4j knowledge graph
for doc in documents:
    doc_title = doc["title"]
    text = doc["text"]
    chunks = hybrid_chunk(text)
    embeddings = embedding_model.encode(chunks)

    # Create Document node
    graph.query(f"MERGE (d:Document {{title: '{doc_title}'}})")

    for idx, chunk in enumerate(chunks):
        node_id = f"{doc_title}_chunk_{idx}"
        embedding_str = ','.join(map(str, embeddings[idx]))
        safe_text = chunk.replace("'", "\\'")
        # Create Chunk node and connect
        graph.query(f"""
            MERGE (c:Chunk {{id: '{node_id}'}})
            SET c.text = '{safe_text}', c.embedding = '[{embedding_str}]',
                c.query_count = coalesce(c.query_count, 0),
                c.last_queried = coalesce(c.last_queried, 0)
            WITH c
            MATCH (d:Document {{title: '{doc_title}'}})
            MERGE (d)-[:HAS_CHUNK]->(c)
        """)

print("✅ All documents and chunks stored in Neo4j!")

# -------------------------------------------------------
# ✅ Hybrid ranking code

def fetch_chunks_with_metadata(graph):
    result = graph.query("""
        MATCH (c:Chunk)
        RETURN c.id AS id, c.text AS text, c.embedding AS embedding,
               coalesce(c.query_count, 0) AS query_count,
               coalesce(c.last_queried, 0) AS last_queried
    """)
    
    chunk_ids, chunk_texts, embeddings, query_counts, last_queried = [], [], [], [], []

    for row in result:
        chunk_ids.append(row["id"])
        chunk_texts.append(row["text"])
        emb = np.fromstring(row["embedding"].strip('[]'), sep=',')
        embeddings.append(emb)
        query_counts.append(int(row["query_count"]))
        last_queried.append(int(row["last_queried"]))

    return chunk_ids, chunk_texts, np.vstack(embeddings), np.array(query_counts), np.array(last_queried)

def compute_similarity(query_embedding, chunk_embeddings):
    return cosine_similarity([query_embedding], chunk_embeddings)[0]

def compute_normalized_scores(query_counts, last_queried, now=None, decay_factor=1e-8):
    if now is None:
        now = int(time.time() * 1000)
    # Popularity normalization
    popularity = query_counts / query_counts.max() if query_counts.max() > 0 else np.zeros_like(query_counts, dtype=float)
    # Recency decay
    age = now - last_queried
    recency = np.exp(-decay_factor * age)
    return popularity, recency

def combine_scores(similarity, popularity, recency, alpha=0.6, beta=0.3, gamma=0.1):
    return alpha * similarity + beta * popularity + gamma * recency

def update_query_metadata(graph, chunk_ids):
    for chunk_id in chunk_ids:
        graph.query(f"""
            MATCH (c:Chunk {{id: '{chunk_id}'}})
            SET c.query_count = coalesce(c.query_count, 0) + 1,
                c.last_queried = timestamp()
        """)

def hybrid_rank_chunks(graph, query_text, top_k=5, alpha=0.6, beta=0.3, gamma=0.1):
    query_embedding = embedding_model.encode([query_text])[0]
    chunk_ids, chunk_texts, chunk_embeddings, query_counts, last_queried = fetch_chunks_with_metadata(graph)
    similarity = compute_similarity(query_embedding, chunk_embeddings)
    popularity, recency = compute_normalized_scores(query_counts, last_queried)
    final_scores = combine_scores(similarity, popularity, recency, alpha, beta, gamma)
    
    ranked = sorted(zip(chunk_ids, chunk_texts, final_scores), key=lambda x: x[2], reverse=True)
    top_chunk_ids = [chunk_id for chunk_id, _, _ in ranked[:top_k]]
    update_query_metadata(graph, top_chunk_ids)
    return ranked[:top_k]

# -------------------------------------------------------
# ✅ Example: run a query

query = "How does AI help reduce costs?"
top_chunks = hybrid_rank_chunks(graph, query, top_k=5)

print("\n📊 Top ranked chunks:")
for rank, (chunk_id, text, score) in enumerate(top_chunks, 1):
    print(f"Rank {rank}: Chunk ID={chunk_id}, Score={score:.4f}")
    print(f"Text: {text}\n")
