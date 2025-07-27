# 🚀 AI Knowledge Graph with Hybrid Ranking

This project constructs a **knowledge graph** in Neo4j from a collection of AI-related documents. It enables **intelligent retrieval and ranking** by integrating:

- ✅ **Semantic similarity** (via sentence embeddings)
- 🔁 **Popularity** (query frequency of chunks)
- 🕒 **Recency** (timestamp decay based on latest queries)

Ideal for developing **adaptive semantic search systems** that evolve based on user interaction.

---

## 🧠 Features

- Builds a knowledge graph with `Document` and `Chunk` nodes.
- Embeds text chunks using the `all-MiniLM-L6-v2` model.
- Implements hybrid ranking for search queries using:
  - Cosine similarity
  - Query count
  - Last queried timestamp (decay factor)
- Automatically updates metadata based on search activity.

---

## 📁 Project Structure

```bash
├── hybrid_kg.py         # Main script: graph construction + hybrid query ranking
├── README.md            # Project documentation
├── requirements.txt     # List of Python dependencies
⚙️ Setup Instructions
1️⃣ Clone the Repository
2️⃣ Create Virtual Environment (Recommended)
bash
Copy
Edit
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
3️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
4️⃣ Set up Neo4j
Download Neo4j Community Edition: 👉 Neo4j Download Page

Start Neo4j Browser at: http://localhost:7474

Default Neo4j connection URL: neo4j://127.0.0.1:7687

Use default user: neo4j

Set or update the password in hybrid_kg.py:

python
Copy
Edit
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "your_password_here"
▶️ Running the Project
After setup, run:

bash
Copy
Edit
python hybrid_kg.py
This will:

Load AI sample documents

Split into chunks

Embed each chunk

Build graph in Neo4j

Accept a sample query like:

"How does AI reduce cost in businesses?"

And return top-ranked answers based on hybrid score:

hybrid_score = α * similarity + β * popularity + γ * recency

🧾 Visualize the Knowledge Graph (Neo4j)
To see all documents and chunks:

cypher
Copy
Edit
MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
RETURN d, c
To explore all graph relationships:

cypher
Copy
Edit
MATCH (n)-[r]->(m)
RETURN n, r, m
🛠️ Customization
➕ Add Your Own Documents
Replace or extend the list:

python
Copy
Edit
documents = [
    "AI is transforming healthcare...",
    ...
]
🧩 Adjust Chunking Parameters
python
Copy
Edit
RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
🔧 Tune Ranking Weights
python
Copy
Edit
alpha = 0.6   # Semantic similarity
beta = 0.3    # Popularity (query_count)
gamma = 0.1   # Recency (last_queried)