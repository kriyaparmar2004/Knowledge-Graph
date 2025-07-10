# 🚀 AI Knowledge Graph with Hybrid Ranking

This project builds a **knowledge graph** in Neo4j from a set of AI-related documents, then lets you **rank and search** the graph by combining:
- Semantic similarity (using embeddings)
- Popularity (how often chunks were queried)
- Recency (how recently chunks were queried)

🔎 Ideal for building **smart semantic search systems** that adapt to user behavior.

---

## 📂 **Project structure**

- `hybrid_kg.py` — main Python script (builds the graph + runs hybrid ranking)
- `README.md` — project guide

---

## ⚙️ **Setup instructions**

### ✅ 1. Clone the repo (or copy files)

```bash
git clone https://github.com/Rezinix-AI/Enterprise-Rag-standalone.git
cd Enterprise-Rag-standalone
git checkout knowledgeGraphWithRanking


✅ 2. Install dependencies
Create a virtual environment (recommended):

bash
Copy
Edit
python -m venv venv
source venv/bin/activate    # On Linux/macOS
venv\Scripts\activate       # On Windows
Install required packages:

bash
Copy
Edit
pip install langchain sentence-transformers scikit-learn neo4j numpy
✅ 3. Set up Neo4j
Download and install Neo4j Community Edition:
👉 https://neo4j.com/download/

Start Neo4j Browser (default at http://localhost:7474)

Ensure Neo4j is running on neo4j://127.0.0.1:7687

Use user: neo4j and set password to:

text
Copy
Edit
Your password when you set up te instance
(Or change the password in hybrid_kg.py if you prefer.)

▶️ How to run
After setup, simply run:

bash
Copy
Edit
python hybrid_kg.py
What it does:

Creates Document and Chunk nodes in Neo4j (from 20 AI sample documents)

Embeds text chunks using all-MiniLM-L6-v2 model

Lets you fire a sample query like:

“How does AI help reduce costs?”

Ranks results by a hybrid score:

Semantic similarity

Popularity (query_count)

Recency (last_queried with decay)

Updates popularity counters and timestamps automatically

🧠 How to visualize the knowledge graph
In Neo4j Browser, run:

cypher
Copy
Edit
MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
RETURN d, c
To see all nodes and relationships:

cypher
Copy
Edit
MATCH (n)-[r]->(m)
RETURN n, r, m
✏️ Customize
Add your own documents in documents = [...] list

Change chunk size / overlap in:

python
Copy
Edit
RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
Tune ranking weights:

python
Copy
Edit
alpha=0.6  # similarity
beta=0.3   # popularity
gamma=0.1  # recency
