import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import time
import uuid
from collections import defaultdict

@dataclass
class GraphRAGResult:
    """Structure for GraphRAG results"""
    answer: str
    sources: List[str]
    confidence_score: float
    personalization_factors: Dict
    retrieved_chunks: List[Dict]

class PersonalizedGraphRAG:
    """
    Integrated Personalized GraphRAG System that combines:
    1. Knowledge Graph (Neo4j) for structured relationships
    2. Vector embeddings for semantic similarity
    3. Personalization engine for user-specific results
    4. LLM for answer generation
    """
    
    def __init__(self, neo4j_url: str, neo4j_username: str, neo4j_password: str, 
                 openai_api_key: str = None, embedding_model_name: str = 'all-MiniLM-L6-v2'):
        
        # Initialize core components
        self.graph = Neo4jGraph(
            url=neo4j_url,
            username=neo4j_username,
            password=neo4j_password
        )
        
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Set up LLM
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        else:
            # Fallback to HuggingFace or other LLM
            from langchain_huggingface import HuggingFacePipeline
            self.llm = HuggingFacePipeline.from_model_id(
                model_id="microsoft/DialoGPT-medium",
                task="text-generation"
            )
        
        # Initialize personalization
        self.user_profiles = {}
        self.domain_keywords = {
            'healthcare': ['medical', 'doctor', 'patient', 'diagnosis', 'treatment', 'hospital', 'disease', 'health'],
            'finance': ['bank', 'trading', 'fraud', 'credit', 'investment', 'financial', 'money', 'market'],
            'education': ['student', 'teacher', 'learning', 'classroom', 'tutor', 'grade', 'academic', 'school'],
            'retail': ['customer', 'shopping', 'product', 'inventory', 'sales', 'store', 'purchase', 'recommendation'],
            'manufacturing': ['factory', 'production', 'assembly', 'maintenance', 'quality', 'equipment', 'machinery'],
            'transportation': ['vehicle', 'traffic', 'driving', 'road', 'navigation', 'safety', 'cars'],
            'technology': ['AI', 'machine learning', 'algorithm', 'data', 'model', 'system', 'automation']
        }
        
        # Setup schema
        self._setup_schema()
        
        # RAG prompts
        self.setup_prompts()
    
    def _setup_schema(self):
        """Create comprehensive Neo4j schema"""
        schema_queries = [
            # User and interaction constraints
            """CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE""",
            """CREATE CONSTRAINT interaction_id IF NOT EXISTS FOR (i:Interaction) REQUIRE i.id IS UNIQUE""",
            """CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE""",
            """CREATE CONSTRAINT document_title IF NOT EXISTS FOR (d:Document) REQUIRE d.title IS UNIQUE""",
            
            # Indexes for performance
            """CREATE INDEX user_role IF NOT EXISTS FOR (u:User) ON (u.role)""",
            """CREATE INDEX chunk_embedding IF NOT EXISTS FOR (c:Chunk) ON (c.embedding)""",
            """CREATE INDEX interaction_timestamp IF NOT EXISTS FOR (i:Interaction) ON (i.timestamp)""",
            
            # Entity and relationship indexes
            """CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)""",
            """CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)"""
        ]
        
        for query in schema_queries:
            try:
                self.graph.query(query)
            except Exception as e:
                print(f"Schema setup note: {e}")
    
    def setup_prompts(self):
        """Setup RAG prompts for different scenarios"""
        
        # Base GraphRAG prompt
        self.graphrag_template = """
You are an expert assistant that provides accurate, personalized answers based on retrieved knowledge.

Context from Knowledge Graph:
{graph_context}

Retrieved Text Chunks:
{vector_context}

User Profile:
- Role: {user_role}
- Department: {user_department}
- Experience Level: {experience_level}

Personalization Insights:
{personalization_insights}

Question: {question}

Instructions:
1. Provide a comprehensive answer that combines information from both the graph context and text chunks
2. Tailor your response to the user's role and expertise level
3. Use specific examples relevant to their domain
4. If information is incomplete, clearly state limitations
5. Cite which sources informed your answer

Answer:"""

        # Confidence assessment prompt
        self.confidence_template = """
Based on the following factors, rate the confidence of this answer on a scale of 0.0 to 1.0:

Retrieved Information Quality: {retrieval_quality}
Source Relevance: {source_relevance}  
Personalization Match: {personalization_match}
Question Complexity: {question_complexity}

Answer: {answer}

Provide only a confidence score between 0.0 and 1.0:"""

        self.graphrag_prompt = PromptTemplate.from_template(self.graphrag_template)
        self.confidence_prompt = PromptTemplate.from_template(self.confidence_template)
    
    def create_user_profile(self, user_id: str, role: str, department: str, 
                          experience_level: str = "intermediate", preferences: Dict = None) -> Dict:
        """Create comprehensive user profile"""
        preferences = preferences or {}
        
        # Store in Neo4j
        user_query = f"""
            MERGE (u:User {{id: '{user_id}'}})
            SET u.role = '{role}',
                u.department = '{department}',
                u.experience_level = '{experience_level}',
                u.preferences = '{json.dumps(preferences)}',
                u.created_at = timestamp(),
                u.updated_at = timestamp()
            RETURN u
        """
        
        self.graph.query(user_query)
        
        # Store in memory for fast access
        self.user_profiles[user_id] = {
            'role': role,
            'department': department,
            'experience_level': experience_level,
            'preferences': preferences,
            'domain_affinities': defaultdict(float),
            'interaction_history': [],
            'similar_users': []
        }
        
        return self.user_profiles[user_id]
    
    def enhanced_graph_retrieval(self, query: str, user_id: str, top_k: int = 5) -> Tuple[List[Dict], str]:
        """
        Enhanced graph retrieval that combines:
        1. Semantic search on chunks
        2. Graph traversal for related entities
        3. User personalization
        """
        
        # Step 1: Get user profile
        user_profile = self.user_profiles.get(user_id, {})
        user_role = user_profile.get('role', 'general')
        user_dept = user_profile.get('department', 'general')
        
        # Step 2: Semantic vector search with personalization
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Get chunks with metadata
        chunk_query = """
            MATCH (c:Chunk)
            OPTIONAL MATCH (c)<-[:HAS_CHUNK]-(d:Document)
            RETURN c.id AS chunk_id, c.text AS text, c.embedding AS embedding,
                   coalesce(c.query_count, 0) AS query_count,
                   coalesce(c.last_queried, 0) AS last_queried,
                   d.title AS document_title
        """
        
        chunks_data = self.graph.query(chunk_query)
        
        # Compute personalized scores
        ranked_chunks = self._compute_personalized_ranking(
            query_embedding, chunks_data, user_id, top_k
        )
        
        # Step 3: Graph traversal for entity relationships
        # Extract key entities from top chunks
        top_chunk_ids = [chunk['chunk_id'] for chunk in ranked_chunks[:3]]
        
        graph_context = self._get_graph_context(top_chunk_ids, query)
        
        return ranked_chunks, graph_context
    
    def _compute_personalized_ranking(self, query_embedding: np.ndarray, 
                                    chunks_data: List[Dict], user_id: str, top_k: int) -> List[Dict]:
        """Compute personalized ranking combining multiple signals"""
        
        scored_chunks = []
        current_time = int(time.time() * 1000)
        
        for chunk in chunks_data:
            # Parse embedding
            embedding_str = chunk['embedding'].strip('[]')
            chunk_embedding = np.fromstring(embedding_str, sep=',')
            
            # Semantic similarity
            similarity = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
            
            # Popularity score
            query_count = chunk['query_count']
            popularity = min(query_count / 100.0, 1.0)  # Normalize
            
            # Recency score
            last_queried = chunk['last_queried']
            if last_queried > 0:
                age_hours = (current_time - last_queried) / (1000 * 3600)
                recency = np.exp(-age_hours / 168)  # Decay over week
            else:
                recency = 0.1
            
            # Personalization scores
            domain_pref = self._compute_domain_preference(user_id, chunk['text'])
            role_boost = self._compute_role_boost(user_id, chunk['text'])
            
            # Combined score
            final_score = (0.4 * similarity + 
                          0.2 * popularity + 
                          0.1 * recency + 
                          0.2 * domain_pref + 
                          0.1 * role_boost)
            
            scored_chunks.append({
                'chunk_id': chunk['chunk_id'],
                'text': chunk['text'],
                'document_title': chunk['document_title'],
                'final_score': final_score,
                'similarity': similarity,
                'domain_preference': domain_pref,
                'role_boost': role_boost,
                'popularity': popularity,
                'recency': recency
            })
        
        # Sort by final score
        scored_chunks.sort(key=lambda x: x['final_score'], reverse=True)
        return scored_chunks[:top_k]
    
    def _compute_domain_preference(self, user_id: str, text: str) -> float:
        """Compute domain preference score"""
        if user_id not in self.user_profiles:
            return 0.0
        
        profile = self.user_profiles[user_id]
        domain_affinities = profile['domain_affinities']
        
        text_lower = text.lower()
        preference_score = 0.0
        
        for domain, keywords in self.domain_keywords.items():
            domain_relevance = sum(1 for keyword in keywords if keyword in text_lower)
            if domain_relevance > 0:
                affinity = domain_affinities.get(domain, 0.1)
                preference_score += affinity * domain_relevance * 0.1
        
        return min(preference_score, 1.0)
    
    def _compute_role_boost(self, user_id: str, text: str) -> float:
        """Compute role-based content boost"""
        if user_id not in self.user_profiles:
            return 0.0
        
        role = self.user_profiles[user_id]['role']
        text_lower = text.lower()
        
        role_keywords = {
            'doctor': ['diagnosis', 'patient', 'medical', 'treatment', 'clinical'],
            'financial_analyst': ['trading', 'market', 'investment', 'risk', 'analysis'],
            'teacher': ['student', 'learning', 'education', 'curriculum', 'teaching'],
            'retail_manager': ['customer', 'sales', 'inventory', 'revenue', 'store'],
            'engineer': ['technical', 'system', 'optimization', 'performance', 'design'],
            'data_scientist': ['model', 'prediction', 'analysis', 'statistics', 'algorithm']
        }
        
        if role in role_keywords:
            keywords = role_keywords[role]
            relevance = sum(1 for keyword in keywords if keyword in text_lower)
            return min(relevance * 0.15, 0.5)
        
        return 0.0
    
    def _get_graph_context(self, chunk_ids: List[str], query: str) -> str:
        """Get graph context through entity relationships"""
        
        # Extract entities mentioned in top chunks
        entity_query = f"""
            MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
            WHERE c.id IN {chunk_ids}
            RETURN e.name AS entity_name, e.type AS entity_type, 
                   collect(c.id) AS chunk_ids
            LIMIT 10
        """
        
        entities = self.graph.query(entity_query)
        
        if not entities:
            return "No additional graph context available."
        
        # Get relationships between entities
        entity_names = [e['entity_name'] for e in entities]
        
        relationships_query = f"""
            MATCH (e1:Entity)-[r]->(e2:Entity)
            WHERE e1.name IN {entity_names} AND e2.name IN {entity_names}
            RETURN e1.name AS source, type(r) AS relationship, e2.name AS target
            LIMIT 20
        """
        
        relationships = self.graph.query(relationships_query)
        
        # Format graph context
        context_parts = []
        
        if entities:
            context_parts.append("Key Entities:")
            for entity in entities:
                context_parts.append(f"- {entity['entity_name']} ({entity['entity_type']})")
        
        if relationships:
            context_parts.append("\nEntity Relationships:")
            for rel in relationships:
                context_parts.append(f"- {rel['source']} {rel['relationship']} {rel['target']}")
        
        return "\n".join(context_parts) if context_parts else "Limited graph context available."
    
    def track_interaction(self, user_id: str, query: str, clicked_chunks: List[str], 
                         feedback_score: float = 0.0):
        """Track user interaction for learning"""
        
        interaction_id = str(uuid.uuid4())
        timestamp = int(time.time() * 1000)
        
        # Store in Neo4j
        interaction_query = f"""
            MATCH (u:User {{id: '{user_id}'}})
            CREATE (i:Interaction {{
                id: '{interaction_id}',
                query: '{query.replace("'", "\\'")}',
                clicked_chunks: '{json.dumps(clicked_chunks)}',
                feedback_score: {feedback_score},
                timestamp: {timestamp}
            }})
            CREATE (u)-[:HAS_INTERACTION]->(i)
        """
        
        self.graph.query(interaction_query)
        
        # Update personalization
        self._update_user_preferences(user_id, query, clicked_chunks, feedback_score)
    
    def _update_user_preferences(self, user_id: str, query: str, 
                               clicked_chunks: List[str], feedback_score: float):
        """Update user preferences based on interaction"""
        
        if user_id not in self.user_profiles:
            return
        
        profile = self.user_profiles[user_id]
        
        # Update domain affinities
        for domain, keywords in self.domain_keywords.items():
            query_lower = query.lower()
            domain_relevance = sum(1 for keyword in keywords if keyword in query_lower)
            
            if domain_relevance > 0:
                # Positive feedback increases affinity
                boost = domain_relevance * 0.1 * max(feedback_score, 0.1)
                profile['domain_affinities'][domain] += boost
        
        # Add to interaction history
        profile['interaction_history'].append({
            'query': query,
            'clicked_chunks': clicked_chunks,
            'feedback_score': feedback_score,
            'timestamp': int(time.time() * 1000)
        })
        
        # Keep recent history
        if len(profile['interaction_history']) > 50:
            profile['interaction_history'] = profile['interaction_history'][-50:]
    
    def generate_answer(self, query: str, user_id: str, 
                       include_sources: bool = True) -> GraphRAGResult:
        """
        Generate personalized answer using GraphRAG
        """
        
        # Step 1: Retrieve relevant content
        retrieved_chunks, graph_context = self.enhanced_graph_retrieval(query, user_id, top_k=5)
        
        # Step 2: Get user profile
        user_profile = self.user_profiles.get(user_id, {})
        
        # Step 3: Format context
        vector_context = "\n\n".join([
            f"Source: {chunk['document_title']}\nContent: {chunk['text']}"
            for chunk in retrieved_chunks
        ])
        
        # Step 4: Create personalization insights
        top_domains = dict(sorted(user_profile.get('domain_affinities', {}).items(),
                                key=lambda x: x[1], reverse=True)[:3])
        
        personalization_insights = f"""
- Primary interests: {list(top_domains.keys())}
- Recent interaction patterns: {len(user_profile.get('interaction_history', []))} interactions
- Personalization factors applied: domain preferences, role-based content boosting
        """.strip()
        
        # Step 5: Generate answer
        prompt_input = {
            'graph_context': graph_context,
            'vector_context': vector_context,
            'user_role': user_profile.get('role', 'general'),
            'user_department': user_profile.get('department', 'general'),
            'experience_level': user_profile.get('experience_level', 'intermediate'),
            'personalization_insights': personalization_insights,
            'question': query
        }
        
        response = self.llm.invoke(self.graphrag_prompt.format(**prompt_input))
        answer = response.content if hasattr(response, 'content') else str(response)
        
        # Step 6: Compute confidence
        confidence = self._compute_confidence(retrieved_chunks, answer)
        
        # Step 7: Prepare sources
        sources = [f"{chunk['document_title']}: {chunk['text'][:100]}..." 
                  for chunk in retrieved_chunks] if include_sources else []
        
        # Step 8: Track this query
        chunk_ids = [chunk['chunk_id'] for chunk in retrieved_chunks]
        self._update_chunk_metadata(chunk_ids)
        
        return GraphRAGResult(
            answer=answer,
            sources=sources,
            confidence_score=confidence,
            personalization_factors={
                'domain_preferences': dict(top_domains),
                'role_boost_applied': user_profile.get('role', 'none'),
                'retrieval_personalization': True
            },
            retrieved_chunks=retrieved_chunks
        )
    
    def _compute_confidence(self, retrieved_chunks: List[Dict], answer: str) -> float:
        """Compute confidence score for the answer"""
        
        if not retrieved_chunks:
            return 0.1
        
        # Average similarity score
        avg_similarity = np.mean([chunk['similarity'] for chunk in retrieved_chunks])
        
        # Source diversity
        unique_docs = len(set(chunk['document_title'] for chunk in retrieved_chunks))
        diversity_score = min(unique_docs / 3.0, 1.0)
        
        # Answer length (reasonable answers should have some substance)
        length_score = min(len(answer.split()) / 100.0, 1.0)
        
        # Combined confidence
        confidence = 0.5 * avg_similarity + 0.3 * diversity_score + 0.2 * length_score
        
        return min(confidence, 0.95)  # Cap at 95%
    
    def _update_chunk_metadata(self, chunk_ids: List[str]):
        """Update chunk query metadata"""
        for chunk_id in chunk_ids:
            self.graph.query(f"""
                MATCH (c:Chunk {{id: '{chunk_id}'}})
                SET c.query_count = coalesce(c.query_count, 0) + 1,
                    c.last_queried = timestamp()
            """)

# Usage Example
def demo_personalized_graphrag():
    """Demo the integrated system"""
    
    # Initialize system
    graphrag = PersonalizedGraphRAG(
        neo4j_url="neo4j://localhost:7687",
        neo4j_username="neo4j", 
        neo4j_password="RezinixKnowledgeGraph",
        openai_api_key=""  # Set your actual key
    )
    
    # Create users
    users = [
        {'id': 'doctor_001', 'role': 'doctor', 'department': 'healthcare', 'level': 'expert'},
        {'id': 'analyst_001', 'role': 'financial_analyst', 'department': 'finance', 'level': 'intermediate'},
    ]
    
    for user in users:
        graphrag.create_user_profile(user['id'], user['role'], user['department'], user['level'])
    
    # Simulate interactions to build preferences
    graphrag.track_interaction('doctor_001', 'AI medical diagnosis', 
                             ['AI in Healthcare_chunk_0'], feedback_score=0.9)
    
    # Test personalized retrieval
    query = "How does AI improve efficiency and reduce costs?"
    
    for user in users:
        print(f"\n{'='*60}")
        print(f"Results for {user['role']} ({user['id']})")
        print('='*60)
        
        result = graphrag.generate_answer(query, user['id'])
        
        print(f"🤖 Answer (Confidence: {result.confidence_score:.2f}):")
        print(result.answer)
        print(f"\n📊 Personalization Factors:")
        for key, value in result.personalization_factors.items():
            print(f"- {key}: {value}")
        
        if result.sources:
            print(f"\n📚 Sources:")
            for i, source in enumerate(result.sources[:3], 1):
                print(f"{i}. {source}")

if __name__ == "__main__":
    demo_personalized_graphrag()