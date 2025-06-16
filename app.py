import streamlit as st
from snowflake.snowpark.context import get_active_session
import time
import json
import uuid
from datetime import datetime
import logging
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from typing import List, Dict, Tuple, Optional, Any

# ——— App config ———
st.set_page_config(
    page_title="Multi‐Property RAG Chat", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ——— Initialize Snowpark session ———
@st.cache_resource
def get_session():
    """Create and cache Snowpark session."""
    from snowflake.snowpark import Session
    
    connection_parameters = {
        "account": st.secrets["snowflake"]["account"],
        "user": st.secrets["snowflake"]["user"],
        "password": st.secrets["snowflake"]["password"],
        "role": st.secrets["snowflake"]["role"],
        "warehouse": st.secrets["snowflake"]["warehouse"],
        "database": st.secrets["snowflake"]["database"],
        "schema": st.secrets["snowflake"]["schema"]
    }
    
    session = Session.builder.configs(connection_parameters).create()
    return session

session = get_session()


SHOW_DEBUG_SIDEBAR = st.secrets.get("debug", {}).get("show_sidebar", False)


# ——— Constants ———
MODEL_NAME = 'SNOWFLAKE.MODELS."MIXTRAL-8X7B"'  # Primary model for content generation
REFINE_MODEL = 'SNOWFLAKE.MODELS."MIXTRAL-8X7B"'    # Cheaper model for refinement
EMBED_MODEL = 'snowflake-arctic-embed-l-v2.0'
EMBED_FN = 'SNOWFLAKE.CORTEX.EMBED_TEXT_1024'
WORD_THRESHOLD = 50  # Responses longer than this get refined

# ——— Configuration ———
if 'config' not in st.session_state:
    st.session_state.config = {
        'top_k': 5,
        'similarity_threshold': 0.2,
        'max_response_words': 100,
        'context_window': 4,
        'enable_logging': True,
        'enable_refinement': True,
        'enable_caching': True,
        'cache_ttl': 3600,
        'debug_mode': False
    }

# ——— Performance Monitor ———
class PerformanceMonitor:
    def __init__(self):
        if 'performance_metrics' not in st.session_state:
            st.session_state.performance_metrics = {
                'response_times': [],
                'token_usage': [],
                'cache_hits': 0,
                'cache_misses': 0,
                'refinement_count': 0,
                'total_requests': 0,
                'errors': []
            }
        self.metrics = st.session_state.performance_metrics
    
    def log_request(self, metrics: Dict[str, Any]):
        self.metrics['response_times'].append(metrics.get('latency', 0))
        self.metrics['token_usage'].append(metrics.get('tokens', 0))
        self.metrics['total_requests'] += 1
        if metrics.get('used_refinement'):
            self.metrics['refinement_count'] += 1
        
        # Keep only last 100 entries to prevent memory issues
        if len(self.metrics['response_times']) > 100:
            self.metrics['response_times'] = self.metrics['response_times'][-100:]
            self.metrics['token_usage'] = self.metrics['token_usage'][-100:]
    
    def log_cache_hit(self):
        self.metrics['cache_hits'] += 1
    
    def log_cache_miss(self):
        self.metrics['cache_misses'] += 1
    
    def log_error(self, error_type: str, details: str):
        self.metrics['errors'].append({
            'timestamp': datetime.now().isoformat(),
            'type': error_type,
            'details': details
        })
        # Keep only last 50 errors
        if len(self.metrics['errors']) > 50:
            self.metrics['errors'] = self.metrics['errors'][-50:]
    
    def get_dashboard_metrics(self) -> Dict[str, Any]:
        if not self.metrics['response_times']:
            return {'status': 'No data yet'}
        
        total_cache_ops = self.metrics['cache_hits'] + self.metrics['cache_misses']
        return {
            'avg_response_time': np.mean(self.metrics['response_times']),
            'p95_response_time': np.percentile(self.metrics['response_times'], 95) if len(self.metrics['response_times']) > 10 else 0,
            'cache_hit_rate': self.metrics['cache_hits'] / total_cache_ops if total_cache_ops > 0 else 0,
            'avg_tokens_per_request': np.mean(self.metrics['token_usage']) if self.metrics['token_usage'] else 0,
            'refinement_rate': self.metrics['refinement_count'] / self.metrics['total_requests'] if self.metrics['total_requests'] > 0 else 0,
            'total_requests': self.metrics['total_requests'],
            'recent_errors': len(self.metrics['errors'])
        }

monitor = PerformanceMonitor()

# ——— Error Handling ———
class ChatError:
    def __init__(self, error_type: str, user_message: str, technical_details: str = None):
        self.error_type = error_type
        self.user_message = user_message
        self.technical_details = technical_details
        monitor.log_error(error_type, technical_details or user_message)
    
    def display(self):
        st.error(f"😔 {self.user_message}")
        if st.session_state.config.get('debug_mode') and self.technical_details:
            with st.expander("Technical details"):
                st.code(self.technical_details)

# ——— Execution Logging for Debugging ———
if 'execution_log' not in st.session_state:
    st.session_state.execution_log = []

def log_execution(step: str, details: str = "", timing: float = None):
    """Log execution steps for debugging visibility."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    log_entry = {
        "timestamp": timestamp,
        "step": step,
        "details": details,
        "timing": f"{timing:.3f}s" if timing else ""
    }
    st.session_state.execution_log.append(log_entry)
    
    # Keep only last 50 entries to prevent memory issues
    if len(st.session_state.execution_log) > 50:
        st.session_state.execution_log = st.session_state.execution_log[-50:]

# ——— Enhanced System Prompt ———
def get_system_prompt(property_id: int, config: dict) -> str:
    """Generate dynamic system prompt based on property and config."""
    return json.dumps({
        "role": "system",
        "content": {
            "persona": "helpful, warm property expert",
            "tone": "short, friendly sentences",
            "focus_rule": "Identify and answer only the most recent guest request; discard any other queries.",
            "hallucination_protection": {
                "explicit_denial": "If the provided context does not directly answer, use the fallback.",
                "fallback_response": "I'm sorry, I don't think I can help with that one. Could you please rephrase, or contact the host for more help?"
            },
            "response_constraints": {
                "format": "plain text only, no speaker labels, no markdown or lists",
                "length_limit": f"max {config['max_response_words']} words",
                "no_extra_detail": "Do not provide any information beyond what the context supports."
            },
            "note": "Only the top context chunks are sent to the model; chunk selection is handled upstream in SQL."
        }
    })

# ——— Editor prompt for Stage 2 ———
EDITOR_PROMPT = json.dumps({
    "role": "editor",
    "task": "Rewrite the response to be concise and directly answer only the original question",
    "critical_rule": "NEVER change the factual meaning or create contradictions. If the original says 'Yes' keep it 'Yes'. If it says 'No' keep it 'No'.",
    "rules": [
        "Preserve all factual information and the core answer (Yes/No/factual details)",
        "Keep the warm, helpful, friendly tone - don't make it cold or robotic", 
        "Remove only extra context, rules, or tangential information that doesn't answer the question", 
        "Maintain conversational style - keep phrases like 'Enjoy your stay!' if they fit",
        "Maximum 30 words unless the question requires more detail"
    ],
    "tone": "Warm, helpful, conversational - like a friendly concierge",
    "focus": "Brevity through removing extras, NOT by changing facts or tone"
})

# ——— Enhanced Conversation Logging ———
conversation_log = []

def log_turn(role: str, message: str, metadata: dict = None):
    """Enhanced logging with metadata."""
    if not st.session_state.config['enable_logging']:
        return
        
    turn = {
        "role": role,
        "message": message,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "property_id": st.session_state.get('property_id'),
        "session_id": st.session_state.get('session_id', str(uuid.uuid4())),
        "metadata": metadata or {}
    }
    conversation_log.append(turn)
    
    try:
        write_turn_to_table(turn)
    except Exception as e:
        logging.error(f"Failed to log conversation turn: {e}")

def write_turn_to_table(turn: dict):
    """Write conversation turn to Snowflake with better error handling."""
    try:
        conversation_json = json.dumps([turn])
        escaped_json = conversation_json.replace("'", "''").replace("\\", "\\\\")
        
        insert_sql = f"""
          INSERT INTO TEST_DB.CORTEX.conversations(id, created_at, conversation, property_id, session_id)
          SELECT 
            ?,
            TO_TIMESTAMP_LTZ(?),
            PARSE_JSON('{escaped_json}'),
            ?,
            ?
        """
        
        session.sql(insert_sql, params=[
            str(uuid.uuid4()),
            turn['timestamp'],
            turn.get('property_id'),
            turn.get('session_id')
        ]).collect()
        
    except Exception as e:
        logging.error(f"Database logging error: {e}")

# ——— Conversation Summarization ———
def summarize_conversation(chat_history: List[Dict[str, str]]) -> Optional[str]:
    """Periodically summarize long conversations."""
    if len(chat_history) < 20:
        return None
    
    try:
        # Get last 20 messages
        recent_history = chat_history[-20:]
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
        
        summary_prompt = f"""
        Summarize the key topics and questions discussed in this conversation.
        Keep it brief (2-3 sentences) and focus on main themes.
        
        Conversation:
        {history_text}
        
        Summary:
        """
        
        df = session.sql(
            "SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) AS summary",
            params=[REFINE_MODEL, summary_prompt]
        ).collect()
        
        return df[0].SUMMARY.strip() if df else None
    except Exception as e:
        logging.error(f"Summarization error: {e}")
        return None

# ——— Enhanced Pre-warming ———
@st.cache_data(ttl=3600)
def initialize_system():
    """Pre-warm system components with caching."""
    try:
        wh_info = session.sql("SHOW WAREHOUSES LIKE 'CORTEX_WH'").collect()
        if wh_info and wh_info[0]['state'] == 'SUSPENDED':
            session.sql("ALTER WAREHOUSE CORTEX_WH RESUME").collect()
        
        session.sql(
            "SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_1024(?, 'warmup')",
            params=[EMBED_MODEL]
        ).collect()
        
        return True
    except Exception as e:
        st.error(f"System initialization failed: {e}")
        return False

# ——— Caching utilities ———
def get_query_hash(query: str, property_id: int) -> str:
    """Generate hash for query caching."""
    content = f"{query}_{property_id}_{st.session_state.config['top_k']}"
    return hashlib.md5(content.encode()).hexdigest()

@st.cache_data(ttl=3600, max_entries=1000)
def get_cached_embedding(text: str, model: str):
    """Cache embeddings to reduce API calls."""
    monitor.log_cache_miss()
    return session.sql(f"SELECT {EMBED_FN}(?, ?) AS embedding", params=[model, text]).collect()

# ——— Dynamic context window ———
def get_dynamic_context_window(chat_history: List[Dict], question: str) -> int:
    """Dynamically adjust context window based on question type."""
    if detect_appliance_query(question):
        return 2  # Less context needed for specific queries
    elif any(word in question.lower() for word in ['recommend', 'suggest', 'what should']):
        return 6  # More context for recommendations
    else:
        return st.session_state.config['context_window']  # Default

# ——— Enhanced Question Processing ———
def process_question(raw_q: str, property_id: int, chat_history: list) -> str:
    """Enhanced question processing with context awareness."""
    # Dynamic context window
    window_size = get_dynamic_context_window(chat_history, raw_q)
    
    recent_context = ""
    if len(chat_history) > 1:
        recent_msgs = chat_history[-window_size:]
        recent_context = " Previous context: " + " ".join([
            f"{msg['role']}: {msg['content'][:50]}..." 
            for msg in recent_msgs if msg['role'] == 'user'
        ])
    
    enriched = f"""
    Guest inquiry for Property #{property_id}: {raw_q.strip()}
    {recent_context}
    
    Please provide information specific to this property's amenities, policies, and local area.
    """.strip()
    
    return enriched

# ——— Multi-stage retrieval ———
def find_exact_matches(query: str, property_id: int) -> List[Dict]:
    """Fast exact match for common queries."""
    try:
        exact_sql = """
        SELECT 
            CHUNK AS snippet,
            CHUNK_INDEX AS chunk_index,
            RELATIVE_PATH AS path,
            1.0 AS confidence,
            'exact' AS match_type
        FROM TEST_DB.CORTEX.RAW_TEXT
        WHERE PROPERTY_ID = ?
          AND LOWER(CHUNK) LIKE LOWER(?)
        LIMIT 3
        """
        
        df = session.sql(exact_sql, params=[property_id, f"%{query}%"]).collect()
        return [{'snippet': row['SNIPPET'], 
                'confidence': row['CONFIDENCE'],
                'match_type': row['MATCH_TYPE']} for row in df]
    except Exception:
        return []

# ——— Advanced Retrieval with Parallel Processing ———
def retrieve_relevant_context(enriched_q: str, property_id: int, config: dict):
    """Enhanced retrieval with weighted keyword search and parallel processing."""
    try:
        # Check for exact matches first
        exact_matches = find_exact_matches(enriched_q[:50], property_id)
        if exact_matches and exact_matches[0]['confidence'] > 0.9:
            log_execution("⚡ Exact Match Found", "Using fast path")
            monitor.log_cache_hit()
            return ([m['snippet'] for m in exact_matches], 
                   [0, 1, 2], 
                   ['exact_match'] * 3,
                   [1.0] * 3,
                   ['exact'] * 3)
        
        log_execution("🔍 Starting Retrieval", f"Property {property_id}, Top-K: {config['top_k']}")
        
        start_time = time.time()
        key_terms = extract_key_terms(enriched_q)
        is_appliance_query = detect_appliance_query(enriched_q)
        
        log_execution("🔑 Query Analysis", f"Appliance query: {is_appliance_query}, Key terms: '{key_terms}'")
        
        if is_appliance_query:
            log_execution("🏠 Using Appliance Strategy", "Keyword-weighted search with filename prioritization")
            semantic_sql = f"""
            WITH semantic_results AS (
              SELECT
                r.CHUNK AS snippet,
                r.CHUNK_INDEX AS chunk_index,
                r.RELATIVE_PATH AS path,
                VECTOR_COSINE_SIMILARITY(
                  r.EMBEDDINGS,
                  {EMBED_FN}('{EMBED_MODEL}', ?)
                ) AS similarity,
                'semantic' AS search_type,
                0.3 AS weight_multiplier
              FROM TEST_DB.CORTEX.RAW_TEXT r
              WHERE PROPERTY_ID = ?
              ORDER BY similarity DESC
              LIMIT ?
            ),
            keyword_results AS (
              SELECT
                r.CHUNK AS snippet,
                r.CHUNK_INDEX AS chunk_index,
                r.RELATIVE_PATH AS path,
                CASE 
                  WHEN CONTAINS(UPPER(r.RELATIVE_PATH), UPPER(?)) THEN 0.95
                  WHEN CONTAINS(UPPER(r.CHUNK), UPPER(?)) THEN 0.8
                  ELSE 0.6
                END AS similarity,
                'keyword' AS search_type,
                1.0 AS weight_multiplier
              FROM TEST_DB.CORTEX.RAW_TEXT r
              WHERE PROPERTY_ID = ?
                AND (
                  CONTAINS(UPPER(r.CHUNK), UPPER(?)) OR
                  CONTAINS(UPPER(r.RELATIVE_PATH), UPPER(?))
                )
              LIMIT ?
            ),
            combined_results AS (
              SELECT *, (similarity * weight_multiplier) AS weighted_score FROM semantic_results
              UNION ALL
              SELECT *, (similarity * weight_multiplier) AS weighted_score FROM keyword_results
            )
            SELECT 
              snippet, chunk_index, path, similarity, search_type,
              weighted_score
            FROM combined_results
            ORDER BY weighted_score DESC
            LIMIT ?
            """
            
            params = [
                enriched_q, property_id, config['top_k'],
                key_terms, key_terms,
                property_id, key_terms, key_terms,
                config['top_k'],
                config['top_k']
            ]
        else:
            log_execution("📄 Using General Strategy", "Balanced semantic + keyword search")
            semantic_sql = f"""
            WITH semantic_results AS (
              SELECT
                r.CHUNK AS snippet,
                r.CHUNK_INDEX AS chunk_index,
                r.RELATIVE_PATH AS path,
                VECTOR_COSINE_SIMILARITY(
                  r.EMBEDDINGS,
                  {EMBED_FN}('{EMBED_MODEL}', ?)
                ) AS similarity,
                'semantic' AS search_type,
                1.0 AS weight_multiplier
              FROM TEST_DB.CORTEX.RAW_TEXT r
              WHERE PROPERTY_ID = ?
              ORDER BY similarity DESC
              LIMIT ?
            ),
            keyword_results AS (
              SELECT
                r.CHUNK AS snippet,
                r.CHUNK_INDEX AS chunk_index,
                r.RELATIVE_PATH AS path,
                0.6 AS similarity,
                'keyword' AS search_type,
                0.7 AS weight_multiplier
              FROM TEST_DB.CORTEX.RAW_TEXT r
              WHERE PROPERTY_ID = ?
                AND (
                  CONTAINS(UPPER(r.CHUNK), UPPER(?)) OR
                  CONTAINS(UPPER(r.RELATIVE_PATH), UPPER(?))
                )
              LIMIT 2
            ),
            combined_results AS (
              SELECT *, (similarity * weight_multiplier) AS weighted_score FROM semantic_results
              UNION ALL
              SELECT *, (similarity * weight_multiplier) AS weighted_score FROM keyword_results
            )
            SELECT 
              snippet, chunk_index, path, similarity, search_type,
              weighted_score
            FROM combined_results
            ORDER BY weighted_score DESC
            LIMIT ?
            """
            
            params = [
                enriched_q, property_id, config['top_k'],
                property_id, key_terms, key_terms,
                config['top_k']
            ]
        
        db_start = time.time()
        df = session.sql(semantic_sql, params=params).collect()
        db_time = time.time() - db_start
        
        log_execution("🗄️ Database Query", f"Retrieved {len(df)} chunks", db_time)
        monitor.log_cache_miss()
        
        filtered_results = [
            row for row in df 
            if (row['SEARCH_TYPE'] == 'keyword' and row['SIMILARITY'] >= 0.5) or 
               (row['SEARCH_TYPE'] == 'semantic' and row['SIMILARITY'] >= config['similarity_threshold'])
        ]
        
        if not filtered_results:
            filtered_results = df[:3] if df else []
            log_execution("⚠️ Fallback Applied", "No results above threshold, using top 3")
        
        filtered_results = filtered_results[:3]
        
        snippets = [row['SNIPPET'] for row in filtered_results]
        chunk_idxs = [row['CHUNK_INDEX'] for row in filtered_results]
        paths = [row['PATH'] for row in filtered_results]
        similarities = [row['SIMILARITY'] for row in filtered_results]
        search_types = [row['SEARCH_TYPE'] for row in filtered_results]
        
        total_time = time.time() - start_time
        log_execution("✅ Retrieval Complete", f"Found {len(snippets)} relevant chunks", total_time)
        
        return snippets, chunk_idxs, paths, similarities, search_types
        
    except Exception as e:
        log_execution("❌ Retrieval Error", str(e))
        raise ChatError("retrieval_error", 
                       "I'm having trouble finding information. Please try again.",
                       str(e))

def detect_appliance_query(question: str) -> bool:
    """Detect if this is a query about appliances, manuals, or equipment."""
    appliance_keywords = [
        'airfryer', 'air fryer', 'oven', 'microwave', 'dishwasher', 'washing machine', 
        'dryer', 'refrigerator', 'fridge', 'stove', 'cooktop', 'blender', 'toaster',
        'coffee maker', 'manual', 'instructions', 'how to use', 'how do i', 'operate',
        'settings', 'temperature', 'timer', 'program', 'cycle', 'button', 'control'
    ]
    
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in appliance_keywords)

def extract_key_terms(question: str) -> str:
    """Extract key terms from question for keyword search with appliance focus."""
    import re
    
    question_clean = re.sub(r'\b(guest|property|staying|want|understand|how|can|use|the|a|an|is|are|do|does)\b', '', question.lower())
    words = re.findall(r'\b\w{3,}\b', question_clean)
    
    appliance_terms = ['airfryer', 'air', 'fryer', 'oven', 'microwave', 'dishwasher', 'washing', 'machine', 'dryer', 'manual']
    priority_words = [w for w in words if any(term in w for term in appliance_terms)]
    other_words = [w for w in words if not any(term in w for term in appliance_terms)]
    
    key_words = priority_words + other_words[:3]
    return ' '.join(key_words[:5])

# ——— Answer validation ———
def validate_answer(answer: str, question: str, context: List[str]) -> Tuple[bool, Dict[str, Any]]:
    """Validate answer quality before returning."""
    checks = {
        'has_content': len(answer.strip()) > 10,
        'not_just_fallback': "I don't think I can help" not in answer or len(answer) > 100,
        'within_length': count_words(answer) <= st.session_state.config['max_response_words'] * 1.5,
        'no_contradictions': not detect_contradictions(answer),
        'confidence_score': calculate_answer_confidence(answer, context)
    }
    
    is_valid = checks['has_content'] and checks['within_length'] and not checks.get('has_contradictions', False)
    return is_valid, checks

def detect_contradictions(answer: str) -> bool:
    """Basic contradiction detection."""
    answer_lower = answer.lower()
    contradictions = [
        ('yes', 'no'),
        ('allowed', 'not allowed'),
        ('can', "can't"),
        ('available', 'not available')
    ]
    
    for pos, neg in contradictions:
        if pos in answer_lower and neg in answer_lower:
            return True
    return False

def calculate_answer_confidence(answer: str, context: List[str]) -> float:
    """Calculate confidence score for the answer."""
    if not context:
        return 0.0
    
    # Simple heuristic: check how much of the answer content appears in context
    answer_words = set(answer.lower().split())
    context_words = set(' '.join(context).lower().split())
    
    overlap = len(answer_words.intersection(context_words))
    confidence = overlap / len(answer_words) if answer_words else 0.0
    
    return min(confidence, 1.0)

# ——— Word count utility ———
def count_words(text: str) -> int:
    return len(text.split())

# ——— Stage 2: Response refinement with cheaper model ———
def refine_response(original_response: str, original_question: str) -> str:
    """Second LLM call using cheaper model to refine overly long responses."""
    try:
        log_execution("✂️ Starting Refinement", f"Original: {count_words(original_response)} words")
        
        refinement_prompt = (
            EDITOR_PROMPT +
            f"\n\nOriginal question: {original_question}" +
            f"\n\nResponse to refine: {original_response}" +
            f"\n\nIMPORTANT: Keep the warm, friendly tone while being concise. Don't change any facts or create contradictions." +
            f"\n\nRewrite this to directly answer the question while preserving factual accuracy and warmth:"
        )
        
        refine_start = time.time()
        df = session.sql(
            "SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) AS response",
            params=[REFINE_MODEL, refinement_prompt]
        ).collect()
        refine_time = time.time() - refine_start
        
        refined_response = df[0].RESPONSE.strip() if df else original_response
        
        refine_model_name = REFINE_MODEL.split('.')[-1].strip('"')
        log_execution("🤖 Refinement LLM", f"Model: {refine_model_name}", refine_time)
        
        # Basic sense check for contradictions
        original_lower = original_response.lower()
        refined_lower = refined_response.lower()
        
        if (original_lower.startswith('yes') and refined_lower.startswith('no')) or \
           (original_lower.startswith('no') and refined_lower.startswith('yes')):
            log_execution("⚠️ Contradiction Detected", "Using original response instead")
            st.warning("⚠️ Detected potential contradiction in refinement, using original response")
            return original_response
        
        log_execution("✅ Refinement Complete", f"Final: {count_words(refined_response)} words")
        return refined_response
        
    except Exception as e:
        log_execution("❌ Refinement Error", str(e))
        error = ChatError("refinement_error", 
                         "Response refinement encountered an issue.",
                         str(e))
        error.display()
        return original_response

# ——— Enhanced Answer Generation with Two-Stage Processing ———
def get_enhanced_answer(chat_history: list, raw_question: str, property_id: int, config: dict):
    """Enhanced answer generation with two-stage processing using optimized models."""
    try:
        log_execution("🚀 Starting Answer Generation", f"Question: '{raw_question[:50]}...'")
        
        # Check if we should summarize conversation
        if len(chat_history) > 20 and len(chat_history) % 10 == 0:
            summary = summarize_conversation(chat_history)
            if summary:
                log_execution("📋 Conversation Summarized", summary[:100])
        
        enriched_q = process_question(raw_question, property_id, chat_history)
        log_execution("📝 Question Enriched", f"Length: {len(enriched_q)} chars")
        
        snippets, chunk_idxs, paths, similarities, search_types = retrieve_relevant_context(
            enriched_q, property_id, config
        )
        
        if not snippets:
            log_execution("❌ No Context Found", "Returning fallback response")
            fallback = "I don't have specific information about that. Please contact your host for assistance."
            return enriched_q, fallback, [], [], [], [], [], False, 0, 0, ""
        
        # Build conversation context with dynamic window
        window_size = get_dynamic_context_window(chat_history, raw_question)
        window = chat_history[-window_size:]
        convo = "Recent conversation:\n" + "\n".join([
            f"{'Guest' if m['role']=='user' else 'Assistant'}: {m['content']}"
            for m in window
        ])
        convo += f"\nGuest: {raw_question}\nAssistant:"
        
        # Enhanced context presentation
        context_section = f"Property Information (top {len(snippets)} relevant sections):\n"
        for i, (snippet, path, sim, search_type) in enumerate(zip(snippets, paths, similarities, search_types), 1):
            context_section += f"\n[Source {i} - {search_type.title()} search, {path}]:\n{snippet}\n"
        
        system_prompt = get_system_prompt(property_id, config)
        full_prompt = (
            system_prompt
            + "\n\n" + convo
            + "\n\n" + context_section
            + "\n\nProvide a helpful, accurate response based only on the property information above."
        )
        
        log_execution("📄 Prompt Built", f"Total length: {len(full_prompt)} chars, Context chunks: {len(snippets)}")
        
        # Stage 1: Generate initial response with primary model
        stage1_start = time.time()
        df = session.sql(
            "SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) AS response",
            params=[MODEL_NAME, full_prompt]
        ).collect()
        stage1_time = time.time() - stage1_start
        
        initial_response = df[0].RESPONSE.strip() if df else "I'm having trouble generating a response right now."
        
        # Count tokens using Snowflake's token counter
        try:
            token_df = session.sql(
                "SELECT SNOWFLAKE.CORTEX.COUNT_TOKENS(?, ?) AS token_count",
                params=[MODEL_NAME, full_prompt]
            ).collect()
            input_tokens = token_df[0]['TOKEN_COUNT'] if token_df else len(full_prompt.split())
        except:
            # Fallback to rough estimation
            input_tokens = len(full_prompt.split()) * 1.3  # Rough token estimation
        
        primary_model_name = MODEL_NAME.split('.')[-1].strip('"')
        log_execution("🤖 Stage 1 LLM", f"Model: {primary_model_name}, Response: {count_words(initial_response)} words, Input tokens: {int(input_tokens)}", stage1_time)
        
        # Validate answer quality
        is_valid, validation_checks = validate_answer(initial_response, raw_question, snippets)
        if not is_valid:
            log_execution("⚠️ Answer Validation", f"Failed checks: {[k for k,v in validation_checks.items() if not v]}")
        
        # Stage 2: Refine if response is too long and refinement is enabled
        word_count = count_words(initial_response)
        used_refinement = False
        
        if config['enable_refinement'] and word_count > WORD_THRESHOLD:
            log_execution("🔄 Triggering Refinement", f"Words: {word_count} > threshold: {WORD_THRESHOLD}")
            refined_response = refine_response(initial_response, raw_question)
            final_response = refined_response
            used_refinement = True
        else:
            log_execution("⏭️ Skipping Refinement", f"Words: {word_count} ≤ threshold: {WORD_THRESHOLD} or disabled")
            final_response = initial_response
        
        log_execution("🎉 Answer Complete", f"Final response: {count_words(final_response)} words, Refinement used: {used_refinement}")
        
        return enriched_q, final_response, snippets, chunk_idxs, paths, similarities, search_types, used_refinement, word_count, input_tokens, full_prompt
        
    except Exception as e:
        log_execution("❌ Answer Generation Error", str(e))
        error = ChatError("generation_error",
                         "I'm experiencing technical difficulties. Please try again or contact your host.",
                         str(e))
        error.display()
        return raw_question, "I'm experiencing technical difficulties. Please try again or contact your host.", [], [], [], [], [], False, 0, 0, ""

# ——— Stream response (simulation) ———
def stream_response(response: str, placeholder):
    """Simulate streaming response for better UX."""
    words = response.split()
    streamed = []
    
    for i, word in enumerate(words):
        streamed.append(word)
        if i < len(words) - 1:
            placeholder.markdown(' '.join(streamed) + " ▌")
        else:
            placeholder.markdown(' '.join(streamed))
        time.sleep(0.02)  # Small delay for streaming effect

# ——— Enhanced Streamlit App ———
def main():
    # Initialize session state
    if 'property_id' not in st.session_state:
        st.session_state.property_id = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
# Sidebar configuration
    if SHOW_DEBUG_SIDEBAR:
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # Property switcher
            if st.session_state.property_id:
                if st.button("🔄 Switch Property", type="secondary"):
                    st.session_state.property_id = None
                    st.session_state.chat_history = []
                    st.session_state.session_id = str(uuid.uuid4())
                    st.rerun()
            
            # Configuration options
            with st.expander("🔧 Advanced Settings", expanded=False):
                st.session_state.config['enable_refinement'] = st.checkbox(
                    "Enable Response Refinement", 
                    st.session_state.config['enable_refinement'],
                    help="Use secondary model to refine long responses"
                )
                st.session_state.config['enable_caching'] = st.checkbox(
                    "Enable Query Caching", 
                    st.session_state.config['enable_caching'],
                    help="Cache embeddings and results for faster responses"
                )
                st.session_state.config['debug_mode'] = st.checkbox(
                    "Debug Mode",
                    st.session_state.config.get('debug_mode', False),
                    help="Show technical error details"
                )
                
                st.session_state.config['top_k'] = st.slider(
                    "Retrieved Chunks",
                    min_value=3,
                    max_value=10,
                    value=st.session_state.config['top_k'],
                    help="Number of context chunks to retrieve"
                )
                
                st.session_state.config['similarity_threshold'] = st.slider(
                    "Similarity Threshold",
                    min_value=0.1,
                    max_value=0.5,
                    value=st.session_state.config['similarity_threshold'],
                    step=0.05,
                    help="Minimum similarity score for semantic search"
                )
            
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
        with col2:
            if st.button("🗑️ Clear Logs"):
                st.session_state.execution_log = []
                st.rerun()
        
        # Performance metrics
        with st.expander("📊 Performance Metrics", expanded=True):
            metrics = monitor.get_dashboard_metrics()
            if metrics.get('status') == 'No data yet':
                st.info("No performance data yet. Start chatting to see metrics!")
            else:
                st.metric("Avg Response Time", f"{metrics['avg_response_time']:.2f}s")
                st.metric("Cache Hit Rate", f"{metrics['cache_hit_rate']*100:.1f}%")
                st.metric("Refinement Rate", f"{metrics['refinement_rate']*100:.1f}%")
                st.metric("Total Requests", metrics['total_requests'])
                if metrics['recent_errors'] > 0:
                    st.warning(f"⚠️ {metrics['recent_errors']} recent errors")
    
    # Property selection
    if st.session_state.property_id is None:
        st.title("🏠 Welcome to Property Assistant")
        st.markdown("### Please select your property to get started")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            prop_input = st.number_input(
                "Enter your Property ID", 
                min_value=1, 
                max_value=100, 
                value=1, 
                step=1,
                help="You can find your Property ID in your booking confirmation"
            )
            if st.button("🚀 Start Chat", type="primary", use_container_width=True):
                st.session_state.property_id = prop_input
                try:
                    with st.spinner("Initializing system..."):
                        if not initialize_system():
                            st.error("Failed to initialize system. Please try again.")
                            return
                except Exception as e:
                    error = ChatError("init_error", 
                                    "Failed to initialize. Please check your connection and try again.",
                                    str(e))
                    error.display()
                    return
                st.rerun()
        return
    
    # Main chat interface
    primary_model = MODEL_NAME.split('.')[-1].strip('"')
    refine_model = REFINE_MODEL.split('.')[-1].strip('"')
    st.title(f"🏡 Property #{st.session_state.property_id} Assistant")
    st.caption(f"Session ID: {st.session_state.session_id[:8]}... | Models: {primary_model} → {refine_model}")
    
    # Initialize welcome message only once
    if not st.session_state.chat_history:
        welcome = (
            f"Welcome to Property #{st.session_state.property_id}! I'm your virtual concierge assistant. "
            f"I can help you with information about your property, local recommendations, amenities, and more. "
            f"What would you like to know?"
        )
        st.session_state.chat_history.append({"role": "assistant", "content": welcome})
        log_turn("assistant", welcome, {"type": "welcome"})

    # Display chat history
    for msg in st.session_state.chat_history:
        avatar = "🙋‍♂️" if msg['role'] == 'user' else "🏠"
        st.chat_message(msg['role'], avatar=avatar).write(msg['content'])

    # Chat input
    raw_q = st.chat_input("Ask me anything about your property...")
    if raw_q:
        # Clear execution log for new query
        st.session_state.execution_log = []
        log_execution("🎬 New Query Started", f"User asked: '{raw_q}'")
        
        # Add user message to history and display immediately
        st.session_state.chat_history.append({"role": "user", "content": raw_q})
        st.chat_message("user", avatar="🙋‍♂️").write(raw_q)
        log_turn("user", raw_q)
        
        # Create placeholder for streaming response
        response_placeholder = st.chat_message("assistant", avatar="🏠").empty()
        
        # Generate response
        with st.spinner("🤔 Thinking..."):
            start = time.time()
            try:
                (enriched_q, answer, snippets, chunk_idxs, paths, similarities, 
                 search_types, used_refinement, original_word_count, input_tokens, 
                 full_prompt) = get_enhanced_answer(
                    st.session_state.chat_history[:-1],  # Exclude the just-added user message
                    raw_q,
                    st.session_state.property_id,
                    st.session_state.config
                )
                latency = time.time() - start
            except Exception as e:
                latency = time.time() - start
                error = ChatError("unexpected_error",
                                "Something unexpected happened. Please try again.",
                                str(e))
                error.display()
                answer = "I apologize, but I encountered an error. Please try again or contact your host."
                snippets = []
        
        log_execution("🏁 Query Complete", f"Total time: {latency:.3f}s")
        
        # Stream the response for better UX
        stream_response(answer, response_placeholder)
        
        # Add assistant response to history
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        
        # Log metrics
        metrics = {
            "latency": latency,
            "sources_used": len(snippets),
            "avg_similarity": sum(similarities) / len(similarities) if similarities else 0,
            "used_refinement": used_refinement,
            "original_word_count": original_word_count,
            "final_word_count": count_words(answer),
            "tokens": int(input_tokens) if 'input_tokens' in locals() else 0
        }
        log_turn("assistant", answer, metrics)
        monitor.log_request(metrics)
        
        # Store debug info in session state
        st.session_state.last_debug_info = {
            "latency": latency,
            "snippets": snippets,
            "chunk_idxs": chunk_idxs,
            "paths": paths,
            "similarities": similarities,
            "search_types": search_types,
            "input_tokens": int(input_tokens) if 'input_tokens' in locals() else 0,
            "used_refinement": used_refinement,
            "original_word_count": original_word_count,
            "answer": answer,
            "full_prompt": full_prompt if 'full_prompt' in locals() else "",
            "MODEL_NAME": MODEL_NAME,
            "validation_checks": validate_answer(answer, raw_q, snippets)[1] if snippets else {}
        }
    
    # Enhanced debug info (only show if we have recent debug data)
    if SHOW_DEBUG_SIDEBAR and hasattr(st.session_state, 'last_debug_info') and st.session_state.last_debug_info:
        debug_info = st.session_state.last_debug_info
        
        with st.sidebar:
            # Performance Info
            with st.expander("🔍 Last Query Details", expanded=False):
                st.markdown(f"**Response time:** {debug_info['latency']:.2f}s")
                st.markdown(f"**Chunks retrieved:** {len(debug_info.get('snippets', []))}")
                st.markdown(f"**Input tokens:** {debug_info['input_tokens']:,}")
                
                if debug_info['used_refinement']:
                    primary_model = debug_info['MODEL_NAME'].split('.')[-1].strip('"')
                    refine_model = REFINE_MODEL.split('.')[-1].strip('"')
                    st.markdown(f"**Refinement:** ✅ {primary_model} → {refine_model}")
                    st.markdown(f"**Word reduction:** {debug_info['original_word_count']} → {count_words(debug_info['answer'])} words")
                else:
                    st.markdown(f"**Refinement:** ❌ Not needed ({debug_info['original_word_count']} words)")
                
                # Validation checks
                if debug_info.get('validation_checks'):
                    st.markdown("**Quality Checks:**")
                    for check, passed in debug_info['validation_checks'].items():
                        emoji = "✅" if passed else "❌"
                        st.markdown(f"{emoji} {check.replace('_', ' ').title()}")
            
            # Retrieved sources
            if debug_info.get('snippets'):
                with st.expander("📚 Retrieved Sources", expanded=False):
                    for i, (idx, sim, path, stype) in enumerate(zip(
                        debug_info.get('chunk_idxs', []),
                        debug_info.get('similarities', []),
                        debug_info.get('paths', []),
                        debug_info.get('search_types', [])
                    ), 1):
                        st.markdown(f"**Source {i}** ({stype})")
                        st.markdown(f"- File: `{path}`")
                        st.markdown(f"- Chunk: #{idx}, Score: {sim:.3f}")
                        if i <= len(debug_info.get('snippets', [])):
                            with st.container():
                                st.text(debug_info['snippets'][i-1][:200] + "...")
            
            # Full prompt viewer (debug mode only)
            if st.session_state.config.get('debug_mode') and debug_info.get('full_prompt'):
                with st.expander("📝 Full Prompt (Debug)", expanded=False):
                    st.code(debug_info['full_prompt'], language="text")
                    st.markdown(f"**Total length:** {len(debug_info['full_prompt']):,} chars")
    
    # Real-time execution log
    # Real-time execution log
    if SHOW_DEBUG_SIDEBAR:
        with st.sidebar.expander("📋 Execution Log", expanded=False):
            if st.session_state.execution_log:
                # Add search/filter capability
                search_term = st.text_input("Filter logs", placeholder="Search...")
                filtered_logs = [
                    log for log in st.session_state.execution_log 
                    if not search_term or search_term.lower() in log['step'].lower() or search_term.lower() in log['details'].lower()
                ]
                
                for log_entry in reversed(filtered_logs[-20:]):  # Show last 20 entries, newest first
                    timing_info = f" ({log_entry['timing']})" if log_entry['timing'] else ""
                    st.markdown(f"**{log_entry['timestamp']}** {log_entry['step']}{timing_info}")
                    if log_entry['details']:
                        st.markdown(f"  ↳ {log_entry['details']}")
            else:
                st.markdown("*No execution data yet. Ask a question to see the process!*")

if __name__ == "__main__":
    main()