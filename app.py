import streamlit as st
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import Session
import time
import json
import uuid
from datetime import datetime
import logging
import hashlib
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import re
import os
from groq import Groq


# ——— Conversation Logging ———
# To check the schema of the messages table in Snowflake, run:
# DESC TABLE CHAT_MESSAGES;
class ConversationLogger:
    """Minimal conversation logging using VARIANT for metadata."""
    def __init__(self, session: Session):
        self.session = session
        self.table_name = "CHAT_CONVERSATIONS"
        self.messages_table = "CHAT_MESSAGES"
        self._ensure_tables_exist()

    def _ensure_tables_exist(self):
        try:
            conversations_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                CONVERSATION_ID VARCHAR PRIMARY KEY,
                PROPERTY_ID INTEGER,
                SESSION_ID VARCHAR,
                START_TIME TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
                END_TIME TIMESTAMP_NTZ,
                STATUS VARCHAR DEFAULT 'ACTIVE'
            )
            """
            messages_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.messages_table} (
                MESSAGE_ID VARCHAR PRIMARY KEY,
                CONVERSATION_ID VARCHAR,
                PROPERTY_ID INTEGER,
                MESSAGE_ORDER INTEGER,
                ROLE VARCHAR,
                CONTENT TEXT,
                METADATA VARIANT,
                CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )
            """
            self.session.sql(conversations_sql).collect()
            self.session.sql(messages_sql).collect()
        except Exception:
            pass

    def start_conversation(self, property_id: int, session_id: str) -> str:
        conversation_id = str(uuid.uuid4())
        insert_sql = f"""
        INSERT INTO {self.table_name} (
            CONVERSATION_ID, PROPERTY_ID, SESSION_ID
        ) VALUES (?, ?, ?)
        """
        try:
            self.session.sql(insert_sql, params=[conversation_id, property_id, session_id]).collect()
            return conversation_id
        except Exception:
            return None

    def log_message(self, conversation_id: str, role: str, content: str, metadata: dict = None, property_id: int = None) -> bool:
        if not conversation_id:
            return False
        message_id = str(uuid.uuid4())
        # Get message order (1-based)
        try:
            count_sql = f"SELECT COUNT(*) as msg_count FROM {self.messages_table} WHERE CONVERSATION_ID = ?"
            count_result = self.session.sql(count_sql, params=[conversation_id]).collect()
            message_order = count_result[0].MSG_COUNT + 1 if count_result else 1
        except Exception:
            message_order = 1
        insert_sql = f"""
        INSERT INTO {self.messages_table} (
            MESSAGE_ID, CONVERSATION_ID, PROPERTY_ID, MESSAGE_ORDER, ROLE, CONTENT, METADATA
        ) SELECT ?, ?, ?, ?, ?, ?, PARSE_JSON(?)
        """
        try:
            params = [
                message_id,
                conversation_id,
                property_id,
                message_order,
                role,
                content,
                json.dumps(metadata or {})
            ]
            self.session.sql(insert_sql, params=params).collect()
            return True
        except Exception as e:
            log_execution("❌ Snowflake log_message error", str(e))
            return False

    def end_conversation(self, conversation_id: str):
        if not conversation_id:
            return
        update_sql = f"""
        UPDATE {self.table_name}
        SET END_TIME = CURRENT_TIMESTAMP(), STATUS = 'COMPLETED'
        WHERE CONVERSATION_ID = ?
        """
        try:
            self.session.sql(update_sql, params=[conversation_id]).collect()
        except Exception:
            pass

    def get_conversation_history(self, property_id: int, limit: int = 10) -> List[Dict]:
        query_sql = f"""
        SELECT 
            CONVERSATION_ID,
            SESSION_ID,
            START_TIME,
            END_TIME,
            STATUS
        FROM {self.table_name}
        WHERE PROPERTY_ID = ?
        ORDER BY START_TIME DESC
        LIMIT ?
        """
        try:
            results = self.session.sql(query_sql, params=[property_id, limit]).collect()
            def row_to_dict(row):
                if hasattr(row, 'as_dict'):
                    return row.as_dict()
                return {
                    "CONVERSATION_ID": getattr(row, "CONVERSATION_ID", None),
                    "SESSION_ID": getattr(row, "SESSION_ID", None),
                    "START_TIME": getattr(row, "START_TIME", None),
                    "END_TIME": getattr(row, "END_TIME", None),
                    "STATUS": getattr(row, "STATUS", None)
                }
            return [row_to_dict(row) for row in results]
        except Exception:
            return []

    def get_conversation_messages(self, conversation_id: str) -> List[Dict]:
        query_sql = f"""
        SELECT 
            MESSAGE_ORDER,
            ROLE,
            CONTENT,
            PROPERTY_ID,
            METADATA,
            CREATED_AT
        FROM {self.messages_table}
        WHERE CONVERSATION_ID = ?
        ORDER BY MESSAGE_ORDER
        """
        try:
            results = self.session.sql(query_sql, params=[conversation_id]).collect()
            def row_to_dict(row):
                if hasattr(row, 'as_dict'):
                    return row.as_dict()
                return {
                    "MESSAGE_ORDER": getattr(row, "MESSAGE_ORDER", None),
                    "ROLE": getattr(row, "ROLE", None),
                    "CONTENT": getattr(row, "CONTENT", None),
                    "PROPERTY_ID": getattr(row, "PROPERTY_ID", None),
                    "METADATA": getattr(row, "METADATA", None),
                    "CREATED_AT": getattr(row, "CREATED_AT", None)
                }
            return [row_to_dict(row) for row in results]
        except Exception:
            return []


# ——— App config ———
st.set_page_config(
    page_title="Multi-Property RAG Chat", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ——— Initialize Groq client ———
@st.cache_resource
def get_groq_client():
    """Initialize Groq client with API key."""
    api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("groq", {}).get("api_key")
    if not api_key:
        st.error("⚠️ Groq API key not found. Please set GROQ_API_KEY environment variable or add to Streamlit secrets.")
        return None
    return Groq(api_key=api_key)

groq_client = get_groq_client()

# ——— Initialize Snowpark session ———
@st.cache_resource
def get_session():
    """Create and cache Snowpark session."""
    try:
        # Try to get active session first (for Snowflake environment)
        return get_active_session()
    except:
        # Fallback for local development
        connection_parameters = {
            "account": st.secrets["snowflake"]["account"],
            "user": st.secrets["snowflake"]["user"],
            "password": st.secrets["snowflake"]["password"],
            "role": st.secrets["snowflake"]["role"],
            "warehouse": st.secrets["snowflake"]["warehouse"],
            "database": st.secrets["snowflake"]["database"],
            "schema": st.secrets["snowflake"]["schema"]
        }
        return Session.builder.configs(connection_parameters).create()

session = get_session()

# ——— Initialize Conversation Logger ———
@st.cache_resource
def get_conversation_logger():
    """Initialize conversation logger."""
    return ConversationLogger(session)

conversation_logger = get_conversation_logger()

# ——— Constants ———
MODEL_NAME = 'llama3-70b-8192'  # Updated Groq model name (without context size)
FALLBACK_MODEL = 'MIXTRAL-8X7B'  # Snowflake Cortex fallback
EMBED_MODEL = 'SNOWFLAKE-ARCTIC-EMBED-L-V2.0'
EMBED_FN = 'SNOWFLAKE.CORTEX.EMBED_TEXT_1024'
WORD_THRESHOLD = 100  # Increased from 50 to 100
TOP_K = 5  # Fixed value, no longer configurable
SIMILARITY_THRESHOLD = 0.2  # Fixed value, no longer configurable

# ——— Configuration ———
if 'config' not in st.session_state:
    st.session_state.config = {
        'max_response_words': 100,
        'context_window': 4,
        'enable_logging': True
    }

# ——— Performance Monitor ———
class PerformanceMonitor:
    def __init__(self):
        if 'performance_metrics' not in st.session_state:
            st.session_state.performance_metrics = {
                'response_times': [],
                'retrieval_times': [],
                'refinement_count': 0,
                'total_requests': 0,
                'errors': []
            }
        self.metrics = st.session_state.performance_metrics
    
    def log_request(self, metrics: Dict[str, Any]):
        self.metrics['response_times'].append(metrics.get('latency', 0))
        self.metrics['retrieval_times'].append(metrics.get('retrieval_time', 0))
        self.metrics['total_requests'] += 1
        if metrics.get('used_refinement'):
            self.metrics['refinement_count'] += 1
        
        # Keep only last 100 entries
        if len(self.metrics['response_times']) > 100:
            self.metrics['response_times'] = self.metrics['response_times'][-100:]
            self.metrics['retrieval_times'] = self.metrics['retrieval_times'][-100:]
    
    def log_error(self, error_type: str, details: str):
        self.metrics['errors'].append({
            'timestamp': datetime.now().isoformat(),
            'type': error_type,
            'details': details
        })
        if len(self.metrics['errors']) > 50:
            self.metrics['errors'] = self.metrics['errors'][-50:]
    
    def get_dashboard_metrics(self) -> Dict[str, Any]:
        if not self.metrics['response_times']:
            return {'status': 'No data yet'}
        
        return {
            'avg_response_time': np.mean(self.metrics['response_times']),
            'avg_retrieval_time': np.mean(self.metrics['retrieval_times']) if self.metrics['retrieval_times'] else 0,
            'p95_response_time': np.percentile(self.metrics['response_times'], 95) if len(self.metrics['response_times']) > 10 else 0,
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
        if self.technical_details:
            with st.expander("Technical details"):
                st.code(self.technical_details)

# ——— Execution Logging ———
if 'execution_log' not in st.session_state:
    st.session_state.execution_log = []

def log_execution(step: str, details: str = "", timing: float = None):
    """Log execution steps for debugging."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    log_entry = {
        "timestamp": timestamp,
        "step": step,
        "details": details,
        "timing": f"{timing:.3f}s" if timing else ""
    }
    st.session_state.execution_log.append(log_entry)
    
    if len(st.session_state.execution_log) > 50:
        st.session_state.execution_log = st.session_state.execution_log[-50:]

# ——— System Initialization ———
def optimize_warehouse():
    """Set warehouse for retrieval operations."""
    try:
        # Use RETRIEVAL warehouse for all operations
        session.sql("USE WAREHOUSE RETRIEVAL").collect()
        session.sql("ALTER SESSION SET USE_CACHED_RESULT = TRUE").collect()
        session.sql("ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = 300").collect()
        return True
    except Exception as e:
        logging.error(f"Warehouse optimization failed: {e}")
        return False

# Initialize warehouse on startup
if 'warehouse_initialized' not in st.session_state:
    st.session_state.warehouse_initialized = optimize_warehouse()

# ——— System Prompt ———
def get_system_prompt(property_id: int) -> str:
    """Generate system prompt for the model."""
    return json.dumps({
        "role": "system",
        "content": {
            "persona": "helpful, warm property expert",
            "tone": "short, friendly sentences",
            "focus_rule": "Answer only the most recent guest request",
            "fallback_response": "I'm sorry, I don't have that information. Please contact your host for assistance.",
            "response_constraints": {
                "format": "plain text only",
                "length_limit": f"max {st.session_state.config['max_response_words']} words",
                "no_hallucination": "Only use information from the provided context"
            }
        }
    })

# ——— Refinement Prompt ———
EDITOR_PROMPT = json.dumps({
    "role": "editor",
    "task": "Make the response more concise while keeping all facts",
    "rules": [
        "Keep the warm, friendly tone",
        "Preserve all factual information",
        "Remove redundancy and filler",
        "Maximum 50 words unless more detail is essential"
    ]
})

# ——— Content Sanitization ———
def sanitize_for_markdown(content: str) -> str:
    """Sanitize content to prevent markdown rendering issues."""
    if not content:
        return ""
    
    # Don't sanitize at all - let Streamlit handle markdown safely
    # This prevents the regex errors on mobile
    return content

# ——— Question Processing ———
def process_question(raw_q: str, property_id: int, chat_history: list) -> str:
    """Process and enrich the user question with smart context detection."""
    # Simple enrichment - add property context
    enriched = f"Guest inquiry for Property #{property_id}: {raw_q.strip()}"
    
    # Smart context detection using entity tracking and explicit reference patterns
    if len(chat_history) > 1:
        raw_lower = raw_q.lower()
        
        # 1. Check for explicit reference patterns that indicate follow-up
        # Using simpler patterns that are more mobile-compatible
        explicit_patterns = [
            r'\b(it|this|that)\s+(is|was|does|can|will|should|would)',  # "how does it work", "what is this"
            r'\bwhat\s+about\s+(it|this|that|them)\b',  # "what about it"
            r'\b(tell|explain|show)\s+me\s+more\b',  # "tell me more"
            r'\belse\s+about\b',  # "what else about"
            r'\bthe\s+same\s+',  # "the same thing"
            r'\balso\b.*\?',  # questions with "also"
            r'^(and|but|so)\s+',  # starts with conjunctions
            r'\b(how|why|when|where)\s+do\s+(i|you)\s+(use|turn|activate|access)\s+(it|this|that|them)\b'  # specific action questions
        ]
        
        has_explicit_reference = any(re.search(pattern, raw_lower) for pattern in explicit_patterns)
        
        # 2. Entity/topic tracking - extract key nouns from previous exchange
        if not has_explicit_reference and len(chat_history) >= 2:
            # Get the last user question and assistant response
            last_user_msg = next((msg['content'] for msg in reversed(chat_history[:-1]) if msg['role'] == 'user'), "")
            last_assistant_msg = next((msg['content'] for msg in reversed(chat_history) if msg['role'] == 'assistant'), "")
            
            # Extract meaningful entities (nouns/topics) from previous exchange
            # Focus on domain-specific terms that are likely to be referenced
            entity_patterns = [
                r'\b(pool|spa|hot tub|jacuzzi)\b',
                r'\b(towel|linen|sheet|blanket)s?\b',
                r'\b(kitchen|bedroom|bathroom|living room|garage|patio|deck|balcony)\b',
                r'\b(wifi|internet|password|network)\b',
                r'\b(parking|car|vehicle|garage|driveway)\b',
                r'\b(key|lock|door|gate|access|code)\b',
                r'\b(appliance|dishwasher|washer|dryer|oven|stove|microwave|refrigerator|fridge)\b',
                r'\b(induction|cooktop|burner)\b',
                r'\b(tv|television|remote|cable|streaming)\b',
                r'\b(heat|ac|air|thermostat|temperature)\b',
                r'\b(trash|garbage|recycling|bin)\b',
                r'\b(checkout|checkin|arrival|departure)\b'
            ]
            
            # Find entities in previous messages
            previous_entities = set()
            for pattern in entity_patterns:
                previous_entities.update(re.findall(pattern, last_user_msg.lower()))
                previous_entities.update(re.findall(pattern, last_assistant_msg.lower()))
            
            # Check if current question mentions any previous entities
            current_entities = set()
            for pattern in entity_patterns:
                current_entities.update(re.findall(pattern, raw_lower))
            
            # If there's entity overlap, it might be a follow-up
            has_entity_overlap = bool(previous_entities & current_entities)
        else:
            has_entity_overlap = False
        
        # 3. Apply context only if we have strong signals
        if has_explicit_reference or (has_entity_overlap and len(raw_q.split()) < 10):
            # Get the most relevant previous content
            last_exchange = chat_history[-2:] if len(chat_history) >= 2 else chat_history
            last_content = last_exchange[-1]['content']
            
            # Only add context if the last message was substantial and not a greeting
            if len(last_content) > 30 and not any(greeting in last_content.lower() for greeting in ['welcome', 'hello', 'hi there']):
                # Extract the most relevant part of the previous message
                context_preview = last_content[:50].replace('\n', ' ').strip()
                if not context_preview.endswith('.'):
                    context_preview += '...'
                context = f" (Following up on: {context_preview})"
                enriched += context
    
    return enriched

# ——— Hybrid Retrieval ———
def retrieve_relevant_context(enriched_q: str, property_id: int):
    """Fast, smart hybrid retrieval using semantic and keyword search."""
    try:
        log_execution("🔍 Starting Retrieval", f"Property {property_id}")
        start_time = time.time()
        
        # Ensure we're using RETRIEVAL warehouse
        session.sql("USE WAREHOUSE RETRIEVAL").collect()

        # Step 1: Extract meaningful keyword tokens (≥ 4 chars, deduplicated, lowercased)
        # Exclude common words that appear in every query due to enrichment
        stop_words = {'guest', 'inquiry', 'property', 'discussing', 'context'}
        tokens = re.findall(r'\b\w{4,}\b', enriched_q.lower())
        # Filter out stop words and deduplicate
        keywords = []
        seen = set()
        for token in tokens:
            if token not in stop_words and token not in seen:
                keywords.append(token)
                seen.add(token)
                if len(keywords) >= 5:  # limit to top 5 unique keywords
                    break
        keyword_json = json.dumps(keywords)

        # Step 2: Execute hybrid SQL with embedded keyword logic
        # NOTE: Update EMBEDDINGS column name to match your table schema
        hybrid_sql = f"""
        WITH semantic_results AS (
            SELECT
                CHUNK AS snippet,
                CHUNK_INDEX AS chunk_index,
                RELATIVE_PATH AS path,
                VECTOR_COSINE_SIMILARITY(
                    LABEL_EMBED,
                    {EMBED_FN}('{EMBED_MODEL}', ?)
                ) AS similarity,
                'semantic' AS search_type
            FROM TEST_DB.CORTEX.RAW_TEXT
            WHERE PROPERTY_ID = ?
            AND label_embed IS NOT NULL  -- Only valid, embedded chunks
            ORDER BY similarity DESC
            LIMIT {TOP_K}
        ),
        keyword_results AS (
            SELECT
                CHUNK AS snippet,
                CHUNK_INDEX AS chunk_index,
                RELATIVE_PATH AS path,
                0.48 AS similarity,  -- lowered keyword match score to avoid dominance
                'keyword' AS search_type
            FROM TEST_DB.CORTEX.RAW_TEXT
            WHERE PROPERTY_ID = ?
            AND label_embed IS NOT NULL  -- Only valid, embedded chunks
              AND EXISTS (
                SELECT 1
                FROM TABLE(FLATTEN(INPUT => PARSE_JSON(?))) kw
                WHERE UPPER(CHUNK) LIKE CONCAT('%', UPPER(kw.value), '%')
              )
            LIMIT 2
        )
        SELECT DISTINCT 
            snippet, chunk_index, path, similarity, search_type
        FROM (
            SELECT * FROM semantic_results
            UNION ALL
            SELECT * FROM keyword_results
        )
        WHERE similarity >= {SIMILARITY_THRESHOLD}
        ORDER BY similarity DESC
        LIMIT {TOP_K}
        """

        # Execute the query with parameters
        params = (enriched_q, property_id, property_id, keyword_json)
        results = session.sql(hybrid_sql, params).collect()

        log_execution("✅ Retrieval complete", f"{len(results)} results in {time.time() - start_time:.2f}s")
        return results

    except Exception as e:
        log_execution("❌ Retrieval error", str(e))
        return []

# ——— Answer Generation ———
def get_enhanced_answer(chat_history: list, raw_question: str, property_id: int):
    """Generate answer with optional refinement."""
    try:
        log_execution("🚀 Starting Answer Generation", f"Question: '{raw_question[:50]}...'")
        
        enriched_q = process_question(raw_question, property_id, chat_history)
        
        # Retrieve context
        retrieval_start = time.time()
        results = retrieve_relevant_context(enriched_q, property_id)
        retrieval_time = time.time() - retrieval_start
        
        # Parse results
        snippets = []
        chunk_idxs = []
        paths = []
        similarities = []
        search_types = []
        
        for row in results:
            # Handle both dictionary-style and attribute-style access
            if hasattr(row, 'SNIPPET'):
                snippets.append(row.SNIPPET)
                chunk_idxs.append(row.CHUNK_INDEX)
                paths.append(row.PATH)
                similarities.append(row.SIMILARITY)
                search_types.append(row.SEARCH_TYPE)
            elif isinstance(row, dict):
                snippets.append(row.get('SNIPPET', row.get('snippet', '')))
                chunk_idxs.append(row.get('CHUNK_INDEX', row.get('chunk_index', 0)))
                paths.append(row.get('PATH', row.get('path', '')))
                similarities.append(row.get('SIMILARITY', row.get('similarity', 0)))
                search_types.append(row.get('SEARCH_TYPE', row.get('search_type', '')))
            else:
                # If row is a list/tuple, assume order: snippet, chunk_index, path, similarity, search_type
                if len(row) >= 5:
                    snippets.append(row[0])
                    chunk_idxs.append(row[1])
                    paths.append(row[2])
                    similarities.append(row[3])
                    search_types.append(row[4])
        
        if not snippets:
            fallback = "I don't have specific information about that. Please contact your host for assistance."
            return enriched_q, fallback, [], [], [], [], [], False, 0, retrieval_time
        
        # Build prompt
        context_section = f"Property Information:\n"
        for i, snippet in enumerate(snippets, 1):
            context_section += f"\n[Section {i}]:\n{snippet}\n"
        
        system_prompt = get_system_prompt(property_id)
        full_prompt = (
            system_prompt + "\n\n" +
            f"Guest: {raw_question}\n\n" +
            context_section + "\n\n" +
            "Assistant: Based on the property information above, "
        )
        
        # Generate response
        stage1_start = time.time()
        
        # Try Groq first, fallback to Cortex if needed
        use_groq = st.session_state.config.get('use_groq', True)
        if use_groq and groq_client:
            try:
                # Convert prompt to Groq format
                messages = [
                    {"role": "system", "content": "You are a helpful, warm property expert. Answer only the most recent guest request using the provided property information. Keep responses short and friendly, max 100 words."},
                    {"role": "user", "content": f"Guest: {raw_question}\n\n{context_section}\n\nBased on the property information above, please answer the guest's question."}
                ]
                
                # Call Groq API
                completion = groq_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=200,
                    top_p=0.9,
                    stream=False
                )
                
                initial_response = completion.choices[0].message.content.strip()
                log_execution("🚀 Groq Response", f"Tokens: {completion.usage.total_tokens}", time.time() - stage1_start)
                
            except Exception as e:
                # Fallback to Cortex - DO NOT try Groq again with different model
                log_execution("⚠️ Groq failed, using Cortex", str(e))
                stage1_start = time.time()  # Reset timer for Cortex
                
                # Switch to CORTEX_WH only for LLM generation
                session.sql("USE WAREHOUSE CORTEX_WH").collect()
                
                df = session.sql(
                    "SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) AS response",
                    params=[FALLBACK_MODEL, full_prompt]
                ).collect()
                initial_response = df[0].RESPONSE.strip() if df else "I'm having trouble generating a response."
                log_execution("🤖 Cortex Fallback Response", f"{len(initial_response.split())} words", time.time() - stage1_start)
                
                # Switch back to RETRIEVAL warehouse
                session.sql("USE WAREHOUSE RETRIEVAL").collect()
        else:
            # Use Cortex directly
            # Switch to CORTEX_WH only for LLM generation
            session.sql("USE WAREHOUSE CORTEX_WH").collect()
            
            df = session.sql(
                "SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) AS response",
                params=[FALLBACK_MODEL, full_prompt]
            ).collect()
            initial_response = df[0].RESPONSE.strip() if df else "I'm having trouble generating a response."
            log_execution("🤖 LLM Response", f"{len(initial_response.split())} words", time.time() - stage1_start)
            
            # Switch back to RETRIEVAL warehouse
            session.sql("USE WAREHOUSE RETRIEVAL").collect()
        
        stage1_time = time.time() - stage1_start
        word_count = len(initial_response.split())
        
        log_execution("🤖 LLM Response", f"{word_count} words", stage1_time)
        
        return (enriched_q, initial_response, snippets, chunk_idxs, paths, 
                similarities, search_types, False, word_count, retrieval_time)
        
    except Exception as e:
        log_execution("❌ Generation Error", str(e))
        return raw_question, "I'm experiencing technical difficulties. Please try again.", [], [], [], [], [], False, 0, 0

# ——— Stream Response ———
def stream_response(response: str, placeholder):
    """Simulate streaming for better UX."""
    words = response.split()
    streamed = []
    
    for i, word in enumerate(words):
        streamed.append(word)
        if i < len(words) - 1:
            placeholder.markdown(' '.join(streamed) + " ▌")
        else:
            placeholder.markdown(' '.join(streamed))
        time.sleep(0.02)

# ——— Main App ———
def main():
    # Initialize session state
    if 'property_id' not in st.session_state:
        st.session_state.property_id = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'conversation_id' not in st.session_state:
        st.session_state.conversation_id = None
    if 'message_counter' not in st.session_state:
        st.session_state.message_counter = 0
    if 'last_query_info' not in st.session_state:
        st.session_state.last_query_info = None
    
    # Sidebar
    with st.sidebar:
        st.markdown("📊 System Information")
        
        # Property switcher
        if st.session_state.property_id:
            if st.button("🔄 Switch Property", type="secondary"):
                #
                # End current conversation
                if st.session_state.conversation_id:
                    conversation_logger.end_conversation(st.session_state.conversation_id)
                
                st.session_state.property_id = None
                st.session_state.chat_history = []
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.conversation_id = None
                st.session_state.message_counter = 0
                st.session_state.last_query_info = None
                st.rerun()
        
        # System settings (read-only)
        with st.expander("⚙️ System Configuration", expanded=False):
            st.markdown("Configuration is optimized for performance")
            st.markdown(f"Retrieved chunks: {TOP_K}")
            st.markdown(f"Similarity threshold: {SIMILARITY_THRESHOLD}")
            st.markdown(f"Refinement threshold: {WORD_THRESHOLD} words")
            
            # LLM Provider Toggle
            use_groq = st.checkbox(
                "🚀 Use Groq (95% cheaper!)",
                value=st.session_state.config.get('use_groq', True),
                help="Toggle between Groq and Snowflake Cortex",
                disabled=not groq_client
            )
            st.session_state.config['use_groq'] = use_groq
            
            # LLM Status
            if use_groq and groq_client:
                st.markdown("✅ Using Groq - $0.0017/query")
            elif not groq_client:
                st.markdown("❌ Groq API key not found")
                st.markdown("💰 Using Snowflake Cortex - $0.0375/query")
            else:
                st.markdown("💰 Using Snowflake Cortex - $0.0375/query")
            
            # Conversation logging toggle
            st.session_state.config['enable_conversation_logging'] = st.checkbox(
                "💾 Log Conversations",
                st.session_state.config.get('enable_conversation_logging', True),
                help="Save conversations to Snowflake for analysis"
            )
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat"):
                # End current conversation
                if st.session_state.conversation_id:
                    conversation_logger.end_conversation(st.session_state.conversation_id)
                
                st.session_state.chat_history = []
                st.session_state.message_counter = 0
                st.session_state.last_query_info = None
                st.rerun()
        with col2:
            if st.button("🗑️ Clear Logs"):
                st.session_state.execution_log = []
                st.rerun()
        
        # Performance metrics
        with st.expander("📈 Performance Metrics", expanded=True):
            metrics = monitor.get_dashboard_metrics()
            if metrics.get('status') == 'No data yet':
                st.markdown("Start chatting to see metrics!")
            else:
                st.metric("Avg Response Time", f"{metrics['avg_response_time']:.2f}s")
                st.metric("Avg Retrieval Time", f"{metrics['avg_retrieval_time']:.2f}s")
                st.metric("Refinement Rate", f"{metrics['refinement_rate']*100:.1f}%")
                st.metric("Total Requests", metrics['total_requests'])
                if metrics['recent_errors'] > 0:
                    st.markdown(f"⚠️ {metrics['recent_errors']} recent errors")
        
        # Retrieved Chunks - Always visible after a query
        if st.session_state.last_query_info:
            with st.expander("📚 Retrieved Chunks (Similarity & Type)", expanded=True):
                info = st.session_state.last_query_info
                st.markdown(f"**Total chunks found:** {len(info.get('snippets', []))}")
                
                # Show enriched query
                if info.get('enriched_query') != info.get('raw_query'):
                    st.markdown(f"**Enriched Query:** {info.get('enriched_query', '')}")
                
                if info.get('snippets'):
                    for i, snippet in enumerate(info.get('snippets', []), 1):
                        stype = info.get('search_types', ['unknown'])[i-1] if i-1 < len(info.get('search_types', [])) else 'unknown'
                        sim = info.get('similarities', [0])[i-1] if i-1 < len(info.get('similarities', [])) else 0
                        path = info.get('paths', [''])[i-1] if i-1 < len(info.get('paths', [])) else ''
                        
                        st.markdown(f"**Chunk {i}:**")
                        st.caption(f"Type: {'Cosine (semantic)' if stype == 'semantic' else 'Keyword'} | Similarity: {sim:.3f}")
                        if path:
                            st.caption(f"Source: {path}")
                        st.text_area(f"chunk_{i}_content", snippet, height=100, disabled=True, label_visibility="collapsed")
                        if i < len(info.get('snippets', [])):
                            st.divider()
                else:
                    st.markdown("*No chunks retrieved for this query.*")
                
                # Show performance stats
                if info.get('latency'):
                    st.divider()
                    st.caption(f"**Performance:** Response: {info.get('latency', 0):.2f}s | Retrieval: {info.get('retrieval_time', 0):.2f}s")
        
        # Execution log
        with st.expander("📋 Execution Log", expanded=False):
            if st.session_state.execution_log:
                # Add search/filter capability
                search_term = st.text_input("Filter logs", placeholder="Search...", key="log_search")
                filtered_logs = [
                    log for log in st.session_state.execution_log 
                    if not search_term or search_term.lower() in log['step'].lower() or search_term.lower() in log['details'].lower()
                ]
                
                # Show last 20 entries, newest first
                for log_entry in reversed(filtered_logs[-20:]):
                    timing_info = f" ({log_entry['timing']})" if log_entry['timing'] else ""
                    st.markdown(f"**{log_entry['timestamp']}** {log_entry['step']}{timing_info}")
                    if log_entry['details']:
                        st.markdown(f"  ↳ {log_entry['details']}")
                
                # Summary stats at the bottom
                if filtered_logs:
                    st.divider()
                    total_steps = len(filtered_logs)
                    timed_steps = [float(log['timing'][:-1]) for log in filtered_logs if log['timing']]
                    if timed_steps:
                        st.caption(f"Steps: {total_steps} | Total time: {sum(timed_steps):.3f}s")
            else:
                st.markdown("*No execution data yet. Ask a question to see the process!*")
    
    # Property selection
    if st.session_state.property_id is None:
        st.markdown("## 🏠 Welcome to Property Assistant")
        st.markdown("### Please select your property to get started")
        
        prop_input_text = st.text_input("Property ID", value="1")
        try:
            prop_input = int(prop_input_text) if prop_input_text else 1
        except ValueError:
            prop_input = 1
            
        if st.button("Start Chat"):
            st.session_state.property_id = prop_input
            # Start new conversation
            st.session_state.conversation_id = conversation_logger.start_conversation(
                prop_input, st.session_state.session_id
            )
            st.session_state.message_counter = 0
            st.rerun()
        return
    
    # Main chat interface
    st.markdown(f"## 🏡 Property #{st.session_state.property_id} Assistant")
    st.caption(f"Session: {st.session_state.session_id[:8]}...")
    
    # Welcome message
    if not st.session_state.chat_history:
        welcome = (
            f"Welcome to Property #{st.session_state.property_id}! "
            f"I'm here to help with information about your property. What would you like to know?"
        )
        st.session_state.chat_history.append({"role": "assistant", "content": welcome})

    # Display chat history
    for msg in st.session_state.chat_history:
        avatar = "🙋‍♂️" if msg['role'] == 'user' else "🏠"
        with st.chat_message(msg['role'], avatar=avatar):
            st.markdown(msg['content'])

    # Chat input
    raw_q = st.chat_input("Ask me anything about your property...")
    if raw_q:
        # Clear execution log for new query
        st.session_state.execution_log = []
        log_execution("🎬 New Query Started", f"User asked: '{raw_q}'")
        
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": raw_q})
        with st.chat_message("user", avatar="🙋‍♂️"):
            st.markdown(raw_q)
        
        # Generate response
        response_placeholder = st.chat_message("assistant", avatar="🏠").empty()
        
        with st.spinner("🤔 Thinking..."):
            start = time.time()
            try:
                (enriched_q, answer, snippets, chunk_idxs, paths, similarities, 
                 search_types, used_refinement, word_count, retrieval_time) = get_enhanced_answer(
                    st.session_state.chat_history[:-1],
                    raw_q,
                    st.session_state.property_id
                )
                latency = time.time() - start
                
                # Store query info for sidebar display (always, not just in debug mode)
                st.session_state.last_query_info = {
                    "latency": latency,
                    "retrieval_time": retrieval_time,
                    "snippets": snippets,
                    "chunk_idxs": chunk_idxs,
                    "paths": paths,
                    "similarities": similarities,
                    "search_types": search_types,
                    "used_refinement": used_refinement,
                    "original_word_count": word_count,
                    "enriched_query": enriched_q,
                    "raw_query": raw_q
                }
                
            except Exception as e:
                latency = time.time() - start
                answer = "I apologize, but I encountered an error. Please try again."
                snippets = []
                retrieval_time = 0
                enriched_q = raw_q
                
                # Store minimal info even on error
                st.session_state.last_query_info = {
                    "latency": latency,
                    "retrieval_time": 0,
                    "snippets": [],
                    "error": str(e),
                    "raw_query": raw_q
                }
        
        log_execution("🏁 Query Complete", f"Total time: {latency:.3f}s")
        
        # Stream response
        stream_response(answer, response_placeholder)
        
        # Add to history
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        
        # Log conversation to Snowflake
        if st.session_state.conversation_id and st.session_state.config.get('enable_conversation_logging', True):
            log_execution("💾 Starting Conversation Logging", f"Conversation ID: {st.session_state.conversation_id}")
            
            # Calculate costs (approximate)
            llm_provider = "GROQ" if st.session_state.config.get('use_groq', True) and groq_client else "CORTEX"
            tokens_used = len(answer.split()) * 1.3  # Rough estimate
            cost = tokens_used * 0.0000017 if llm_provider == "GROQ" else tokens_used * 0.0000375
            
            # Log user message
            st.session_state.message_counter += 1
            user_message_data = {
                "message_order": st.session_state.message_counter,
                "role": "user",
                "content": raw_q,
                "enriched_query": enriched_q if 'enriched_q' in locals() else raw_q,
                "retrieval_time": 0.0,
                "llm_response_time": 0.0,
                "total_response_time": 0.0,
                "tokens_used": 0,
                "cost": 0.0,
                "sources_used": 0,
                "similarity_scores": [],
                "source_paths": [],
                "search_types": [],
                "final_word_count": len(raw_q.split()),
                "llm_provider": llm_provider,
                "error_message": ""
            }
            
            user_logged = conversation_logger.log_message(st.session_state.conversation_id, "user", raw_q, user_message_data, st.session_state.property_id)
            log_execution("💾 User Message Logged", f"Success: {user_logged}")
            
            # Log assistant response
            st.session_state.message_counter += 1
            assistant_message_data = {
                "message_order": st.session_state.message_counter,
                "role": "assistant",
                "content": answer,
                "enriched_query": enriched_q if 'enriched_q' in locals() else raw_q,
                "retrieval_time": retrieval_time if 'retrieval_time' in locals() else 0.0,
                "llm_response_time": latency - retrieval_time if 'retrieval_time' in locals() else latency,
                "total_response_time": latency,
                "tokens_used": int(tokens_used),
                "cost": cost,
                "sources_used": len(snippets) if 'snippets' in locals() else 0,
                "similarity_scores": similarities if 'similarities' in locals() else [],
                "source_paths": paths if 'paths' in locals() else [],
                "search_types": search_types if 'search_types' in locals() else [],
                "final_word_count": len(answer.split()),
                "llm_provider": llm_provider,
                "error_message": ""
            }
            
            assistant_logged = conversation_logger.log_message(st.session_state.conversation_id, "assistant", answer, assistant_message_data, st.session_state.property_id)
            log_execution("💾 Assistant Message Logged", f"Success: {assistant_logged}, Cost: ${cost:.4f}")
        
        # Log metrics with all performance data
        metrics = {
            "latency": latency,
            "retrieval_time": retrieval_time,
            "sources_used": len(snippets) if 'snippets' in locals() else 0,
            "final_word_count": len(answer.split()),
            "sources": [{"path": p, "similarity": s, "type": t} for p, s, t in zip(paths, similarities, search_types)] if 'snippets' in locals() and snippets else []
        }
        monitor.log_request(metrics)

if __name__ == "__main__":
    main()