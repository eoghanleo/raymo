import streamlit as st
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import Session
import time
import json
import uuid
from datetime import datetime
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import re
import os
from groq import Groq


# ——— Conversation Logging (Simplified) ———
class ConversationLogger:
    """Handles logging of chat conversations to Snowflake."""
    
    def __init__(self, session: Session):
        self.session = session
        self.table_name = "CHAT_CONVERSATIONS"
        self.messages_table = "CHAT_MESSAGES"
        self._ensure_tables_exist()
    
    def _ensure_tables_exist(self):
        """Create conversation logging tables if they don't exist."""
        try:
            # Create conversations table
            conversations_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                CONVERSATION_ID VARCHAR(36) PRIMARY KEY,
                PROPERTY_ID INTEGER NOT NULL,
                SESSION_ID VARCHAR(36) NOT NULL,
                START_TIME TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
                END_TIME TIMESTAMP_NTZ,
                TOTAL_MESSAGES INTEGER DEFAULT 0,
                STATUS VARCHAR(20) DEFAULT 'ACTIVE'
            )
            """
            
            # Create messages table with VARIANT for arrays
            messages_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.messages_table} (
                MESSAGE_ID VARCHAR(36) PRIMARY KEY,
                CONVERSATION_ID VARCHAR(36) NOT NULL,
                MESSAGE_ORDER INTEGER NOT NULL,
                ROLE VARCHAR(20) NOT NULL,
                CONTENT TEXT NOT NULL,
                RESPONSE_TIME FLOAT,
                SOURCES_USED INTEGER DEFAULT 0,
                METADATA VARIANT,
                CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
                FOREIGN KEY (CONVERSATION_ID) REFERENCES {self.table_name}(CONVERSATION_ID)
            )
            """
            
            self.session.sql(conversations_sql).collect()
            self.session.sql(messages_sql).collect()
            
        except Exception as e:
            logging.error(f"Failed to create conversation tables: {e}")
    
    def start_conversation(self, property_id: int, session_id: str) -> str:
        """Start a new conversation and return conversation ID."""
        conversation_id = str(uuid.uuid4())
        
        try:
            insert_sql = f"""
            INSERT INTO {self.table_name} (
                CONVERSATION_ID, PROPERTY_ID, SESSION_ID
            ) VALUES (?, ?, ?)
            """
            self.session.sql(insert_sql, params=[conversation_id, property_id, session_id]).collect()
            return conversation_id
        except Exception as e:
            logging.error(f"Failed to start conversation: {e}")
            return None
    
    def log_message(self, conversation_id: str, role: str, content: str, 
                   response_time: float = 0, sources_used: int = 0, metadata: dict = None) -> bool:
        """Log a single message to the database."""
        if not conversation_id:
            return False
            
        message_id = str(uuid.uuid4())
        
        # Get current message count
        count_sql = f"SELECT COUNT(*) as msg_count FROM {self.messages_table} WHERE CONVERSATION_ID = ?"
        count_result = self.session.sql(count_sql, params=[conversation_id]).collect()
        message_order = count_result[0].MSG_COUNT + 1 if count_result else 1
        
        try:
            insert_sql = f"""
            INSERT INTO {self.messages_table} (
                MESSAGE_ID, CONVERSATION_ID, MESSAGE_ORDER, ROLE, CONTENT,
                RESPONSE_TIME, SOURCES_USED, METADATA
            ) VALUES (?, ?, ?, ?, ?, ?, ?, PARSE_JSON(?))
            """
            
            params = [
                message_id,
                conversation_id,
                message_order,
                role,
                content,
                response_time,
                sources_used,
                json.dumps(metadata or {})
            ]
            
            self.session.sql(insert_sql, params=params).collect()
            
            # Update conversation message count
            update_sql = f"""
            UPDATE {self.table_name}
            SET TOTAL_MESSAGES = (
                SELECT COUNT(*) FROM {self.messages_table} 
                WHERE CONVERSATION_ID = ?
            )
            WHERE CONVERSATION_ID = ?
            """
            self.session.sql(update_sql, params=[conversation_id, conversation_id]).collect()
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to log message: {e}")
            return False
    
    def end_conversation(self, conversation_id: str):
        """Mark conversation as ended."""
        if not conversation_id:
            return
            
        try:
            update_sql = f"""
            UPDATE {self.table_name}
            SET END_TIME = CURRENT_TIMESTAMP(), STATUS = 'COMPLETED'
            WHERE CONVERSATION_ID = ?
            """
            self.session.sql(update_sql, params=[conversation_id]).collect()
        except Exception as e:
            logging.error(f"Failed to end conversation: {e}")


# ——— App config ———
st.set_page_config(
    page_title="Property Assistant Chat",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ——— Initialize Groq client ———
@st.cache_resource
def get_groq_client():
    """Initialize Groq client with API key."""
    api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("groq", {}).get("api_key")
    if not api_key:
        st.error("⚠️ Groq API key not found. Please set GROQ_API_KEY environment variable.")
        return None
    return Groq(api_key=api_key)

groq_client = get_groq_client()

# ——— Initialize Snowpark session ———
@st.cache_resource
def get_session():
    """Create and cache Snowpark session."""
    try:
        return get_active_session()
    except:
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
MODEL_NAME = 'llama3-70b-8192'
FALLBACK_MODEL = 'MIXTRAL-8X7B'
EMBED_MODEL = 'SNOWFLAKE-ARCTIC-EMBED-L-V2.0'
EMBED_FN = 'SNOWFLAKE.CORTEX.EMBED_TEXT_1024'
WORD_THRESHOLD = 100
TOP_K = 5
SIMILARITY_THRESHOLD = 0.2

# ——— Performance Tracking ———
if 'metrics' not in st.session_state:
    st.session_state.metrics = {
        'total_queries': 0,
        'avg_response_time': 0,
        'response_times': [],
        'last_query_details': None
    }

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

def track_response_time(response_time: float):
    """Track response time metrics."""
    st.session_state.metrics['response_times'].append(response_time)
    st.session_state.metrics['total_queries'] += 1
    # Keep only last 50 times
    if len(st.session_state.metrics['response_times']) > 50:
        st.session_state.metrics['response_times'] = st.session_state.metrics['response_times'][-50:]
    st.session_state.metrics['avg_response_time'] = np.mean(st.session_state.metrics['response_times'])

# ——— System Initialization ———
def optimize_warehouse():
    """Set warehouse for retrieval operations."""
    try:
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

# ——— Question Processing ———
def process_question(raw_q: str, property_id: int, chat_history: list) -> str:
    """Process and enrich the user question."""
    enriched = f"Guest inquiry for Property #{property_id}: {raw_q.strip()}"
    
    # Simple context detection for follow-up questions
    if len(chat_history) > 1:
        raw_lower = raw_q.lower()
        
        # Check for explicit references
        explicit_patterns = [
            r'\b(it|this|that)\s+(is|was|does|can|will|should|would)',
            r'\btell\s+me\s+more\b',
            r'^(and|but|so)\s+',
        ]
        
        if any(re.search(pattern, raw_lower) for pattern in explicit_patterns):
            # Add context from last exchange
            last_exchange = chat_history[-2:] if len(chat_history) >= 2 else chat_history
            if last_exchange:
                last_content = last_exchange[-1]['content'][:50].replace('\n', ' ').strip()
                enriched += f" (Following up on: {last_content}...)"
    
    return enriched

# ——— Retrieval ———
def retrieve_relevant_context(enriched_q: str, property_id: int):
    """Fast hybrid retrieval using semantic and keyword search."""
    try:
        log_execution("🔍 Starting Retrieval", f"Property {property_id}")
        start_time = time.time()
        
        # Extract keywords
        tokens = re.findall(r'\b\w{4,}\b', enriched_q.lower())
        keywords = list(dict.fromkeys(tokens[:5]))  # Keep first 5 unique keywords
        keyword_json = json.dumps(keywords)

        # Hybrid search query
        hybrid_sql = f"""
        WITH semantic_results AS (
            SELECT
                CHUNK AS snippet,
                RELATIVE_PATH AS path,
                VECTOR_COSINE_SIMILARITY(
                    LABEL_EMBED,
                    {EMBED_FN}('{EMBED_MODEL}', ?)
                ) AS similarity,
                'semantic' AS search_type
            FROM TEST_DB.CORTEX.RAW_TEXT
            WHERE PROPERTY_ID = ?
            AND label_embed IS NOT NULL
            ORDER BY similarity DESC
            LIMIT {TOP_K}
        ),
        keyword_results AS (
            SELECT
                CHUNK AS snippet,
                RELATIVE_PATH AS path,
                0.48 AS similarity,
                'keyword' AS search_type
            FROM TEST_DB.CORTEX.RAW_TEXT
            WHERE PROPERTY_ID = ?
            AND label_embed IS NOT NULL
            AND EXISTS (
                SELECT 1
                FROM TABLE(FLATTEN(INPUT => PARSE_JSON(?))) kw
                WHERE UPPER(CHUNK) LIKE CONCAT('%', UPPER(kw.value), '%')
            )
            LIMIT 2
        )
        SELECT DISTINCT snippet, path, similarity, search_type
        FROM (
            SELECT * FROM semantic_results
            UNION ALL
            SELECT * FROM keyword_results
        )
        WHERE similarity >= {SIMILARITY_THRESHOLD}
        ORDER BY similarity DESC
        LIMIT {TOP_K}
        """

        params = (enriched_q, property_id, property_id, keyword_json)
        results = session.sql(hybrid_sql, params).collect()
        
        retrieval_time = time.time() - start_time
        log_execution("✅ Retrieval complete", f"{len(results)} results in {retrieval_time:.2f}s")
        return results, retrieval_time, keywords

    except Exception as e:
        log_execution("❌ Retrieval error", str(e))
        logging.error(f"Retrieval error: {e}")
        return [], 0, []

# ——— Answer Generation ———
def generate_answer(raw_question: str, enriched_q: str, property_id: int, snippets: list):
    """Generate answer using LLM."""
    if not snippets:
        return "I don't have specific information about that. Please contact your host for assistance.", 0
    
    # Build context
    context = "Property Information:\n"
    for i, snippet in enumerate(snippets, 1):
        context += f"\n[Section {i}]:\n{snippet}\n"
    
    # System prompt
    system_content = f"""You are a helpful property assistant for Property #{property_id}.
Answer only the guest's most recent question using the provided property information.
Keep responses short, friendly, and under 100 words.
If the information isn't available, politely say so."""
    
    log_execution("🤖 Starting LLM Generation")
    start_time = time.time()
    
    try:
        if groq_client:
            # Use Groq
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"Guest: {raw_question}\n\n{context}"}
            ]
            
            completion = groq_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.3,
                max_tokens=200,
                stream=False
            )
            
            llm_time = time.time() - start_time
            log_execution("✅ Groq Response", f"Generated in {llm_time:.2f}s")
            return completion.choices[0].message.content.strip(), llm_time
        else:
            # Fallback to Cortex
            session.sql("USE WAREHOUSE CORTEX_WH").collect()
            
            prompt = f"{system_content}\n\nGuest: {raw_question}\n\n{context}\n\nAssistant:"
            
            df = session.sql(
                "SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) AS response",
                params=[FALLBACK_MODEL, prompt]
            ).collect()
            
            session.sql("USE WAREHOUSE RETRIEVAL").collect()
            
            llm_time = time.time() - start_time
            log_execution("✅ Cortex Response", f"Generated in {llm_time:.2f}s")
            response = df[0].RESPONSE.strip() if df else "I'm having trouble generating a response."
            return response, llm_time
            
    except Exception as e:
        log_execution("❌ Generation error", str(e))
        logging.error(f"Generation error: {e}")
        llm_time = time.time() - start_time
        return "I apologize, but I encountered an error. Please try again.", llm_time

# ——— Refine Response ———
def refine_response(response: str, word_count: int) -> Tuple[str, float]:
    """Refine response if too long."""
    if word_count <= WORD_THRESHOLD:
        return response, 0
    
    log_execution("✂️ Starting Refinement", f"Original: {word_count} words")
    start_time = time.time()
    
    try:
        if groq_client:
            messages = [
                {"role": "system", "content": "Make this response more concise (under 50 words) while keeping all key information:"},
                {"role": "user", "content": response}
            ]
            
            completion = groq_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.3,
                max_tokens=100,
                stream=False
            )
            
            refinement_time = time.time() - start_time
            refined = completion.choices[0].message.content.strip()
            log_execution("✅ Refinement complete", f"Final: {len(refined.split())} words in {refinement_time:.2f}s")
            return refined, refinement_time
        else:
            # Use Cortex
            session.sql("USE WAREHOUSE CORTEX_WH").collect()
            
            prompt = f"Make this response more concise (under 50 words) while keeping all key information:\n\n{response}"
            
            df = session.sql(
                "SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) AS response",
                params=[FALLBACK_MODEL, prompt]
            ).collect()
            
            session.sql("USE WAREHOUSE RETRIEVAL").collect()
            
            refinement_time = time.time() - start_time
            refined = df[0].RESPONSE.strip() if df else response
            log_execution("✅ Refinement complete", f"Final: {len(refined.split())} words in {refinement_time:.2f}s")
            return refined, refinement_time
            
    except Exception as e:
        log_execution("❌ Refinement error", str(e))
        logging.error(f"Refinement error: {e}")
        refinement_time = time.time() - start_time
        return response, refinement_time

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
    
    # Sidebar - Simplified
    with st.sidebar:
        st.header("📊 Chat Settings")
        
        # Property switcher
        if st.session_state.property_id:
            if st.button("🔄 Switch Property", type="secondary", use_container_width=True):
                if st.session_state.conversation_id:
                    conversation_logger.end_conversation(st.session_state.conversation_id)
                
                st.session_state.property_id = None
                st.session_state.chat_history = []
                st.session_state.conversation_id = None
                st.rerun()
        
        # Performance metrics
        with st.expander("📊 Performance Metrics", expanded=True):
            if st.session_state.metrics['total_queries'] > 0:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Queries", st.session_state.metrics['total_queries'])
                    st.metric("Avg Response Time", f"{st.session_state.metrics['avg_response_time']:.2f}s")
                
                # Last query details
                if st.session_state.metrics.get('last_query_details'):
                    with col2:
                        details = st.session_state.metrics['last_query_details']
                        st.metric("Last Query Total", f"{details['total_time']:.2f}s")
                        st.metric("Sources Found", details['sources_used'])
                    
                    # Breakdown
                    st.caption("**Last Query Breakdown:**")
                    st.text(f"🔍 Retrieval: {details['retrieval_time']:.3f}s")
                    st.text(f"🤖 LLM Generation: {details['llm_time']:.3f}s")
                    if details.get('refinement_time', 0) > 0:
                        st.text(f"✂️ Refinement: {details['refinement_time']:.3f}s")
                    st.text(f"📝 Word Count: {details['word_count']} → {details['final_word_count']}")
            else:
                st.info("Start chatting to see metrics!")
        
        # Retrieved chunks debug info
        if st.session_state.metrics.get('last_query_details') and st.session_state.metrics['last_query_details'].get('chunks'):
            with st.expander("🔍 Retrieved Chunks Debug", expanded=False):
                details = st.session_state.metrics['last_query_details']
                
                # Query info
                st.markdown("**Enriched Query:**")
                st.text(details['enriched_query'])
                
                # Keywords used
                if details.get('keywords'):
                    st.markdown("**Keywords Extracted:**")
                    st.text(', '.join(details['keywords']))
                
                st.divider()
                
                # Show each chunk
                for i, (chunk, path, sim, stype) in enumerate(zip(
                    details['chunks'], 
                    details['paths'], 
                    details['similarities'], 
                    details['search_types']
                ), 1):
                    # Chunk header with type and score
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.markdown(f"**Chunk {i}: {path}**")
                    with col2:
                        if stype == 'semantic':
                            st.success(f"🧠 {stype}")
                        else:
                            st.info(f"🔤 {stype}")
                    with col3:
                        st.metric("Score", f"{sim:.3f}", label_visibility="collapsed")
                    
                    # Chunk content
                    with st.container():
                        st.text_area(
                            f"chunk_{i}", 
                            chunk, 
                            height=100, 
                            disabled=True,
                            label_visibility="collapsed"
                        )
                    
                    if i < len(details['chunks']):
                        st.divider()
        
        # Execution log
        with st.expander("📋 Execution Log", expanded=False):
            if st.session_state.execution_log:
                # Show last 20 entries, newest first
                for log_entry in reversed(st.session_state.execution_log[-20:]):
                    timing_info = f" ({log_entry['timing']})" if log_entry['timing'] else ""
                    st.markdown(f"**{log_entry['timestamp']}** {log_entry['step']}{timing_info}")
                    if log_entry['details']:
                        st.markdown(f"  ↳ {log_entry['details']}")
            else:
                st.markdown("*No execution data yet. Ask a question to see the process!*")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", type="secondary", use_container_width=True):
            if st.session_state.conversation_id:
                conversation_logger.end_conversation(st.session_state.conversation_id)
            st.session_state.chat_history = []
            st.session_state.execution_log = []
            st.rerun()
        
        # LLM Status
        st.divider()
        if groq_client:
            st.success("✅ Using Groq LLM")
        else:
            st.warning("⚠️ Using Snowflake Cortex")
    
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
                step=1
            )
            if st.button("🚀 Start Chat", type="primary", use_container_width=True):
                st.session_state.property_id = prop_input
                st.session_state.conversation_id = conversation_logger.start_conversation(
                    prop_input, st.session_state.session_id
                )
                st.rerun()
        return
    
    # Main chat interface
    st.title(f"🏡 Property #{st.session_state.property_id} Assistant")
    
    # Welcome message
    if not st.session_state.chat_history:
        welcome = f"Welcome to Property #{st.session_state.property_id}! I'm here to help with information about your property. What would you like to know?"
        st.session_state.chat_history.append({"role": "assistant", "content": welcome})

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
        
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": raw_q})
        st.chat_message("user", avatar="🙋‍♂️").write(raw_q)
        
        # Generate response
        response_placeholder = st.chat_message("assistant", avatar="🏠").empty()
        
        with st.spinner("🤔 Thinking..."):
            start_time = time.time()
            
            # Process question
            enriched_q = process_question(raw_q, st.session_state.property_id, st.session_state.chat_history[:-1])
            
            # Retrieve context
            results, retrieval_time, keywords = retrieve_relevant_context(enriched_q, st.session_state.property_id)
            
            # Extract snippets and metadata
            snippets = []
            paths = []
            similarities = []
            search_types = []
            
            for row in results:
                snippets.append(row.SNIPPET if hasattr(row, 'SNIPPET') else row['snippet'])
                paths.append(row.PATH if hasattr(row, 'PATH') else row['path'])
                similarities.append(row.SIMILARITY if hasattr(row, 'SIMILARITY') else row['similarity'])
                search_types.append(row.SEARCH_TYPE if hasattr(row, 'SEARCH_TYPE') else row['search_type'])
            
            sources_used = len(snippets)
            
            # Generate answer
            answer, llm_time = generate_answer(raw_q, enriched_q, st.session_state.property_id, snippets)
            
            # Refine if needed
            word_count = len(answer.split())
            refinement_time = 0
            final_word_count = word_count
            
            if word_count > WORD_THRESHOLD:
                answer, refinement_time = refine_response(answer, word_count)
                final_word_count = len(answer.split())
            
            total_time = time.time() - start_time
            log_execution("🏁 Query Complete", f"Total time: {total_time:.3f}s")
        
        # Stream response
        stream_response(answer, response_placeholder)
        
        # Add to history
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        
        # Track metrics
        track_response_time(total_time)
        
        # Store last query details for display
        st.session_state.metrics['last_query_details'] = {
            'total_time': total_time,
            'retrieval_time': retrieval_time,
            'llm_time': llm_time,
            'refinement_time': refinement_time,
            'sources_used': sources_used,
            'word_count': word_count,
            'final_word_count': final_word_count,
            'enriched_query': enriched_q,
            'keywords': keywords,
            'chunks': snippets,
            'paths': paths,
            'similarities': similarities,
            'search_types': search_types
        }
        
        # Log to Snowflake
        if st.session_state.conversation_id:
            log_execution("💾 Logging to Snowflake", f"Conversation ID: {st.session_state.conversation_id}")
            
            # Log user message
            user_logged = conversation_logger.log_message(
                st.session_state.conversation_id,
                role="user",
                content=raw_q
            )
            
            if user_logged:
                log_execution("✅ User message logged")
            else:
                log_execution("❌ Failed to log user message")
            
            # Log assistant response
            metadata = {
                "enriched_query": enriched_q,
                "retrieval_time": retrieval_time,
                "llm_time": llm_time,
                "refinement_time": refinement_time,
                "total_time": total_time,
                "sources_used": sources_used,
                "original_word_count": word_count,
                "final_word_count": final_word_count
            }
            
            assistant_logged = conversation_logger.log_message(
                st.session_state.conversation_id,
                role="assistant",
                content=answer,
                response_time=total_time,
                sources_used=sources_used,
                metadata=metadata
            )
            
            if assistant_logged:
                log_execution("✅ Assistant message logged")
            else:
                log_execution("❌ Failed to log assistant message")

if __name__ == "__main__":
    main()