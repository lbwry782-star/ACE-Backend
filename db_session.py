"""
Persistent session storage using SQLite.
This ensures session state (consumed, locked) survives restarts, refreshes, and multiple workers.
"""
import sqlite3
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import os

logger = logging.getLogger(__name__)

# Database file path
DB_PATH = os.environ.get('SESSION_DB_PATH', 'sessions.db')

def get_db_connection():
    """Get a database connection."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database schema."""
    conn = get_db_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS payment_sessions (
                payment_session TEXT PRIMARY KEY,
                paid BOOLEAN NOT NULL DEFAULT 0,
                docnum TEXT,
                doctype TEXT,
                paid_at TIMESTAMP,
                consumed INTEGER NOT NULL DEFAULT 0,
                locked BOOLEAN NOT NULL DEFAULT 0,
                max_quota INTEGER NOT NULL DEFAULT 3,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cookie_bindings (
                cookie_id TEXT PRIMARY KEY,
                payment_session TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (payment_session) REFERENCES payment_sessions(payment_session)
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS page_tokens (
                payment_session TEXT PRIMARY KEY,
                page_token TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (payment_session) REFERENCES payment_sessions(payment_session)
            )
        """)
        
        conn.commit()
        logger.info(f"Database initialized at {DB_PATH}")
    except Exception as e:
        logger.error(f"Error initializing database: {e}", exc_info=True)
        raise
    finally:
        conn.close()

def mark_payment_paid(payment_session: str, docnum: str = '', doctype: str = '') -> None:
    """Mark a payment session as paid."""
    conn = get_db_connection()
    try:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute("""
            INSERT OR REPLACE INTO payment_sessions 
            (payment_session, paid, docnum, doctype, paid_at, updated_at)
            VALUES (?, 1, ?, ?, ?, ?)
        """, (payment_session, docnum, doctype, now, now))
        conn.commit()
        logger.info(f"Payment marked as paid: {payment_session}")
    except Exception as e:
        logger.error(f"Error marking payment as paid: {e}", exc_info=True)
        raise
    finally:
        conn.close()

def is_payment_paid(payment_session: str) -> bool:
    """Check if a payment session is paid."""
    conn = get_db_connection()
    try:
        cursor = conn.execute(
            "SELECT paid FROM payment_sessions WHERE payment_session = ?",
            (payment_session,)
        )
        row = cursor.fetchone()
        return row is not None and bool(row['paid'])
    except Exception as e:
        logger.error(f"Error checking payment status: {e}", exc_info=True)
        return False
    finally:
        conn.close()

def get_quota_info(payment_session: str) -> Optional[Dict[str, Any]]:
    """Get quota information for a payment session."""
    conn = get_db_connection()
    try:
        cursor = conn.execute("""
            SELECT consumed, locked, max_quota 
            FROM payment_sessions 
            WHERE payment_session = ?
        """, (payment_session,))
        row = cursor.fetchone()
        if row:
            return {
                'consumed': row['consumed'],
                'locked': bool(row['locked']),
                'max': row['max_quota']
            }
        return None
    except Exception as e:
        logger.error(f"Error getting quota info: {e}", exc_info=True)
        return None
    finally:
        conn.close()

def init_quota_if_needed(payment_session: str, max_quota: int = 3) -> Dict[str, Any]:
    """Initialize quota for a payment session if it doesn't exist."""
    conn = get_db_connection()
    try:
        # Check if exists
        cursor = conn.execute(
            "SELECT consumed, locked, max_quota FROM payment_sessions WHERE payment_session = ?",
            (payment_session,)
        )
        row = cursor.fetchone()
        
        if row:
            return {
                'consumed': row['consumed'],
                'locked': bool(row['locked']),
                'max': row['max_quota']
            }
        
        # Create new entry
        now = datetime.now(timezone.utc).isoformat()
        conn.execute("""
            INSERT INTO payment_sessions 
            (payment_session, paid, consumed, locked, max_quota, updated_at)
            VALUES (?, 0, 0, 0, ?, ?)
        """, (payment_session, max_quota, now))
        conn.commit()
        logger.info(f"Quota initialized for payment_session: {payment_session}, max={max_quota}")
        return {
            'consumed': 0,
            'locked': False,
            'max': max_quota
        }
    except Exception as e:
        logger.error(f"Error initializing quota: {e}", exc_info=True)
        raise
    finally:
        conn.close()

def consume_quota(payment_session: str) -> int:
    """Consume one quota unit and return new consumed count."""
    conn = get_db_connection()
    try:
        now = datetime.now(timezone.utc).isoformat()
        # Get current consumed
        cursor = conn.execute(
            "SELECT consumed FROM payment_sessions WHERE payment_session = ?",
            (payment_session,)
        )
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Payment session not found: {payment_session}")
        
        new_consumed = row['consumed'] + 1
        conn.execute("""
            UPDATE payment_sessions 
            SET consumed = ?, updated_at = ?
            WHERE payment_session = ?
        """, (new_consumed, now, payment_session))
        conn.commit()
        logger.info(f"Quota consumed for payment_session: {payment_session}, consumed={new_consumed}")
        return new_consumed
    except Exception as e:
        logger.error(f"Error consuming quota: {e}", exc_info=True)
        raise
    finally:
        conn.close()

def lock_session(payment_session: str) -> bool:
    """Lock a session permanently. Returns True if locked, False if already locked."""
    conn = get_db_connection()
    try:
        # Check if already locked
        cursor = conn.execute(
            "SELECT locked FROM payment_sessions WHERE payment_session = ?",
            (payment_session,)
        )
        row = cursor.fetchone()
        if not row:
            return False
        
        if bool(row['locked']):
            return False  # Already locked
        
        # Lock it
        now = datetime.now(timezone.utc).isoformat()
        conn.execute("""
            UPDATE payment_sessions 
            SET locked = 1, updated_at = ?
            WHERE payment_session = ?
        """, (now, payment_session))
        conn.commit()
        logger.info(f"Session locked: {payment_session}")
        return True
    except Exception as e:
        logger.error(f"Error locking session: {e}", exc_info=True)
        return False
    finally:
        conn.close()

def bind_cookie_to_session(cookie_id: str, payment_session: str) -> None:
    """Bind a cookie to a payment session."""
    conn = get_db_connection()
    try:
        conn.execute("""
            INSERT OR REPLACE INTO cookie_bindings (cookie_id, payment_session)
            VALUES (?, ?)
        """, (cookie_id, payment_session))
        conn.commit()
        logger.info(f"Cookie bound: {cookie_id} -> {payment_session}")
    except Exception as e:
        logger.error(f"Error binding cookie: {e}", exc_info=True)
        raise
    finally:
        conn.close()

def get_payment_session_from_cookie(cookie_id: str) -> Optional[str]:
    """Get payment session from cookie ID."""
    conn = get_db_connection()
    try:
        cursor = conn.execute(
            "SELECT payment_session FROM cookie_bindings WHERE cookie_id = ?",
            (cookie_id,)
        )
        row = cursor.fetchone()
        return row['payment_session'] if row else None
    except Exception as e:
        logger.error(f"Error getting payment session from cookie: {e}", exc_info=True)
        return None
    finally:
        conn.close()

def get_cookie_from_payment_session(payment_session: str) -> Optional[str]:
    """Get cookie ID from payment session."""
    conn = get_db_connection()
    try:
        cursor = conn.execute(
            "SELECT cookie_id FROM cookie_bindings WHERE payment_session = ?",
            (payment_session,)
        )
        row = cursor.fetchone()
        return row['cookie_id'] if row else None
    except Exception as e:
        logger.error(f"Error getting cookie from payment session: {e}", exc_info=True)
        return None
    finally:
        conn.close()

def create_page_token(payment_session: str) -> str:
    """Create a new page token for a payment session."""
    import uuid
    conn = get_db_connection()
    try:
        page_token = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        conn.execute("""
            INSERT OR REPLACE INTO page_tokens (payment_session, page_token, created_at)
            VALUES (?, ?, ?)
        """, (payment_session, page_token, now))
        conn.commit()
        logger.info(f"Page token created for payment_session: {payment_session}")
        return page_token
    except Exception as e:
        logger.error(f"Error creating page token: {e}", exc_info=True)
        raise
    finally:
        conn.close()

def validate_page_token(payment_session: str, page_token: Optional[str]) -> bool:
    """Validate a page token for a payment session. Returns True if valid, False otherwise."""
    if not page_token:
        return False
    
    conn = get_db_connection()
    try:
        cursor = conn.execute(
            "SELECT page_token FROM page_tokens WHERE payment_session = ?",
            (payment_session,)
        )
        row = cursor.fetchone()
        if not row:
            return False
        return row['page_token'] == page_token
    except Exception as e:
        logger.error(f"Error validating page token: {e}", exc_info=True)
        return False
    finally:
        conn.close()

def get_page_token(payment_session: str) -> Optional[str]:
    """Get the current page token for a payment session."""
    conn = get_db_connection()
    try:
        cursor = conn.execute(
            "SELECT page_token FROM page_tokens WHERE payment_session = ?",
            (payment_session,)
        )
        row = cursor.fetchone()
        return row['page_token'] if row else None
    except Exception as e:
        logger.error(f"Error getting page token: {e}", exc_info=True)
        return None
    finally:
        conn.close()

