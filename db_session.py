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
            CREATE TABLE IF NOT EXISTS fingerprint_bindings (
                fingerprint_key TEXT PRIMARY KEY,
                payment_session TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                FOREIGN KEY (payment_session) REFERENCES payment_sessions(payment_session)
            )
        """)
        
        # Create index on expires_at for cleanup queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_fingerprint_expires_at 
            ON fingerprint_bindings(expires_at)
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


# Fingerprint-based bindings (weak security, IP + User-Agent based)
def bind_fingerprint_to_session(fingerprint_key: str, payment_session: str, ttl_minutes: int = 45) -> None:
    """
    Bind a fingerprint key to a payment session with TTL.
    
    Args:
        fingerprint_key: Derived from IP + User-Agent
        payment_session: The payment session ID
        ttl_minutes: Time to live in minutes (default 45)
    """
    conn = get_db_connection()
    try:
        now = datetime.now(timezone.utc)
        expires_at = datetime.fromtimestamp(
            now.timestamp() + (ttl_minutes * 60),
            tz=timezone.utc
        ).isoformat()
        
        conn.execute("""
            INSERT OR REPLACE INTO fingerprint_bindings 
            (fingerprint_key, payment_session, created_at, expires_at)
            VALUES (?, ?, ?, ?)
        """, (fingerprint_key, payment_session, now.isoformat(), expires_at))
        conn.commit()
        logger.info(f"Fingerprint bound: {fingerprint_key[:16]}... -> {payment_session}, expires_at={expires_at}")
    except Exception as e:
        logger.error(f"Error binding fingerprint: {e}", exc_info=True)
        raise
    finally:
        conn.close()

def get_payment_session_from_fingerprint(fingerprint_key: str) -> Optional[str]:
    """
    Get payment session from fingerprint key (if not expired).
    
    Returns None if fingerprint not found or expired.
    """
    conn = get_db_connection()
    try:
        now = datetime.now(timezone.utc).isoformat()
        cursor = conn.execute("""
            SELECT payment_session 
            FROM fingerprint_bindings 
            WHERE fingerprint_key = ? AND expires_at > ?
        """, (fingerprint_key, now))
        row = cursor.fetchone()
        return row['payment_session'] if row else None
    except Exception as e:
        logger.error(f"Error getting payment session from fingerprint: {e}", exc_info=True)
        return None
    finally:
        conn.close()

def cleanup_expired_fingerprints() -> int:
    """
    Clean up expired fingerprint bindings.
    
    Returns the number of deleted rows.
    """
    conn = get_db_connection()
    try:
        now = datetime.now(timezone.utc).isoformat()
        cursor = conn.execute("""
            DELETE FROM fingerprint_bindings 
            WHERE expires_at <= ?
        """, (now,))
        deleted = cursor.rowcount
        conn.commit()
        if deleted > 0:
            logger.info(f"Cleaned up {deleted} expired fingerprint bindings")
        return deleted
    except Exception as e:
        logger.error(f"Error cleaning up expired fingerprints: {e}", exc_info=True)
        return 0
    finally:
        conn.close()


