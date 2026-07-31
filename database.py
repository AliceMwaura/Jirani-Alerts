"""
database.py

Mzalendo — SQLite database layer.

Handles storage for raw incoming messages and processed events (clusters /
official broadcasts). Designed to be dead simple for the hackathon demo:
one file, no migrations, safe to delete and recreate at any time.
"""

import sqlite3
from datetime import datetime

DB_PATH = "mzalendo.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender_number TEXT,
            raw_text TEXT,
            received_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT,
            location TEXT,
            urgency TEXT,
            summary TEXT,
            confidence INTEGER,
            report_count INTEGER,
            is_broadcast BOOLEAN DEFAULT 0,
            alert_text TEXT,
            source TEXT,
            reason TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def insert_message(sender_number: str, raw_text: str):
    conn = get_connection()
    conn.execute(
        "INSERT INTO messages (sender_number, raw_text) VALUES (?, ?)",
        (sender_number, raw_text),
    )
    conn.commit()
    conn.close()


def insert_event(result: dict):
    """
    Insert a `process_message()` output dict as a new event row.
    Returns the new event's id.
    """
    conn = get_connection()
    cursor = conn.execute(
        """
        INSERT INTO events (
            event_type, location, urgency, summary, confidence,
            report_count, is_broadcast, alert_text, source, reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result.get("event_type"),
            result.get("location"),
            result.get("urgency"),
            result.get("summary"),
            result.get("confidence"),
            result.get("report_count"),
            result.get("broadcast", False),
            result.get("alert_text"),
            result.get("source"),
            result.get("reason"),
        ),
    )
    conn.commit()
    event_id = cursor.lastrowid
    conn.close()
    return event_id


def get_recent_messages(limit: int = 50):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM messages ORDER BY received_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_events(limit: int = 100):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM events ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_broadcast_history(limit: int = 50):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM events WHERE is_broadcast = 1 ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stats():
    conn = get_connection()
    total_messages = conn.execute("SELECT COUNT(*) as c FROM messages").fetchone()["c"]
    total_events = conn.execute("SELECT COUNT(*) as c FROM events").fetchone()["c"]
    total_broadcasts = conn.execute(
        "SELECT COUNT(*) as c FROM events WHERE is_broadcast = 1"
    ).fetchone()["c"]
    avg_confidence_row = conn.execute(
        "SELECT AVG(confidence) as avg_c FROM events WHERE confidence IS NOT NULL"
    ).fetchone()
    avg_confidence = round(avg_confidence_row["avg_c"], 1) if avg_confidence_row["avg_c"] else 0
    conn.close()
    return {
        "total_messages": total_messages,
        "total_events": total_events,
        "total_broadcasts": total_broadcasts,
        "avg_confidence": avg_confidence,
    }
