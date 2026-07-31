"""
gemma_engine.py

Mzalendo — AI & SMS Intelligence Core (Gemma 4)

This module is the "brain" of Mzalendo. Given a raw SMS message (and sender
number), it:
  1. Extracts structured data from the message using Gemma 4
  2. Semantically clusters it with similar recent reports (via embeddings)
  3. Calculates a confidence score for the cluster
  4. Routes verified/official senders straight to broadcast
  5. Generates a clean, broadcast-ready alert using Gemma 4 once a cluster
     (or an official message) is verified

Main entry point for the backend team: `process_message(sender_number, sms_text)`

Setup:
    pip install google-genai numpy

Environment:
    Set GEMINI_API_KEY as an environment variable, or pass api_key directly
    to init_client(). Get a key at https://aistudio.google.com/apikey
"""

import os
import json
from datetime import datetime, timedelta

import numpy as np
from google import genai

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GEMMA_MODEL = "gemma-4-26b-a4b-it"
EMBEDDING_MODEL = "gemini-embedding-001"

SIMILARITY_THRESHOLD = 0.80     # cosine similarity to consider two reports "the same event"
TIME_WINDOW_MINUTES = 30        # how recent a cluster must be to accept new reports
VERIFIED_THRESHOLD = 70         # confidence score (0-100) required to auto-broadcast

# Map of verified sender phone numbers -> display name/role.
# Backend team: this can be swapped out for a DB-backed lookup later;
# for the hackathon demo, a hardcoded dict is fine.
VERIFIED_SENDERS = {
    "+254700000001": "Rongai Ward Chief",
    "+254700000002": "County Health Officer",
}

# ---------------------------------------------------------------------------
# Client setup
# ---------------------------------------------------------------------------

_client = None


def init_client(api_key: str = None):
    """
    Initialize the Gemini API client. Call this once at startup.
    If api_key is not provided, reads from the GEMINI_API_KEY environment
    variable.
    """
    global _client
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError(
            "No API key provided. Pass api_key to init_client() or set "
            "the GEMINI_API_KEY environment variable."
        )
    _client = genai.Client(api_key=key)
    return _client


def _get_client():
    if _client is None:
        raise RuntimeError(
            "Client not initialized. Call init_client(api_key) before using "
            "any other function in this module."
        )
    return _client


# ---------------------------------------------------------------------------
# In-memory cluster store
#
# NOTE for backend team: this list lives only in this process's memory and
# will reset if the server restarts. For the hackathon demo this is fine.
# If you want clusters to survive restarts, persist `clusters` to your DB
# (e.g. serialize each cluster's reports + centroid_embedding as JSON in a
# `clusters` table) and reload into this list on startup.
# ---------------------------------------------------------------------------

clusters = []


# ---------------------------------------------------------------------------
# Step 1: Extraction (Gemma call)
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """You are a civic report parser for Nairobi. Given a raw SMS message,
extract structured data. Respond ONLY with valid JSON, no markdown, no explanation.

Format:
{"is_civic_relevant": bool, "event_type": string, "location": string, "urgency": "low"|"medium"|"high", "summary": string}

event_type must be one of: power_outage, road_closure, health, safety, water, other

Examples:
Message: "power is out near rongai market since morning"
Output: {"is_civic_relevant": true, "event_type": "power_outage", "location": "Rongai market", "urgency": "medium", "summary": "Power outage reported near Rongai market since morning"}

Message: "hey are we still meeting for lunch"
Output: {"is_civic_relevant": false, "event_type": "", "location": "", "urgency": "", "summary": ""}

Message: "road near karen roundabout closed due to accident, avoid"
Output: {"is_civic_relevant": true, "event_type": "road_closure", "location": "Karen roundabout", "urgency": "high", "summary": "Road closed near Karen roundabout due to accident"}

Now process this message:"""


def extract_report(sms_text: str) -> dict:
    """
    Turn a raw SMS into structured data using Gemma 4.

    Returns a dict:
        {
          "is_civic_relevant": bool,
          "event_type": str,
          "location": str,
          "urgency": "low" | "medium" | "high",
          "summary": str
        }
    """
    client = _get_client()
    response = client.models.generate_content(
        model=GEMMA_MODEL,
        config={"system_instruction": EXTRACTION_PROMPT},
        contents=sms_text,
    )

    text = response.text.strip()
    # Defensive cleanup in case Gemma wraps the JSON in markdown fences
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    return json.loads(text.strip())


# ---------------------------------------------------------------------------
# Step 2: Embeddings + semantic clustering
# ---------------------------------------------------------------------------

def get_embedding(text: str):
    """Return the embedding vector for a piece of text."""
    client = _get_client()
    result = client.models.embed_content(model=EMBEDDING_MODEL, contents=text)
    return result.embeddings[0].values


def cosine_similarity(a, b) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def add_report_to_clusters(sms_text: str) -> dict:
    """
    Extract structured data from an SMS and either:
      - ignore it (not civic relevant)
      - add it to an existing matching cluster
      - start a new cluster

    Returns one of:
        {"status": "ignored", "reason": "not civic relevant"}
        {"status": "added_to_cluster", "cluster": <cluster dict>}
        {"status": "new_cluster", "cluster": <cluster dict>}
    """
    report = extract_report(sms_text)

    if not report["is_civic_relevant"]:
        return {"status": "ignored", "reason": "not civic relevant"}

    embedding = get_embedding(sms_text)
    now = datetime.now()

    for cluster in clusters:
        if cluster["event_type"] != report["event_type"]:
            continue
        if now - cluster["last_updated"] > timedelta(minutes=TIME_WINDOW_MINUTES):
            continue

        similarity = cosine_similarity(embedding, cluster["centroid_embedding"])
        if similarity >= SIMILARITY_THRESHOLD:
            cluster["reports"].append(
                {"text": sms_text, "report": report, "similarity": similarity}
            )
            cluster["last_updated"] = now
            return {"status": "added_to_cluster", "cluster": cluster}

    new_cluster = {
        "event_type": report["event_type"],
        "location": report["location"],
        "reports": [{"text": sms_text, "report": report, "similarity": 1.0}],
        "centroid_embedding": embedding,
        "last_updated": now,
    }
    clusters.append(new_cluster)
    return {"status": "new_cluster", "cluster": new_cluster}


# ---------------------------------------------------------------------------
# Step 3: Confidence scoring (pure math, no AI call)
# ---------------------------------------------------------------------------

def confidence_score(cluster: dict) -> int:
    """
    Combine report count and average similarity into a 0-100 confidence score.
    Caps out once a cluster has 5+ corroborating reports.
    """
    report_count = len(cluster["reports"])
    avg_similarity = sum(r["similarity"] for r in cluster["reports"]) / report_count
    count_score = min(report_count / 5, 1.0) * 60
    similarity_score = avg_similarity * 40
    return round(count_score + similarity_score)


# ---------------------------------------------------------------------------
# Step 4: Verified sender handling
# ---------------------------------------------------------------------------

def handle_incoming(sender_number: str, sms_text: str) -> dict:
    """
    Route a message based on sender. Verified/official senders skip
    clustering entirely and go straight to alert generation.

    Returns one of:
        {"path": "official", "source": <name>, "text": sms_text}
        {"path": "citizen", "result": <add_report_to_clusters() output>}
    """
    if sender_number in VERIFIED_SENDERS:
        return {
            "path": "official",
            "source": VERIFIED_SENDERS[sender_number],
            "text": sms_text,
        }
    result = add_report_to_clusters(sms_text)
    return {"path": "citizen", "result": result}


# ---------------------------------------------------------------------------
# Step 5: Alert generation (Gemma call)
# ---------------------------------------------------------------------------

ALERT_PROMPT = """You write short, clear public alerts for SMS broadcast to a community.
Given either a set of citizen reports about an event, or an official statement, write ONE
concise alert under 160 characters. State what happened/what's announced and where.
No hashtags, no emojis, no extra commentary — just the alert text."""


def generate_alert(source_text: str) -> str:
    """Compose a clean, SMS-ready broadcast alert from report(s) or an official statement."""
    client = _get_client()
    response = client.models.generate_content(
        model=GEMMA_MODEL,
        config={"system_instruction": ALERT_PROMPT},
        contents=source_text,
    )
    return response.text.strip()


# ---------------------------------------------------------------------------
# Main entry point for the backend team
# ---------------------------------------------------------------------------

def process_message(sender_number: str, sms_text: str) -> dict:
    """
    THE MAIN FUNCTION. Call this once per incoming SMS.

    Returns a dict shaped for direct storage in the `events` table and
    direct consumption by the dashboard:

        {
          "broadcast": bool,
          "event_type": str | None,
          "location": str | None,
          "urgency": str | None,
          "summary": str | None,
          "confidence": int | None,      # 0-100, None for official messages
          "report_count": int | None,
          "alert_text": str | None,      # only present if broadcast is True
          "source": "citizen_cluster" | "official" | None,
          "reason": str | None           # only present if broadcast is False
        }
    """
    handled = handle_incoming(sender_number, sms_text)

    # --- Official / verified sender path ---
    if handled["path"] == "official":
        alert = generate_alert(handled["text"])
        return {
            "broadcast": True,
            "event_type": None,
            "location": None,
            "urgency": None,
            "summary": handled["text"],
            "confidence": None,
            "report_count": None,
            "alert_text": alert,
            "source": "official",
            "reason": None,
        }

    # --- Citizen report path ---
    result = handled["result"]

    if result["status"] == "ignored":
        return {
            "broadcast": False,
            "event_type": None,
            "location": None,
            "urgency": None,
            "summary": None,
            "confidence": None,
            "report_count": None,
            "alert_text": None,
            "source": None,
            "reason": "not civic relevant",
        }

    cluster = result["cluster"]
    score = confidence_score(cluster)
    latest_report = cluster["reports"][-1]["report"]

    base_output = {
        "event_type": cluster["event_type"],
        "location": cluster["location"],
        "urgency": latest_report["urgency"],
        "confidence": score,
        "report_count": len(cluster["reports"]),
        "source": "citizen_cluster",
    }

    if score >= VERIFIED_THRESHOLD:
        combined = " | ".join(r["report"]["summary"] for r in cluster["reports"])
        alert = generate_alert(combined)
        return {
            **base_output,
            "broadcast": True,
            "summary": combined,
            "alert_text": alert,
            "reason": None,
        }

    return {
        **base_output,
        "broadcast": False,
        "summary": latest_report["summary"],
        "alert_text": None,
        "reason": "not yet verified",
    }


# ---------------------------------------------------------------------------
# Quick manual test (only runs if this file is executed directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    init_client()  # reads GEMINI_API_KEY from environment

    print(process_message("+254712345678", "power is out near rongai market since morning"))
    print(process_message("+254712345679", "no lights in rongai since this morning, anyone know why"))
    print(process_message("+254712345680", "power outage still ongoing near rongai, its been hours"))
    print(process_message("+254700000001", "Fertilizer delivery arriving at Rongai depot tomorrow 9am"))
