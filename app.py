"""

Jirani Alerts

"""

import streamlit as st
import pandas as pd

import database as db
from gemma_engine import init_client, process_message, VERIFIED_SENDERS

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Mzalendo", page_icon="📡", layout="wide")

db.init_db()

if "client_ready" not in st.session_state:
    st.session_state.client_ready = False

# ---------------------------------------------------------------------------
# Sidebar: API key + simulated SMS input
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📡 Jirani Alerts")
    st.caption("AI powered, SMS based community Alerts")

    st.divider()
    st.subheader("Setup")

    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        help="Get one at aistudio.google.com/apikey. Kept in-memory only.",
    )
    if api_key and not st.session_state.client_ready:
        try:
            init_client(api_key=api_key)
            st.session_state.client_ready = True
            st.success("Connected to Gemma 4")
        except Exception as e:
            st.error(f"Failed to connect: {e}")

    st.divider()
    st.subheader("Simulate an incoming SMS")

    sender_options = ["(citizen - random number)"] + [
        f"{name} ({number})" for number, name in VERIFIED_SENDERS.items()
    ]
    sender_choice = st.selectbox("Sender", sender_options)

    if sender_choice == "(citizen - random number)":
        sender_number = st.text_input("Phone number", value="+254712345678")
    else:
        # extract the number from "Name (+254...)"
        sender_number = sender_choice.split("(")[-1].rstrip(")")

    sms_text = st.text_area(
        "Message text",
        placeholder="e.g. Polio vaccination happening in Muguga",
        height=100,
    )

    submitted = st.button("Send SMS", type="primary", use_container_width=True)

    if submitted:
        if not st.session_state.client_ready:
            st.error("Add your Gemini API key above first.")
        elif not sms_text.strip():
            st.warning("Type a message first.")
        else:
            with st.spinner("Gemma is processing the message..."):
                db.insert_message(sender_number, sms_text)
                try:
                    result = process_message(sender_number, sms_text)
                    db.insert_event(result)
                    if result.get("broadcast"):
                        st.success(f"Broadcast sent: {result['alert_text']}")
                    elif result.get("reason") == "not civic relevant":
                        st.info("Message ignored (not civic relevant).")
                    else:
                        st.info(
                            f"Report logged. Confidence: {result.get('confidence')}/100 "
                            f"(needs 70+ to broadcast)"
                        )
                except Exception as e:
                    st.error(f"Error processing message: {e}")

# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------

st.title("Live Dashboard")
st.caption("📵 No Internet. No Problem. Stay Connected")

stats = db.get_stats()

stats = db.get_stats()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Messages received", stats["total_messages"])
col2.metric("Events tracked", stats["total_events"])
col3.metric("Alerts broadcast", stats["total_broadcasts"])
col4.metric("Avg. confidence", f"{stats['avg_confidence']}%")

st.divider()

tab_feed, tab_broadcasts, tab_messages, tab_sources = st.tabs(
    ["🔴 Live Event Feed", "📢 Broadcast History", "💬 Incoming Messages", "✅ Verified Sources"]
)

# --- Live event feed ---
with tab_feed:
    events = db.get_all_events()
    if not events:
        st.caption("No events yet — send a simulated SMS from the sidebar to get started.")
    for e in events:
        urgency_color = {"high": "#e03131", "medium": "#f0a500", "low": "#2f9e44"}.get(
            (e["urgency"] or "").lower(), "#868e96"
        )
        st.markdown(
            f'<div style="border-left: 6px solid {urgency_color}; padding-left: 12px; margin-bottom: 8px;">',
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            header_col, badge_col = st.columns([4, 1])
            with header_col:
                if e["source"] == "official":
                    st.markdown(f"**📋 Official message** — {e['summary']}")
                else:
                    st.markdown(
                        f"**{(e['event_type'] or 'unknown').replace('_', ' ').title()}** "
                        f"in {e['location'] or 'unknown location'}"
                    )
                    st.caption(e["summary"] or "")
            with badge_col:
                if e["is_broadcast"]:
                    st.success("BROADCAST")
                else:
                    st.warning("PENDING")

            if e["confidence"] is not None:
                st.progress(
                    min(e["confidence"], 100) / 100,
                    text=f"Confidence: {e['confidence']}/100 "
                    f"({e['report_count']} report{'s' if e['report_count'] != 1 else ''})",
                )

            if e["alert_text"]:
                st.info(f"📤 {e['alert_text']}")

           st.caption(f"{e['created_at']}")
        st.markdown("</div>", unsafe_allow_html=True)


# --- Broadcast history ---
with tab_broadcasts:
    broadcasts = db.get_broadcast_history()
    if not broadcasts:
        st.caption("No alerts have been broadcast yet.")
    for b in broadcasts:
        with st.container(border=True):
            st.markdown(f"**{b['alert_text']}**")
            source_label = "Official" if b["source"] == "official" else "Citizen reports (verified)"
            st.caption(f"{source_label} • {b['created_at']}")

# --- Incoming messages log ---
with tab_messages:
    messages = db.get_recent_messages()
    if not messages:
        st.caption("No messages received yet.")
    else:
        df = pd.DataFrame(messages)[["sender_number", "raw_text", "received_at"]]
        df.columns = ["Sender", "Message", "Received at"]
        st.dataframe(df, use_container_width=True, hide_index=True)

# --- Verified sources ---
with tab_sources:
    st.caption("These senders skip clustering/verification and broadcast immediately.")
    for number, name in VERIFIED_SENDERS.items():
        st.markdown(f"- **{name}** — `{number}`")
