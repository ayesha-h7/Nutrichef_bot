import streamlit as st
import requests

# ---------------------------------------
# API KEY
# ---------------------------------------
GROQ_API_KEY = ""

def chat_llama(messages):
    res = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={"model": "llama-3.3-70b-versatile", "messages": messages}
    )
    data = res.json()
    if "error" in data:
        return f"❗ API Error: {data['error'].get('message','Unknown error')}"
    if "choices" not in data:
        return "❗ No response."
    return data["choices"][0]["message"]["content"]


st.set_page_config(page_title="NutriBot", layout="wide")

# ---------------------------------------
# FIXED FLOATING BUTTON USING NATIVE STREAMLIT
# ---------------------------------------
if "bot_open" not in st.session_state:
    st.session_state.bot_open = False

# ---- Spacer to push button down (works reliably)
st.markdown("""
    <div style="height:78vh;"></div>
""", unsafe_allow_html=True)

# ---- Align button to bottom-right
col1, col2, col3 = st.columns([8,1,1])

with col3:
    btn_icon = "✖️" if st.session_state.bot_open else "💬"
    if st.button(btn_icon, key="nutribot_btn", help="Open NutriBot"):
        st.session_state.bot_open = not st.session_state.bot_open
        st.rerun()

# ---------------------------------------
# SHOW CHATBOX WHEN OPEN
# ---------------------------------------
if st.session_state.bot_open:

    with st.container():
        st.markdown("""
            <div style="
                border: 3px solid #4CAF50;
                border-radius: 15px;
                padding: 20px;
                background: white;
                margin-top: -20px;
            ">
        """, unsafe_allow_html=True)

        st.markdown(
            '<h2 style="text-align:center;color:#2c7d3d;">🥗 NutriBot — Your Nutrition Assistant</h2>',
            unsafe_allow_html=True
        )

        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I’m NutriBot 💚. How can I help you today?"}
            ]

        # Display chat
        for msg in st.session_state.messages:
            if msg["role"] == "assistant":
                st.markdown(
                    f"<div style='background:#f9f9f9;padding:12px;border-radius:10px;margin:6px 0;'>{msg['content']}</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div style='background:#e8ffe8;padding:12px;border-radius:10px;margin:6px 0;text-align:right;'>{msg['content']}</div>",
                    unsafe_allow_html=True
                )

        # Chat input
        user_input = st.chat_input("Type your message...")

        st.markdown("</div>", unsafe_allow_html=True)

    # Handle user message
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        reply = chat_llama(st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()
