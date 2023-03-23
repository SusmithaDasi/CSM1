import streamlit as st
import requests


API_URL = "http://127.0.0.1:8000/research"

st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🤖",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main {
        background-color: #0f172a;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        color: #f8fafc;
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
    }

    .hero p {
        color: #cbd5e1;
        font-size: 1.1rem;
    }

    .card {
        padding: 1.2rem;
        border-radius: 16px;
        background-color: #111827;
        border: 1px solid #374151;
        margin-bottom: 1rem;
    }

    .card h3 {
        color: #f8fafc;
        margin-bottom: 0.5rem;
    }

    .card p, .card li {
        color: #d1d5db;
    }

    .small-muted {
        color: #94a3b8;
        font-size: 0.9rem;
    }

    .stTextInput > div > div > input {
        border-radius: 12px;
    }

    .stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 3rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="hero">
        <h1>🤖 Multi-Agent Research Assistant</h1>
        <p>
            Planner agent breaks your question into subtasks, worker agents research each part,
            critic agent reviews the output, and the final agent writes a structured answer.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([2, 1])

with left:
    question = st.text_input(
        "Research question",
        placeholder="Example: What are the latest trends in multi-agent AI systems?",
    )

with right:
    st.markdown("<br>", unsafe_allow_html=True)
    run_button = st.button("🚀 Run Research")

st.markdown("---")

if run_button:
    if not question.strip():
        st.warning("Please enter a research question.")
    else:
        with st.spinner("Agents are planning, searching, criticizing, and writing..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"question": question},
                    timeout=300,
                )
                response.raise_for_status()
                data = response.json()

                st.success("Research complete!")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Subtasks", len(data.get("subtasks", [])))

                with col2:
                    st.metric("Revision Count", data.get("revision_count", 0))

                with col3:
                    st.metric("Pipeline", "Complete")

                st.markdown("## 🧭 Planner Output")
                for i, task in enumerate(data.get("subtasks", []), start=1):
                    st.markdown(
                        f"""
                        <div class="card">
                            <h3>Subtask {i}</h3>
                            <p>{task}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("## 🛠️ Worker Research")

                worker_outputs = data.get("worker_outputs", {})

                for i, (task, output) in enumerate(worker_outputs.items(), start=1):
                    with st.expander(f"Worker {i}: {task}", expanded=False):
                        st.markdown("### Answer")
                        st.write(output.get("answer", ""))

                        st.markdown("### Sources")
                        st.text(output.get("sources", ""))

                st.markdown("## 🧪 Critic Review")
                st.markdown(
                    f"""
                    <div class="card">
                        <p>{data.get("critique", "")}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("## ✅ Final Report")
                st.markdown(
                    f"""
                    <div class="card">
                        {data.get("final_answer", "").replace("\n", "<br>")}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            except requests.exceptions.ConnectionError:
                st.error(
                    "FastAPI backend is not running. Start it with: uvicorn app.api:app --reload"
                )

            except requests.exceptions.Timeout:
                st.error("The request timed out. Try a smaller question or faster model.")

            except Exception as e:
                st.error(f"Something went wrong: {e}")


st.markdown("---")
st.caption("Built with LangGraph, Ollama, Tavily, FastAPI, and Streamlit.")