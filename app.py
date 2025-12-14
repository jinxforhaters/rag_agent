import streamlit as st
from ingestion import DataPipelineConfig, DataIngestionPipeline
from retrieval_agent import RetrievalAgent
from conversation_memory import ConversationMemory
from answer_agent import AnswerAgent
from hf_llm import GroqLLM
from IntentClassifier import IntentClassifier

# ---------------- UI Layout ----------------

st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("📘 RAG Chatbot")
st.write("Upload a PDF and/or enter a URL, then chat with your knowledge base.")

# ---------------- Session State Initialization ----------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "memory" not in st.session_state:
    st.session_state.memory = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "answer_agent" not in st.session_state:
    st.session_state.answer_agent = None

if "intent_classifier" not in st.session_state:
    st.session_state.intent_classifier = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------- Ingestion UI ----------------

st.subheader("Ingest Sources")

pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
url = st.text_input("Website URL")

if st.button("Ingest Source"):
    if not pdf_file and not url:
        st.error("Please upload a PDF OR enter a URL.")
    else:
        pdf_path = None

        if pdf_file:
            with open("uploaded.pdf", "wb") as f:
                f.write(pdf_file.read())
            pdf_path = "uploaded.pdf"

        config = DataPipelineConfig(
            pdf_path=pdf_path,
            url=url if url else None
        )

        with st.spinner("Processing sources..."):
            pipeline = DataIngestionPipeline(config)
            vectorstore, chunks = pipeline.run()

            # Initialize system components
            embedder = pipeline.embedder
            llm = GroqLLM()

            st.session_state.vectorstore = vectorstore
            st.session_state.memory = ConversationMemory(embedder)
            st.session_state.retriever = RetrievalAgent(
                vectorstore, embedder, llm
            )
            st.session_state.answer_agent = AnswerAgent(llm)
            st.session_state.intent_classifier = IntentClassifier(llm)

        st.success("Sources ingested successfully!")


# ---------------- Chat Interface ----------------

st.subheader("Chat with your knowledge base")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask your question...")

if user_input:

    with st.chat_message("user"):
        st.write(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})

    retriever = st.session_state.retriever
    memory = st.session_state.memory
    answer_agent = st.session_state.answer_agent
    classifier = st.session_state.intent_classifier

    if retriever is None or memory is None or answer_agent is None:
        with st.chat_message("assistant"):
            st.error("⚠️ Please ingest sources before chatting.")
        st.stop()

    with st.chat_message("assistant"):
        placeholder = st.empty()

        # ---------------------------------------------
        # 1️⃣ Intent Classification (ONLY social intents)
        # ---------------------------------------------
        intent = classifier.classify(user_input)

        if intent == "GREETING":
            bot_text = "Hello! How can I assist you today? 😊"
            placeholder.write(bot_text)
            st.session_state.messages.append({"role": "assistant", "content": bot_text})
            st.stop()

        if intent == "GOODBYE":
            bot_text = "Goodbye! Feel free to return anytime. 👋"
            placeholder.write(bot_text)
            st.session_state.messages.append({"role": "assistant", "content": bot_text})
            st.stop()

        if intent == "CHITCHAT":
            bot_text = "I can help with questions that relate to the information contained in the uploaded documents or website. Please ask something based on that context."
            placeholder.write(bot_text)
            st.session_state.messages.append({"role": "assistant", "content": bot_text})
            st.stop()

        if intent == "CONTROL_COMMAND":
            memory.clear()
            bot_text = "Conversation memory has been cleared. 🌱"
            placeholder.write(bot_text)
            st.session_state.messages.append({"role": "assistant", "content": bot_text})
            st.stop()

        # --------------------------------------------------
        # 2️⃣ ALL remaining messages go through RetrievalAgent
        # --------------------------------------------------

        rag_result = retriever.run(user_input, memory)

        # Repeated Question
        if rag_result.get("is_repeated"):
            bot_text = rag_result["answer"]
            placeholder.write(bot_text)
            st.session_state.messages.append({"role": "assistant", "content": bot_text})
            memory.add("user", user_input)
            memory.add("assistant", bot_text)
            st.stop()

        # Retrieval Failure
        elif rag_result.get("retrieval_failed"):
            bot_text = rag_result["answer"]
            placeholder.write(bot_text)
            st.session_state.messages.append({"role": "assistant", "content": bot_text})
            memory.add("assistant", bot_text)
            st.stop()

        # User wants to end the session
        elif rag_result.get("is_end"):
            bot_text = rag_result["answer"]
            placeholder.write(bot_text)
            st.session_state.messages.append({"role": "assistant", "content": bot_text})
            st.stop()

        # Goodbye from RetrievalAgent
        elif rag_result.get("is_goodbye"):
            bot_text = rag_result["answer"]
            placeholder.write(bot_text)
            st.session_state.messages.append({"role": "assistant", "content": bot_text})
            st.stop()

        # Greeting from RetrievalAgent
        elif rag_result.get("is_greeting"):
            bot_text = rag_result["answer"]
            placeholder.write(bot_text)
            st.session_state.messages.append({"role": "assistant", "content": bot_text})
            st.stop()

        # Memory-Only Answer
        elif rag_result.get("memory_only"):
            bot_text = "Here’s what you mentioned earlier:\n\n" + "\n".join(rag_result["memory_context"])
            placeholder.write(bot_text)
            st.session_state.messages.append({"role": "assistant", "content": bot_text})
            memory.add("assistant", bot_text)
            st.stop()


        # --------------------------------------------------
        # 3️⃣ Full RAG + Answer Agent (Default case)
        
        # --------------------------------------------------
        else:
            memory.add("user", user_input)

            memory_context = memory.search_relevant_context(
                user_input, retriever.embedder, top_k=3
            )

            messages = answer_agent.build_messages(
                user_input=user_input,
                docs=rag_result["docs"],
                memory_context=memory_context,
                citations=rag_result.get("citations", [])
            )

            bot_text = ""
            for token in answer_agent.llm.stream(messages):
                bot_text += token
                placeholder.write(bot_text)

            auto_end_phrase = "Otherwise, I’ll end the session"
            session_complete = auto_end_phrase.lower() in bot_text.lower()

            sources_list = rag_result.get("citations", ["No sources found"])
            bot_text += "\n\nSources:\n" + "\n".join(f"- {c}" for c in sources_list)

            placeholder.write(bot_text)
            st.session_state.messages.append({"role": "assistant", "content": bot_text})
            memory.add("assistant", bot_text)

            if session_complete:
                st.stop()
