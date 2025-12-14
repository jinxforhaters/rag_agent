class AnswerAgent:
    """
    Uses DeepSeek R1 to generate grounded answers using retrieved docs.
    """

    def __init__(self, llm):
        self.llm = llm

    def build_messages(self, user_input, docs, memory_context,citations=None):
        citations = citations or []
        system_prompt = (
                "You are a helpful assistant. Answer ONLY using the retrieved context.\n\n"
                "Before answering, follow these internal reasoning steps:\n"
                "1. Read all retrieved chunks carefully.\n"
                "2. Detect contradictions:\n"
                "   - different numerical values\n"
                "   - opposite claims\n"
                "3. If contradictions exist:\n"
                "   - DO NOT choose a side\n"
                "   - explain both perspectives neutrally\n"
                "   - mention the limitation\n"
                "4. If no contradictions exist:\n"
                "   - Answer concisely and accurately.\n\n"
                "Formatting rules:\n"
                "- Start with a 1–2 sentence summary.\n"
                "- Then provide clean bullet points.\n"
                "- If context does not contain the answer, say so.\n"
                "- Do NOT guess.\n"
                "- Add the following line ONLY if the answer is complete:\n"
                "  'If you need anything else, feel free to ask. Otherwise, I’ll end the session.'\n"
            )

        citation_block = "\n".join(f"- {c}" for c in citations) if citations else "- No sources found"


        context_text = "\n\n---\n".join([d.page_content for d in docs])
        print(context_text)
        if memory_context:
            context_text = "Memory context:\n" + memory_context + "\n\n" + context_text

        
        combined_context = (
        f"Retrieved Context:\n{context_text}\n\n"
        )
        

        return [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": combined_context},
            {"role": "user", "content": user_input},
        ]

    

    def generate(self, user_input, retrieved_docs, memory_context):
        
        text = user_input.lower().strip()
        
        greetings = ["hi", "hello", "hey", "yo", "good morning", "good evening", "good afternoon"]

        if any(text == g for g in greetings):
            return "Hello! 👋 How can I help you today?"
        context_text = "\n\n".join(
                doc.page_content if hasattr(doc, "page_content") else str(doc)
                for doc in retrieved_docs
            )

        system_prompt = (
            "You are a helpful assistant that answers ONLY using the provided context. "
            "If the answer isn't present in the context, say: "
            "\"I don't have enough information from the sources.\""
        )

        user_prompt = (
            f"User question: {user_input}\n\n"
            f"Relevant memory: {memory_context}\n\n"
            f"Context from sources:\n{context_text}\n\n"
            "Give a clear, concise answer."
        )

        response = self.llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])

        return response.strip()
