from hf_llm import GroqLLM

class IntentClassifier:
    """
    Classifies user messages into high-level chat/social intents.
    Does NOT classify RAG-specific states (repeated, memory-only, retrieval-failed).
    Those are handled by RetrievalAgent.
    """

    def __init__(self, llm: GroqLLM):
        self.llm = llm

    def classify(self, text: str) -> str:
        system_prompt = (
            "Classify the user's message into EXACTLY ONE of these intents:\n"
            "- GREETING: hi, hello, hey, good morning, etc.\n"
            "- GOODBYE: bye, goodbye, see you, thanks\n"
            "- CHITCHAT: personal small talk (how are you, who are you, jokes)\n"
            "- CONTROL_COMMAND: reset, clear memory, restart\n"
            "- RAG_QUERY: ANY factual, informational, or knowledge-based question that may require retrieval\n\n"
            "IMPORTANT RULES:\n"
            "1. If the user asks a question of ANY kind (general, factual, conceptual, academic, or domain-related), classify it as RAG_QUERY.\n"
            "2. Questions such as 'what is the best university', 'who invented X', 'how does Y work', etc. MUST be classified as RAG_QUERY.\n"
            "3. CHITCHAT is ONLY for casual small-talk: 'how are you', 'tell me a joke', 'who are you', 'what's your name'.\n"
            "4. If the message is unclear BUT looks like a question, classify as RAG_QUERY, NOT CHITCHAT.\n"
            "5. Do NOT guess CHITCHAT unless the message is clearly social or personal.\n\n"
            "Return only the intent name in UPPERCASE."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]

        response = self.llm.invoke(messages)
        return response.strip().upper()
