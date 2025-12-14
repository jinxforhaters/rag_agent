from typing import List, Dict
from langchain_core.documents import Document
import numpy as np


class ConversationMemory:
    """
    Lightweight semantic memory store.

    - Stores user & assistant messages
    - Retrieves relevant memory context via embeddings
    - Prevents context explosion
    - Simple + interview-friendly
    """

    def __init__(self, embedder, max_messages=50):
        self.embedder = embedder
        self.max_messages = max_messages
        self.messages: List[Dict] = []   

    def get_previous_answer(self, query: str, embedder, threshold: float = 0.90):
        """
        Returns the assistant answer for a previous similar user question.
        """
        if not self.messages:
            return None

        q_emb = embedder.embed_query(query)

        last_question = None
        last_answer = None

        for msg in self.messages:
            if msg["role"] == "user":
                prev_emb = embedder.embed_query(msg["content"])
                score = self._cosine_similarity(q_emb, prev_emb)
                if score >= threshold:
                    last_question = msg["content"]
            elif msg["role"] == "assistant" and last_question:
                last_answer = msg["content"]
                last_question = None  

        return last_answer    

    def is_repeated_question(self, query: str, embedder, threshold: float = 0.90):
        """
        Returns True if the query is semantically too similar
        to the user's previous questions.
        """
        if not self.messages:
            return False

        q_emb = embedder.embed_query(query)

        for msg in self.messages:
            if msg["role"] == "user":
                prev_emb = embedder.embed_query(msg["content"])
                score = self._cosine_similarity(q_emb, prev_emb)
                print(score)
                if score >= threshold:
                    return True
                
        return False    

    # --------------------
    # Storing messages
    # --------------------
    def add(self, role: str, content: str):
        """Store a message in memory."""
        if not content or content.strip() == "":
            return

        if content.lower().strip() in ["hi", "hello", "hey"]:
            return

        self.messages.append({"role": role, "content": content})

        if len(self.messages) > self.max_messages:
            self.messages.pop(0)

    # --------------------
    # Semantic search across memory
    # --------------------
    def search_relevant_context(self, query: str, embedder, top_k=3) -> str:
        """
        Find the most relevant past messages for retrieval planning.
        """

        if not self.messages:
            return ""

        q_emb = embedder.embed_query(query)

        all_scores = []
        for msg in self.messages:
            emb = embedder.embed_query(msg["content"])
            score = self._cosine_similarity(q_emb, emb)
            all_scores.append((score, msg["content"]))

        all_scores.sort(key=lambda x: x[0], reverse=True)

        top = [c for _, c in all_scores[:top_k]]

        return "\n".join(top)

    # --------------------
    # Utility: Cosine similarity
    # --------------------
    def _cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # --------------------
    # Display memory (debug)
    # --------------------
    def get_messages(self):
        return self.messages

