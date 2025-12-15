import time
from typing import Dict, List
from langchain_core.documents import Document
import re


class RetrievalAgent:
    """
    Retrieval Agent:
    - Reformulates queries using LLM
    - Uses conversation memory
    - Searches vector DB
    - Filters results by score threshold
    - Retries once with better query
    - Logs query, time, count
    """

    def __init__(self, vectorstore, embedder, llm,
                 score_threshold=0.30, max_k=20):
        self.vectorstore = vectorstore
        self.embedder = embedder
        self.llm = llm
        self.score_threshold = score_threshold
        self.max_k = max_k
        self.single_word_greetings = {
            "hi", "hello", "hey", "yo", "sup",
            "hii", "hiii"
        }

        self.multi_word_greetings = {
            "good morning", "good evening", "good afternoon"
        }

    def _is_goodbye(self, text: str) -> bool:
        text = text.lower().strip()
        endings = [
            "bye", "goodbye", "tata", "see you", "thanks", 
            "thank you", "that's it", "done", "stop", "quit"
        ]
        return any(e in text for e in endings)
    

    def _memory_retrieve(self, user_input, memory, threshold=0.85):
        """
        Returns a list of memory messages relevant to the user input.
        If similarity is high enough, memory alone can answer.
        """
        relevant = []

        q_emb = self.embedder.embed_query(user_input)

        for msg in memory.messages:
            emb = self.embedder.embed_query(msg["content"])
            score = memory._cosine_similarity(q_emb, emb)

            if score >= threshold:
                relevant.append(msg["content"])

        return relevant
    
    def _wants_to_end(self, text: str) -> bool:
        text = text.lower().strip()
        endings = [
            "no", "nothing", "nope", "that's all", "that is all",
            "i'm good", "im good", "all good", 
            "stop", "end", "finish", "done"
        ]
        return any(text == e or e in text for e in endings)

    def _extract_citations(self, docs):
            """
            Returns clean grouped citations:
            - PDF pages grouped numerically
            - Website domains grouped
            """

            from urllib.parse import urlparse

            pdf_pages = set()
            domains = set()

            for d in docs:
                meta = d.metadata
                src_type = meta.get("source_type")
                src = meta.get("source")

                if src_type == "pdf":
                    page = meta.get("page")
                    if page is not None:
                        pdf_pages.add(page + 1)  

                elif src_type == "web":
                    if src:
                        domain = urlparse(src).netloc
                        domains.add(domain)

            output = []

            if pdf_pages:
                sorted_pages = sorted(pdf_pages)
                page_list = ", ".join(str(p) for p in sorted_pages)
                output.append(f"PDF pages: {page_list}")

            if domains:
                for d in domains:
                    output.append(f"Website: {d}")

            return output



    def _is_greeting(self, text: str) -> bool:
        """
        Detect greeting anywhere in text without false positives.
        """

        text = text.lower().strip()

        clean = re.sub(r"[^\w\s]", " ", text)

        words = clean.split()

        if any(w in self.single_word_greetings for w in words):
            return True

        for phrase in self.multi_word_greetings:
            if re.search(r"\b" + re.escape(phrase) + r"\b", clean):
                return True

        return False   

    def _generate_query(self, user_input: str, memory_context: str) -> str:
        system_prompt = (
            "You are a retrieval planner. "
            "Rewrite the user query as a highly effective search query. "
            "Do NOT answer the question. Only output the improved search query."
        )

        human_prompt = (
            f"User query: {user_input}\n"
            f"Conversation context: {memory_context}\n\n"
            "Return only the improved search query:"
        )

        resp = self.llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_prompt},
        ])
        print(resp)
        return resp.strip()


    def _refine_query(self, previous_query: str, user_input: str, memory_context: str):
        system_prompt = (
            "Previous retrieval returned no relevant chunks. "
            "Rewrite a BETTER search query with more specific keywords."
        )

        human_prompt = (
            f"Previous query: {previous_query}\n"
            f"User input: {user_input}\n"
            f"Conversation context: {memory_context}\n\n"
            "Return improved search query only:"
        )

        resp = self.llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_prompt},
        ])
        return resp.strip()
    
    def run(self, user_input: str, memory) -> Dict:
        """
        Main retrieval logic with:
        - memory context
        - query generation
        - vector search
        - retry logic
        - filtering
        """
        if memory.is_repeated_question(user_input, self.embedder):
            previous_answer = memory.get_previous_answer(user_input, self.embedder)

            return {
                "is_repeated": True,
                "answer": (
                    "You already asked this earlier — here is the same answer again.\n\n"
                    + (previous_answer if previous_answer else "But I could not retrieve the earlier answer.")
                ),
                "docs": [],
                "citations": [],
                "query_used": None,
                "num_chunks": 0,
                "time_taken": 0
            }



        if self._is_greeting(user_input):
            return {
                "is_greeting": True,
                "answer": "Hello! How can I assist you today?",
                "query_used": None,
                "docs": [],
                "time_taken": 0,
                "num_chunks": 0,
            }

        if self._is_goodbye(user_input):
            return {
                "is_greeting": False,
                "is_goodbye": True,
                "answer": "It was great assisting you! Ending the session now.",
                "docs": [],
                "citations": [],
                "num_chunks": 0,
                "time_taken": 0,
                "query_used": None
            }
        
        if self._wants_to_end(user_input):
            return {
                "is_end": True,
                "answer": "Alright! Ending the session. Feel free to return anytime. 😊",
                "docs": [],
                "citations": [],
                "num_chunks": 0,
                "time_taken": 0
            }
                
        memory_hits = self._memory_retrieve(user_input, memory)

        if memory_hits:
            return {
                "is_greeting": False,
                "is_repeated": False,
                "is_goodbye": False,
                "memory_only": True,
                "docs": [],  # no vector chunks
                "memory_context": memory_hits,
                "citations": ["Conversation Memory"],
                "query_used": None,
                "num_chunks": len(memory_hits),
                "time_taken": 0
            }


        memory_context = memory.search_relevant_context(
            user_input, self.embedder, top_k=3
        )

        # 1. First query attempt
        query1 = self._generate_query(user_input, memory_context)
        print(f"🔍 RetrievalAgent → First query: {query1}")

        start = time.perf_counter()
        docs_with_scores = self.vectorstore.similarity_search_with_score(
            query1, k=self.max_k
        )
        elapsed = time.perf_counter() - start

        good_docs = [d for d, score in docs_with_scores if score >= self.score_threshold]

        print(
            f"🔹 Attempt 1: Returned {len(docs_with_scores)} chunks, "
            f"{len(good_docs)} above threshold, time={elapsed:.3f}s"
        )

        if good_docs:
            return {
                "is_greeting": False,
                "query_used": query1,
                "docs": good_docs,
                "citations": self._extract_citations(good_docs),
                "time_taken": elapsed,
                "num_chunks": len(good_docs),
            }

        # 2. Retry once
        query2 = self._refine_query(query1, user_input, memory_context)
        print(f"🔄 RetrievalAgent → Retry query: {query2}")

        start2 = time.perf_counter()
        docs_with_scores_2 = self.vectorstore.similarity_search_with_score(
            query2, k=self.max_k
        )
        elapsed2 = time.perf_counter() - start2

        good_docs_2 = [
            d for d, score in docs_with_scores_2 if score >= self.score_threshold
        ]

        print(
            f"🔹 Attempt 2: Returned {len(docs_with_scores_2)} chunks, "
            f"{len(good_docs_2)} above threshold, time={elapsed2:.3f}s"
        )

        if good_docs_2:
            return {
                "is_greeting": False,
                "query_used": query2,
                "docs": good_docs_2,
                "citations": self._extract_citations(good_docs_2),
                "time_taken": elapsed + elapsed2,
                "num_chunks": len(good_docs_2),
            }

        # 3. Failure → No relevant chunks
        print("⚠️ RetrievalAgent → No relevant chunks after two attempts.")
        return {
            "is_greeting": False,
            "is_goodbye": False,
            "answer": "I don’t have enough information in the knowledge base to answer this.",
            "docs": [],
            "citations": [],
            "time_taken": elapsed + elapsed2,
            "num_chunks": 0,
            "query_used": query2,
            "retrieval_failed": True
        }
