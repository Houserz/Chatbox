from __future__ import annotations

from typing import Any, Dict, List, Optional

from openai import AzureOpenAI
from pinecone import Pinecone

from .agent_types import truncate, safe_text
from .obnoxious_agent import Obnoxious_Agent
from .greeting_agent import Greeting_Agent
from .context_rewriter_agent import Context_Rewriter_Agent
from .query_agent import Query_Agent
from .relevant_documents_agent import Relevant_Documents_Agent
from .answering_agent import Answering_Agent


class Head_Agent:
    def __init__(self, azure_api_key, azure_endpoint, azure_api_version, azure_deployment, azure_embedding_deployment, pinecone_key, pinecone_index_name, namespace: str = "") -> None:
        self.client = AzureOpenAI(
            api_key=azure_api_key,
            azure_endpoint=azure_endpoint,
            api_version=azure_api_version
        )
        self.deployment = azure_deployment
        self.embedding_deployment = azure_embedding_deployment

        pc = Pinecone(api_key=pinecone_key)
        self.index = pc.Index(pinecone_index_name)
        self.namespace = namespace

        self.setup_sub_agents()

        self.NOT_RELEVANT_MESSAGE = (
            "This query is not relevant to the indexed content. "
            "I can only answer questions that are covered by the available documents."
        )

    def setup_sub_agents(self):
        self.obnoxious_agent = Obnoxious_Agent(self.client, self.deployment)
        self.greeting_agent = Greeting_Agent(self.client, self.deployment)
        self.rewriter_agent = Context_Rewriter_Agent(self.client, self.deployment)
        self.query_agent = Query_Agent(self.index, self.client, deployment=self.deployment, embedding_deployment=self.embedding_deployment, namespace=self.namespace)
        self.relevant_docs_agent = Relevant_Documents_Agent(self.client, self.deployment)
        self.answering_agent = Answering_Agent(self.client, self.deployment)

    def main_loop(self):
        history: List[Dict[str, str]] = []
        while True:
            user_q = input("You: ").strip()
            if user_q.lower() in {"exit", "quit"}:
                break
            out = self.handle(user_q, history)
            print("Bot:", out["final_text"] or "[stream]")
            history.append({"role": "user", "content": user_q})
            history.append({"role": "assistant", "content": out["final_text"] or ""})

    def handle(self, user_query: str, history: Optional[List[Dict[str, str]]] = None, k: int = 5) -> Dict[str, Any]:
        history = history or []

        # 1) Obnoxious check
        if self.obnoxious_agent.check_query(user_query) == "Yes":
            return {
                "final_stream": None,
                "final_text": "Please rephrase politely and try again.",
                "is_obnoxious": True,
                "is_relevant": False,
                "is_greeting": False,
                "rewritten_query": user_query,
                "docs": [],
            }

        # 2) Greeting / small talk
        if self.greeting_agent.is_greeting(user_query):
            reply = self.greeting_agent.get_reply(user_query)
            return {
                "final_stream": None,
                "final_text": reply,
                "is_obnoxious": False,
                "is_relevant": False,
                "is_greeting": True,
                "rewritten_query": user_query,
                "docs": [],
            }

        # 3) Rewrite query for multi-turn ambiguity
        rewritten = self.rewriter_agent.rephrase(history, user_query)

        # 4) Decide whether to query Pinecone
        action = self.query_agent.should_query(rewritten)
        if action == "NO_QUERY":
            return {
                "final_stream": None,
                "final_text": self.NOT_RELEVANT_MESSAGE,
                "is_obnoxious": False,
                "is_relevant": False,
                "is_greeting": False,
                "rewritten_query": rewritten,
                "docs": [],
            }

        # 4) Retrieve (use at least 7 chunks to improve relevance-judge coverage)
        docs = self.query_agent.query_vector_store(rewritten, k=max(k, 7))

        # 5) If no docs retrieved, return not relevant
        if not docs:
            return {
                "final_stream": None,
                "final_text": self.NOT_RELEVANT_MESSAGE,
                "is_obnoxious": False,
                "is_relevant": False,
                "is_greeting": False,
                "rewritten_query": rewritten,
                "docs": [],
            }

        # 6) Check if retrieved docs are actually relevant
        relevance = self.relevant_docs_agent.get_relevance(rewritten, docs)
        if relevance == "Not Relevant":
            return {
                "final_stream": None,
                "final_text": self.NOT_RELEVANT_MESSAGE,
                "is_obnoxious": False,
                "is_relevant": False,
                "is_greeting": False,
                "rewritten_query": rewritten,
                "docs": docs,
            }

        # 7) Answer using docs (stream)
        stream = self.answering_agent.generate_stream(rewritten, docs, history, k=k)
        return {
            "final_stream": stream,
            "final_text": None,
            "is_obnoxious": False,
            "is_relevant": True,
            "is_greeting": False,
            "rewritten_query": rewritten,
            "docs": docs,
        }
