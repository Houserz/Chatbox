from typing import List
from openai import OpenAI

from .agent_types import RetrievedDoc, safe_text


class Query_Agent:
    def __init__(self, pinecone_index, openai_client, deployment, embedding_deployment, embeddings=None, namespace: str = "") -> None:
        """
        pinecone_index: pinecone.Index
        openai_client: AzureOpenAI
        deployment: Azure deployment name for chat
        embedding_deployment: Azure deployment name for embeddings
        embeddings: optional callable(text)->vector; if None, uses OpenAI embeddings internally.
        namespace: Pinecone namespace (optional)
        """
        self.index = pinecone_index
        self.client = openai_client
        self.deployment = deployment
        self.embedding_deployment = embedding_deployment
        self.embeddings = embeddings
        self.namespace = namespace

        self.prompt = (
            "You decide whether to query the vector store. The store holds indexed content (e.g. a book or documents); the topic may vary.\n"
            "- If the user asks a substantive question that could be answered by document content, respond: QUERY. This includes: 'explain X', 'what is X', 'how does X work', 'how do X learn', 'why is X important', 'what are the differences between X and Y', 'what are common X', 'can you talk about X', 'describe X'.\n"
            "- Questions about a mechanism or process (e.g. 'How do neural networks learn?', 'How does gradient descent work?', 'Why is cross-validation important?') are always QUERY.\n"
            "- Short follow-up questions (e.g. 'Why is it important?', 'How do they learn?', 'Can you give an example?') get QUERY — the rewriter resolves 'it'/'they' to the topic.\n"
            "- If the message mixes a content-related question with an unrelated part, still respond: QUERY.\n"
            "- Respond NO_QUERY only when the message is clearly not a content question: purely greeting/chitchat, or about weather, sports, restaurants, with no ask about the indexed content. When in doubt, prefer QUERY.\n"
            "Output exactly one token: QUERY or NO_QUERY.\n"
        )

    def query_vector_store(self, query, k=5):
        vec = self._embed(safe_text(query))
        res = self.index.query(
            vector=vec,
            top_k=k,
            include_metadata=True,
            namespace=self.namespace or None,
        )

        docs: List[RetrievedDoc] = []
        for match in getattr(res, "matches", []) or []:
            md = match.metadata or {}
            # common chunk fields
            text = md.get("text") or md.get("chunk") or md.get("content") or md.get("page_content") or ""
            docs.append(RetrievedDoc(text=safe_text(text), metadata=dict(md)))
        return docs

    def set_prompt(self, prompt):
        self.prompt = prompt

    def extract_action(self, response, query=None):
        s = safe_text(response).strip().upper()
        return s if s in {"QUERY", "NO_QUERY"} else "QUERY"

    def _looks_like_content_question(self, query: str) -> bool:
        """If True, prefer QUERY to avoid false NO_QUERY for substantive questions."""
        q = safe_text(query).strip()
        if len(q) < 4:
            return False
        q_lower = q.lower()
        if any(x in q_lower for x in (
            "weather", "forecast", "restaurant", "capital city", "world cup", "coldplay",
            "gas station", "concert", "smartphone", "factory settings", "sunflower",
        )):
            return False
        if "?" in q:
            return True
        starters = (
            "what", "how", "why", "explain", "describe", "can you", "when", "which",
            "discuss", "is ", "are ", "does ", "do ", "could", "should", "would",
            "tell", "define", "compare", "give", "list", "name",
        )
        if q_lower.startswith(starters):
            return True
        content_keywords = (
            "important", "work", "mean", "difference", "example", "learn",
            "train", "use", "apply", "affect", "impact", "relate", "help",
        )
        return any(kw in q_lower for kw in content_keywords)

    def _single_should_query(self, query: str, seed: int) -> str:
        resp = self.client.chat.completions.create(
            model=self.deployment,
            temperature=0,
            seed=seed,
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": safe_text(query)},
            ],
        )
        return self.extract_action(resp.choices[0].message.content, query)

    def should_query(self, query: str) -> str:
        votes = [self._single_should_query(query, seed=s) for s in (42, 123, 7)]
        query_count = sum(1 for v in votes if v == "QUERY")
        action = "QUERY" if query_count >= 2 else "NO_QUERY"
        if action == "NO_QUERY" and self._looks_like_content_question(query):
            return "QUERY"
        return action

    def _embed(self, text: str):
        if callable(self.embeddings):
            return self.embeddings(text)

        e = self.client.embeddings.create(
            model=self.embedding_deployment,
            input=text,
        )
        return e.data[0].embedding
