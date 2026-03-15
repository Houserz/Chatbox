from openai import OpenAI
from .agent_types import truncate, safe_text, RetrievedDoc


class Answering_Agent:
    def __init__(self, openai_client, deployment) -> None:
        self.client = openai_client
        self.deployment = deployment
        self.system_prompt = (
            "You answer the user's question using ONLY the provided context (from the indexed documents).\n"
            "If the context uses different terminology for the same concept (e.g. approximation error and estimation error for bias-variance tradeoff; accuracy/precision/recall for evaluation metrics), use that to build your answer and explain the connection.\n"
            "If the context is insufficient or does not cover the concept at all, say so briefly. Otherwise be concise and use the context.\n"
        )

    def generate_response(self, query, docs, conv_history, k=5):
        # Non-stream version (if you need it)
        stream = self.generate_stream(query, docs, conv_history, k=k)
        chunks = []
        for event in stream:
            delta = event.choices[0].delta
            if delta and delta.content:
                chunks.append(delta.content)
        return "".join(chunks).strip()

    def generate_stream(self, query, docs, conv_history, k=5):
        docs = (docs or [])[:k]

        context_blocks = []
        for i, d in enumerate(docs):
            if isinstance(d, RetrievedDoc):
                text = d.text
                md = d.metadata
            else:
                text = getattr(d, "text", "") or getattr(d, "page_content", "") or ""
                md = getattr(d, "metadata", {}) or {}

            page = md.get("page_number", md.get("page", "N/A"))
            context_blocks.append(f"[Chunk {i+1} | Page {page}]\n{truncate(text, 1800)}")

        context_text = "\n\n".join(context_blocks)

        user_prompt = f"""QUESTION:
{safe_text(query)}

CONTEXT:
{context_text}"""

        return self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
            temperature=0.2,
        )
