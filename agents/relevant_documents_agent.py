from openai import OpenAI
from .agent_types import safe_text



class Relevant_Documents_Agent:
    def __init__(self, openai_client, deployment) -> None:
        self.client = openai_client
        self.deployment = deployment
        self.prompt = (
            "You are a relevance judge. Given a QUESTION and a CHUNK from the indexed content, decide if the chunk could help answer the question.\n"
            "Be VERY INCLUSIVE in your judgment:\n"
            "- If the chunk discusses the same concept, related concepts, or provides background information, output Relevant\n"
            "- If the chunk mentions the topic even briefly or tangentially, output Relevant\n"
            "- Concepts under different names count as Relevant (e.g. bias-variance tradeoff vs approximation error; evaluation metrics vs accuracy/precision/recall)\n"
            "- For 'How/Why/What' questions, any chunk that provides context, explanation, or related information is Relevant\n"
            "- When in doubt, ALWAYS output Relevant\n"
            "ONLY output Not Relevant if the chunk is clearly about a completely unrelated topic with no connection to the question.\n"
            "Output EXACTLY: Relevant or Not Relevant\n"
        )

    def _judge_one(self, question: str, chunk: str) -> bool:
        resp = self.client.chat.completions.create(
            model=self.deployment,
            temperature=0,
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": f"QUESTION:\n{question}\n\nCHUNK:\n{chunk}"},
            ],
        )
        out = resp.choices[0].message.content.strip().lower()
        return out.startswith("relevant")

    def get_relevance(self, question: str, docs) -> str:
        # docs is list of RetrievedDoc; check up to 10 chunks to reduce false Not Relevant
        if not docs:
            return "Not Relevant"

        # Count how many chunks are relevant
        relevant_count = 0
        check_limit = min(10, len(docs))

        for d in docs[:check_limit]:
            chunk = (d.text or "")[:1500]
            if chunk and self._judge_one(question, chunk):
                relevant_count += 1

        # If at least 1 chunk is relevant, proceed with answering
        return "Relevant" if relevant_count > 0 else "Not Relevant"