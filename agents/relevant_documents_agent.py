from openai import OpenAI
from .agent_types import safe_text



class Relevant_Documents_Agent:
    def __init__(self, openai_client, deployment) -> None:
        self.client = openai_client
        self.deployment = deployment
        self.prompt = (
            "You are a relevance judge. Given a QUESTION and a CHUNK from the indexed content, decide if the chunk helps answer the question.\n"
            "If the chunk discusses the same concept or closely related material (including synonyms and related terms), output Relevant.\n"
            "Concepts under different names count as Relevant (e.g. bias-variance tradeoff vs approximation error and estimation error; evaluation metrics vs accuracy, precision, recall).\n"
            "For 'How does X learn?' or 'How does X work?', a chunk that describes the mechanism, process, or training (e.g. gradient descent, updating weights, optimization) is Relevant.\n"
            "For 'What are common X?' or 'examples of X', a chunk that lists or discusses such X is Relevant. Be inclusive: when in doubt, output Relevant.\n"
            "Otherwise output Not Relevant. Output EXACTLY one token: Relevant or Not Relevant.\n"
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
        # docs is list of RetrievedDoc; check up to 7 chunks to reduce false Not Relevant
        for d in (docs or [])[:7]:
            chunk = (d.text or "")[:1500]
            if chunk and self._judge_one(question, chunk):
                return "Relevant"
        return "Not Relevant"