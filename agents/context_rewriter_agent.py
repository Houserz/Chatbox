from openai import OpenAI
from .agent_types import safe_text


class Context_Rewriter_Agent:
    def __init__(self, openai_client, deployment):
        self.client = openai_client
        self.deployment = deployment
        self.prompt = (
            "Your task: Make the query self-contained by replacing unclear references with specific terms from conversation history.\n\n"
            "ONLY modify the query if it contains:\n"
            "- Pronouns that refer to previous topics: 'it', 'they', 'this', 'that', 'these', 'those'\n"
            "- Explicit references to previous conversation: 'you mentioned', 'you said', 'earlier answer', 'based on your answer'\n"
            "- Unclear abbreviations that need expansion\n\n"
            "If the query has none of these, return it UNCHANGED.\n\n"
            "When modifying:\n"
            "- Make MINIMAL changes - only replace the unclear reference\n"
            "- Keep the original sentence structure, tone, and question format\n"
            "- Do NOT rephrase, reorganize, or change the meaning\n"
            "- Do NOT add extra context beyond what's needed to clarify the reference\n\n"
            "Output only the query, nothing else."
        )

    def rephrase(self, user_history, latest_query):
        history_text = ""
        for m in (user_history or [])[-8:]:
            role = m.get("role", "user")
            content = m.get("content", "")
            history_text += f"{role.upper()}: {content}\n"

        resp = self.client.chat.completions.create(
            model=self.deployment,
            temperature=0,
            seed=42,
            messages=[
                {"role": "system", "content": self.prompt},
                {
                    "role": "user",
                    "content": f"Conversation:\n{history_text}\n\nLatest query:\n{safe_text(latest_query)}",
                },
            ],
        )
        return resp.choices[0].message.content.strip()
