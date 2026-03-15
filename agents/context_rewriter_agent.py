from openai import OpenAI
from .agent_types import safe_text


class Context_Rewriter_Agent:
    def __init__(self, openai_client, deployment):
        self.client = openai_client
        self.deployment = deployment
        self.prompt = (
            "Your ONLY job is to replace pronouns and abbreviations in the user's query. Make MINIMAL changes.\n"
            "RULES:\n"
            "1. If the query contains pronouns (it, they, this, that), replace them with the concrete topic from conversation history\n"
            "2. If the query contains abbreviations/acronyms, expand them to full form (without keeping abbreviation in parentheses)\n"
            "3. Keep the original sentence structure and wording - do NOT rephrase or rewrite the entire question\n"
            "4. If the query is too vague or incomplete (single word like 'what', 'ok', 'see'), return it UNCHANGED\n"
            "Examples:\n"
            "- 'Why is it important?' (after cross-validation) -> 'Why is cross-validation important?'\n"
            "- 'What is SVM?' -> 'What is Support Vector Machine?'\n"
            "- 'what' -> 'what' (unchanged)\n"
            "- 'How does it work?' (after neural networks) -> 'How do neural networks work?'\n"
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
            messages=[
                {"role": "system", "content": self.prompt},
                {
                    "role": "user",
                    "content": f"Conversation:\n{history_text}\n\nLatest query:\n{safe_text(latest_query)}",
                },
            ],
        )
        return resp.choices[0].message.content.strip()
