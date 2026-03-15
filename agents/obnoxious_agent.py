from openai import OpenAI
from .agent_types import safe_text


class Obnoxious_Agent:
    def __init__(self, client, deployment) -> None:
        self.client = client
        self.deployment = deployment
        self.prompt = (
            "You are a strict classifier. Output EXACTLY one token: Yes or No.\n"
            "The user message is obnoxious ONLY if it contains clear: insults to the assistant or person (e.g. idiot, stupid, dumb, moron, lazy bot, worthless), "
            "profanity (e.g. damn, hell, dammit), or hostile tone (e.g. 'just answer me', 'stop being annoying').\n"
            "Mixing a content question with an unrelated clause (e.g. 'Explain X. By the way, what is the weather?') is NOT obnoxious — there are no insults or hostility. Output No for such messages.\n"
            "Example: 'Answer me, you idiot.' -> Yes. Example: 'Explain activation functions. On a different note, what is the weather?' -> No.\n"
        )

    def set_prompt(self, prompt):
        self.prompt = prompt

    def extract_action(self, response) -> bool:
        s = safe_text(response).strip().lower()
        return s == "yes"

    def check_query(self, query):
        resp = self.client.chat.completions.create(
            model=self.deployment,
            temperature=0,
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": safe_text(query)},
            ],
        )
        out = resp.choices[0].message.content.strip()
        return "Yes" if self.extract_action(out) else "No"
