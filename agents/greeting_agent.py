from openai import OpenAI
from .agent_types import safe_text


# Common greetings: if user message (stripped, lower) equals one of these, treat as greeting without calling LLM.
COMMON_GREETINGS = frozenset({
    "hello", "hi", "hey", "hi there", "hey there",
    "good morning", "good afternoon", "good evening", "good night",
    "greetings", "howdy", "how are you", "what's up", "whats up",
    "how's it going", "hows it going", "nice to see you", "good day",
})


class Greeting_Agent:
    """Detects greetings/small talk and returns a friendly response."""

    def __init__(self, client, deployment) -> None:
        self.client = client
        self.deployment = deployment
        self.prompt = (
            "You decide if the user's message is ONLY a greeting or small talk.\n"
            "Examples of greeting/small talk: hello, hi, good morning, good afternoon, "
            "how are you, what's up, hey, greetings, how's it going, nice to see you.\n"
            "If the user asks a substantive question (even with a greeting), output: No.\n"
            "Output EXACTLY one token: Yes or No.\n"
        )
        self.reply_prompt = (
            "The user said a greeting or small talk. Reply in one short, friendly sentence. "
            "Do not push the user toward a specific topic; keep it brief and warm.\n"
        )

    def is_greeting(self, query: str) -> bool:
        q = safe_text(query).strip().lower()
        if q in COMMON_GREETINGS:
            return True
        resp = self.client.chat.completions.create(
            model=self.deployment,
            temperature=0,
            seed=42,
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": safe_text(query)},
            ],
        )
        out = resp.choices[0].message.content.strip().lower()
        return out.startswith("yes")

    def get_reply(self, query: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.deployment,
            temperature=0.3,
            seed=42,
            messages=[
                {"role": "system", "content": self.reply_prompt},
                {"role": "user", "content": safe_text(query)},
            ],
        )
        return (resp.choices[0].message.content or "").strip() or "Hello! How can I help you today?"
