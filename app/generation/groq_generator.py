from groq import Groq

from app.config import settings


class GroqGenerator:
    """
    Generates grounded answers using a Groq-hosted LLM.
    """

    def __init__(
        self,
        model_name: str | None = None,
    ) -> None:
        self.model_name = model_name or settings.groq_model
        self.client = Groq(api_key=settings.groq_api_key)

    def generate(
        self,
        query: str,
        passages: list[dict],
    ) -> str:
        """
        Generate a grounded answer from retrieved passages.
        """

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if not passages:
            raise ValueError(
                "At least one passage is required."
            )

        context_parts = []

        for index, passage in enumerate(passages, start=1):
            context_parts.append(
                f"[Passage {index}]\n"
                f"{passage['passage_text']}"
            )

        context = "\n\n".join(context_parts)

        system_prompt = """
You are a question-answering assistant for a retrieval-augmented
generation system.

Answer the user's question using ONLY the provided context.

Rules:
1. Do not use information that is not supported by the context.
2. If the context does not contain enough information to answer,
   clearly say that the provided context is insufficient.
3. Do not invent facts, sources, or citations.
4. Give a concise and direct answer.
5. Synthesize information from multiple passages when useful.
""".strip()

        user_prompt = f"""
Context:

{context}

Question:
{query}

Answer:
""".strip()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0.0,
            max_tokens=300,
        )

        answer = response.choices[0].message.content

        if not answer:
            raise RuntimeError(
                "LLM returned an empty response."
            )

        return answer.strip()
