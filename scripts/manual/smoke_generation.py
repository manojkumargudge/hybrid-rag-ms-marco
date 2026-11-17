from app.generation.groq_generator import GroqGenerator


def main() -> None:
    print("Loading Groq Generator...")

    generator = GroqGenerator()

    query = "what are the symptoms of diabetes?"

    passages = [
        {
            "passage_id": 24998,
            "passage_text": (
                "Common symptoms of diabetes include frequent "
                "urination, excessive thirst, unusual hunger, "
                "fatigue, and unexplained weight loss."
            ),
            "score": 9.09,
        },
        {
            "passage_id": 191966,
            "passage_text": (
                "Diabetes symptoms can include frequent urination, "
                "intense thirst and hunger, unusual weight loss, "
                "fatigue, and numbness or tingling in the hands "
                "and feet."
            ),
            "score": 9.03,
        },
    ]

    print()
    print(f"Query: {query}")

    print()
    print("Generating answer...")

    answer = generator.generate(
        query=query,
        passages=passages,
    )

    print()
    print("=== GENERATED ANSWER ===")
    print(answer)

    print()
    print("=== VERIFICATION ===")

    assert answer
    assert isinstance(answer, str)

    print("✓ Answer generated successfully")
    print("✓ Answer is a non-empty string")
    print("✓ Groq LLM generation working correctly")


if __name__ == "__main__":
    main()