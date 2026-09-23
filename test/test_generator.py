from app.generator import generate_answer


def main():

    print("=" * 70)
    print("LLM GENERATOR TEST")
    print("=" * 70)

    context = """
[Source 1]
Document: LLM -1 SYLLABUS.pdf
Page: 1

Kinds of Research: Doctrinal and
Non-Doctrinal-Socio-Legal Research and
other Kinds of Researches.
"""

    query = "What types of research are mentioned?"

    print(
        f"\nQuestion: {query}"
    )

    print("\nGenerating answer...\n")

    answer = generate_answer(
        query=query,
        context=context
    )

    print("ANSWER")
    print("-" * 70)
    print(answer)


if __name__ == "__main__":
    main()