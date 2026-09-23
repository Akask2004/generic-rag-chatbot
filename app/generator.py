from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

_tokenizer = None
_model = None
_generator = None


def get_generator():

    global _tokenizer
    global _model
    global _generator

    if _generator is None:

        print("Loading generation model...")

        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME
        )

        _generator = pipeline(
            "text-generation",
            model=_model,
            tokenizer=_tokenizer,
            device=-1
        )

    return _generator


def generate_answer(
    query,
    context,
    max_new_tokens=100
):

    if not query or not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    if not context or not context.strip():

        return (
            "I could not find enough information "
            "in the provided documents to answer "
            "this question."
        )

    generator = get_generator()

    prompt = f"""
You are a document question-answering system.

You MUST answer using ONLY the CONTEXT.

Important rules:

1. Do not use outside knowledge.
2. Do not invent facts.
3. Do not add explanations that are not supported
   by the CONTEXT.
4. If the document only mentions a topic, say
   that the topic is mentioned but not explained.
5. If the answer cannot be determined from the
   CONTEXT, say:
   "The provided document does not contain enough
   information to answer this."
6. Keep the answer concise.
7. Answer only the QUESTION.
8. Do not continue writing after answering.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    result = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False
    )

    answer = result[0]["generated_text"].strip()

    return answer