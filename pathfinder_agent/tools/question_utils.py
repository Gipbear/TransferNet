"""Question text helpers for PathfinderAgent prompts."""


def normalize_question(question: str) -> str:
    """Remove WebQSP tokenization artifacts before sending text to the LLM."""
    question = question.replace("[CLS]", "").replace("[SEP]", "")
    question = question.replace(" ##", "").replace("##", "")
    return " ".join(question.split())
