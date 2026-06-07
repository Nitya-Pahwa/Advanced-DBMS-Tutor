"""
Purpose:
Generates MCQ quizzes on a given DBMS topic using the LLM.

Why:
Adds interactive self-testing alongside RAG Q&A.

Fix:
Uses its own Groq API call with max_tokens=2048 instead of
safe_invoke (which caps at 512). SQL questions include code
in options and need much more room — 512 was cutting the
JSON response mid-way and crashing the parser.

Function:
  generate_quiz(topic, num_questions) -> list[dict]

Each dict:
  {
    "question": str,
    "options":  ["A. ...", "B. ...", "C. ...", "D. ..."],
    "answer":   str   # correct letter e.g. "A"
  }
"""

import os
import re
import json
from openai import OpenAI

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Separate client with higher token budget for quiz generation
_client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)


def _call_llm(prompt: str) -> str:
    """
    Direct Groq call with 2048 max_tokens.
    SQL questions with code in options easily exceed 512 tokens,
    which caused safe_invoke to return truncated (invalid) JSON.
    """
    try:
        response = _client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,      
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Quiz LLM error] {e}")
        return ""


def generate_quiz(topic: str, num_questions: int = 5) -> list:
    """
    Generate MCQs for a DBMS topic.

    Args:
        topic         : e.g. "SQL", "Normalization", "Indexing"
        num_questions : 5 or 10

    Returns:
        List of validated question dicts, or [] on failure.
    """

    prompt = f"""You are a DBMS professor generating a multiple-choice quiz.
Generate exactly {num_questions} MCQ questions about: {topic}

Rules:
- Each question must have exactly 4 options labeled A, B, C, D
- Only one option is correct
- Questions must be DBMS-specific and factual
- Vary difficulty: mix easy, medium, and hard questions
- For SQL topics, include actual SQL syntax in questions and options where relevant

Return ONLY a valid JSON array with no preamble, explanation, or markdown fences.
Each element must have exactly these keys: "question", "options", "answer".
"options" is a list of 4 strings each starting with the letter and a dot e.g. "A. some text".
"answer" is just the letter e.g. "B".

Example of ONE element (do not copy this, generate fresh questions):
{{
  "question": "Which SQL clause filters grouped results?",
  "options": [
    "A. WHERE",
    "B. HAVING",
    "C. GROUP BY",
    "D. ORDER BY"
  ],
  "answer": "B"
}}

Now generate {num_questions} questions about: {topic}
Return only the JSON array."""

    raw = _call_llm(prompt)

    if not raw:
        return []

    try:
        # Strip accidental markdown fences if model adds them
        cleaned = re.sub(r'```(?:json)?|```', '', raw).strip()

        # Extract the JSON array even if there is stray text around it
        m = re.search(r'\[.*\]', cleaned, re.DOTALL)
        if not m:
            print(f"[Quiz] No JSON array found in response:\n{cleaned[:300]}")
            return []

        data = json.loads(m.group())

        # Validate each question has the required structure
        validated = []
        for item in data:
            if (
                "question" in item
                and "options"  in item
                and "answer"   in item
                and isinstance(item["options"], list)
                and len(item["options"]) == 4
                and isinstance(item["answer"], str)
                and len(item["answer"].strip()) == 1   # single letter
            ):
                # Normalise answer to uppercase
                item["answer"] = item["answer"].strip().upper()
                validated.append(item)

        return validated[:num_questions]

    except json.JSONDecodeError as e:
        print(f"[Quiz JSON error] {e}\nRaw response:\n{raw[:400]}")
        return []
    except Exception as e:
        print(f"[Quiz parse error] {e}")
        return []