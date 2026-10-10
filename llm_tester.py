#!/usr/bin/env python3
"""
Local LLM tester for Tasks 1-9.
Requires Ollama to be installed and running locally.
Run: python llm_tester.py
"""

import csv
import json
import re
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

MODEL = "llama3.2:3b"
OLLAMA_BASE_URL = "http://localhost:11434"
RESULTS_FILE = Path("results.csv")
README_FILE = Path("README.md")

# Keep the test prompts and expected outcomes visible and easy to edit.
POLICY = (
    "Acme Company Leave Policy:\n"
    "- Employees receive 20 days of annual leave per year.\n"
    "- Employees can carry forward up to 5 unused leave days.\n"
    "- Leave requests must be submitted at least 3 working days in advance.\n"
    "- This policy does not specify whether unused leave is paid out when an employee leaves.\n"
)

RESULT_FIELDS = [
    "timestamp", "model", "task", "test_id", "test_type", "prompt",
    "expected_result", "actual_response", "score", "status", "notes"
]


def http_json(path, payload=None, timeout=180):
    """Send a JSON request to the local Ollama API and return its JSON response."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_BASE_URL + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def model_response(prompt, temperature=0.0):
    """Generate one non-streaming response from the selected local model."""
    result = http_json(
        "/api/generate",
        {
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        },
    )
    return result.get("response", "").strip()


def has_uncertainty(response):
    """Basic heuristic for checking whether the model acknowledges uncertainty."""
    phrases = [
        "don't know", "do not know", "not specified", "not provided",
        "cannot determine", "can't determine", "unable to determine",
        "no information", "insufficient information", "not enough information",
        "unknown", "cannot say", "can't say", "does not state", "doesn't state",
        "don't have reliable information", "do not have reliable information",
        "not explicitly stated", "not explicitly specified", "not stated explicitly",
        "does not explicitly say", "doesn't explicitly say",
    ]
    lower = response.lower()
    return any(phrase in lower for phrase in phrases)


def has_number(response, number, word=None):
    """Check for a target number as digits or, optionally, as a word."""
    if re.search(rf"(?<!\d){number}(?!\d)", response):
        return True
    return bool(word and re.search(rf"\b{re.escape(word)}\b", response, re.IGNORECASE))


def add_result(rows, task, test_id, test_type, prompt, expected, actual,
               score, notes=""):
    """Store each run as one CSV row. Scores are 1.0 for pass and 0.0 for fail."""
    rows.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": MODEL,
        "task": task,
        "test_id": test_id,
        "test_type": test_type,
        "prompt": prompt,
        "expected_result": expected,
        "actual_response": actual,
        "score": round(float(score), 2),
        "status": "PASS" if score >= 1.0 else "FAIL",
        "notes": notes,
    })


def run_tests():
    rows = []

    # Task 1: confirm that Ollama is reachable and the selected model is available.
    tags = http_json("/api/tags")
    installed_models = [item.get("name", "") for item in tags.get("models", [])]
    model_available = any(name == MODEL or name.startswith(MODEL + "-") for name in installed_models)
    if not model_available:
        raise RuntimeError(
            f"Model '{MODEL}' was not found by Ollama. Run: ollama pull {MODEL}"
        )
    print(f"[Task 1/9] Ollama reachable; model available: {MODEL}")

    # Task 2: generate a simple response to prove the model can answer.
    prompt = "What is an API? Explain in one short sentence."
    answer = model_response(prompt)
    add_result(rows, 2, "T02", "Response generation", prompt,
               "A short, relevant explanation of an API", answer,
               1 if len(answer) > 10 else 0,
               "Basic generation check; relevance is reviewed manually.")
    print("[Task 2/9] Generated a sample response.")

    # Task 3: ground-truth tests use predefined, objectively checkable answers.
    ground_truth_tests = [
        ("T03A", "What is 18 multiplied by 7? Give the calculation and final answer.",
         "126", lambda r: has_number(r, 126)),
        ("T03B",
         "A bookstore has 240 notebooks. It sells 35% of the original stock in the morning "
         "and 25% of the original stock in the afternoon. How many remain? Show your calculation.",
         "96 notebooks", lambda r: has_number(r, 96)),
    ]
    for test_id, prompt, expected, checker in ground_truth_tests:
        actual = model_response(prompt)
        passed = checker(actual)
        add_result(rows, 3, test_id, "Ground truth", prompt, expected, actual,
                   1 if passed else 0,
                   "Automated check looks for the expected answer; inspect reasoning manually.")
    print("[Task 3/9] Ground-truth tests completed.")

    # Task 4: compare exact-answer accuracy before and after adding the same context.
    # These are fictional policy facts, so the model should not know them beforehand.
    context_cases = [
        {
            "id": "T04A",
            "question": "According to Acme Company's internal leave policy, how many unused leave days can be carried forward?",
            "expected": "5 days",
            "checker": lambda r: has_number(r, 5, "five"),
        },
        {
            "id": "T04B",
            "question": "According to Acme Company's internal leave policy, how far in advance must leave be requested?",
            "expected": "3 working days",
            "checker": lambda r: has_number(r, 3, "three") and "working" in r.lower(),
        },
    ]
    context_scores_before = []
    context_scores_after = []
    for case in context_cases:
        before_prompt = (
            case["question"] +
            " If you do not have the policy information, say that it is not specified."
        )
        before_answer = model_response(before_prompt)
        before_pass = case["checker"](before_answer)
        context_scores_before.append(1 if before_pass else 0)
        add_result(
            rows, 4, case["id"] + "-BEFORE", "Context/evidence comparison",
            before_prompt, case["expected"], before_answer, 1 if before_pass else 0,
            "Before context; scored against the exact policy fact."
        )

        after_prompt = (
            "Answer using only this context. If a fact is absent, say it is not specified.\n\n"
            + POLICY + "\nQuestion: " + case["question"]
        )
        after_answer = model_response(after_prompt)
        after_pass = case["checker"](after_answer)
        context_scores_after.append(1 if after_pass else 0)
        add_result(
            rows, 4, case["id"] + "-AFTER", "Context/evidence comparison",
            after_prompt, case["expected"], after_answer, 1 if after_pass else 0,
            "After context; scored against the exact policy fact."
        )
    before_pct = 100 * sum(context_scores_before) / len(context_scores_before)
    after_pct = 100 * sum(context_scores_after) / len(context_scores_after)
    print(f"[Task 4/9] Context comparison: before={before_pct:.0f}%, after={after_pct:.0f}%.")

    # Task 5: hallucination tests check whether the model admits missing information.
    hallucination_tests = [
        (
            "T05A",
            "What is the exact founding date of the fictional company BlueNova Technologies? "
            "If you do not have reliable information, say you do not know. Do not guess.",
            "Admit the date is unknown; do not invent a date.",
        ),
        (
            "T05B",
            'A document states: "Acme Company provides 20 days of annual leave per year." '
            "Based only on this sentence, does Acme pay out unused leave when an employee resigns? "
            "Say it is not specified if the sentence does not say.",
            "Say the payout policy is not specified.",
        ),
    ]
    hallucination_flags = []
    for test_id, prompt, expected in hallucination_tests:
        actual = model_response(prompt)
        uncertain = has_uncertainty(actual)
        # A pass means the answer signals uncertainty instead of asserting an unsupported fact.
        passed = uncertain
        hallucination_flags.append((test_id, prompt, actual, not passed))
        add_result(
            rows, 5, test_id, "Hallucination", prompt, expected, actual,
            1 if passed else 0,
            "Heuristic: checks for uncertainty language; manually inspect for unsupported claims."
        )
    print("[Task 5/9] Hallucination checks completed.")

    # Task 6: repeated runs should agree on the same objectively known answer.
    consistency_prompt = "What is 24 multiplied by 6? Give only the final number."
    consistency_answers = []
    for run_number in range(1, 4):
        actual = model_response(consistency_prompt)
        consistency_answers.append(actual)
        passed = has_number(actual, 144)
        add_result(
            rows, 6, f"T06-R{run_number}", "Consistency", consistency_prompt,
            "144 on every run", actual, 1 if passed else 0,
            f"Repeated run {run_number}/3; compare all three actual responses."
        )
    consistency_normalized = [
        re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", x.lower())).strip()
        for x in consistency_answers
    ]
    consistency_agrees = len(set(consistency_normalized)) == 1
    print(f"[Task 6/9] Consistency completed; identical normalized responses={consistency_agrees}.")

    # Task 7: equivalent prompts should lead to the same correct result.
    sensitivity_tests = [
        ("T07A", "A store starts with 150 pens, sells 40, then receives 25. How many pens remain?",
         "135", lambda r: has_number(r, 135)),
        ("T07B", "Calculate the final stock: 150 pens initially, minus 40 sold, plus 25 delivered.",
         "135", lambda r: has_number(r, 135)),
    ]
    sensitivity_passes = []
    for test_id, prompt, expected, checker in sensitivity_tests:
        actual = model_response(prompt)
        passed = checker(actual)
        sensitivity_passes.append(passed)
        add_result(rows, 7, test_id, "Prompt sensitivity", prompt, expected, actual,
                   1 if passed else 0,
                   "Equivalent wording; score checks the expected final quantity.")
    print("[Task 7/9] Prompt-sensitivity tests completed.")

    # Task 8: ask for strict JSON, then parse and validate its schema.
    format_tests = [
        ("T08A", 'Return only valid JSON with keys "name" (string "Alex") and "age" (integer 25).'),
        ("T08B", 'Return only valid JSON with keys "annual_leave_days" (integer 20) and "carry_forward_days" (integer 5).'),
    ]
    for test_id, prompt in format_tests:
        actual = model_response(prompt)
        passed = False
        note = "Invalid JSON or schema mismatch."
        try:
            # Strip common Markdown fences before parsing; extra prose still fails JSON parsing.
            candidate = actual.strip()
            candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.IGNORECASE)
            obj = json.loads(candidate)
            if test_id == "T08A":
                passed = (
                    set(obj.keys()) == {"name", "age"}
                    and obj.get("name") == "Alex"
                    and type(obj.get("age")) is int
                    and obj.get("age") == 25
                )
                expected = '{"name": "Alex", "age": 25}'
            else:
                passed = (
                    set(obj.keys()) == {"annual_leave_days", "carry_forward_days"}
                    and type(obj.get("annual_leave_days")) is int
                    and type(obj.get("carry_forward_days")) is int
                    and obj.get("annual_leave_days") == 20
                    and obj.get("carry_forward_days") == 5
                )
                expected = '{"annual_leave_days": 20, "carry_forward_days": 5}'
            note = "Valid JSON and schema match." if passed else "JSON parsed, but required keys or values did not match."
        except (json.JSONDecodeError, AttributeError):
            expected = "Valid JSON with required keys and values"
        add_result(rows, 8, test_id, "Format", prompt, expected, actual,
                   1 if passed else 0, note)
    print("[Task 8/9] Format tests completed.")

    # Task 9: write the full run log and summarize scores for comparison.
    with RESULTS_FILE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    scores_by_type = defaultdict(list)
    for row in rows:
        # Task 2 is a manual-review generation check, so exclude it from scored test ranking.
        if row["task"] != 2:
            scores_by_type[row["test_type"]].append(float(row["score"]))

    type_scores = {
        name: 100 * sum(values) / len(values)
        for name, values in scores_by_type.items()
        if values
    }
    overall = 100 * sum(float(r["score"]) for r in rows if r["task"] != 2) / max(
        1, sum(1 for r in rows if r["task"] != 2)
    )
    worst_test = min(type_scores, key=type_scores.get) if type_scores else "Not available"
    hallucinated = [
        (test_id, prompt, actual)
        for test_id, prompt, actual, flagged in hallucination_flags if flagged
    ]

    # Build a run-specific README with actual measured results and cautious interpretations.
    hallucination_section = (
        f"Potential hallucination/unsupported answer detected in **{hallucinated[0][0]}**.\n\n"
        f"- **Question:** {hallucinated[0][1]}\n"
        f"- **Observed response:** {hallucinated[0][2]}\n"
        "- **Prevention:** Provide trusted context, require citations to that context, "
        "instruct the model to abstain when evidence is missing, and keep human review for risky answers."
        if hallucinated else
        "The configured heuristic did not flag a clear hallucination in this run. Do not claim a specific "
        "hallucination unless manual review of `results.csv` identifies one. The heuristic is imperfect and "
        "all flagged/unflagged answers should be reviewed."
    )

    readme = f"""# Local LLM Tester

## Project overview
This project tests a local language model using Ollama. It covers Tasks 1–9: setup, response generation,
ground truth, context/evidence comparison, hallucination, consistency, prompt sensitivity, format validation,
and recording/comparing results.

## Model and setup
- **Model:** `{MODEL}`
- **Runner:** Ollama
- **Language:** Python 3
- **API:** `http://localhost:11434`

## Run the project
1. Install Ollama from https://ollama.com/download.
2. Make sure Ollama is running.
3. Pull the model if needed: `ollama pull {MODEL}`
4. Save `llm_tester.py` in a folder.
5. Run: `python llm_tester.py`
6. The script creates `results.csv` and refreshes this README with the measured scores.

No third-party Python packages are required.

## Scores from the latest run
- **Overall scored-test accuracy:** {overall:.1f}%
- **Ground truth:** {type_scores.get("Ground truth", 0):.1f}%
- **Context/evidence before context:** {before_pct:.1f}%
- **Context/evidence after context:** {after_pct:.1f}%
- **Hallucination checks:** {type_scores.get("Hallucination", 0):.1f}%
- **Consistency:** {type_scores.get("Consistency", 0):.1f}%
- **Prompt sensitivity:** {type_scores.get("Prompt sensitivity", 0):.1f}%
- **Format:** {type_scores.get("Format", 0):.1f}%
- **Lowest-scoring test category:** {worst_test} ({type_scores.get(worst_test, 0):.1f}%)

Scores are based on simple automated checks, not a comprehensive evaluation of language quality.
Review `results.csv` before drawing conclusions. The context score measures exact-answer matching on two
fictional policy questions before and after context is supplied.

## Mentor questions

### Which test did the model perform worst on, and why?
The lowest-scoring category in this run was **{worst_test}** at {type_scores.get(worst_test, 0):.1f}%.
This likely reflects the model outputs and the specific automated criteria used. Review the corresponding
rows in `results.csv`; a low score can also indicate a strict or imperfect heuristic.

### Did the context paragraph (Task 4) improve accuracy?
Before context: **{before_pct:.1f}%**. After context: **{after_pct:.1f}%**.
{"Yes, accuracy improved." if after_pct > before_pct else "Accuracy did not improve in this run." if after_pct == before_pct else "Accuracy decreased in this run."}
This is a small two-question experiment, so it should not be treated as a broad benchmark.

### Share one question where the model hallucinated. What did it make up, and how would you prevent it?
{hallucination_section}

### Would you trust this model to answer customer questions without a human check? Why or why not?
No—not without human review for important or customer-facing answers. This project uses a small test set and
simple heuristics, and a passing score does not guarantee factual reliability. I would use it for low-risk
drafting with source-grounded answers, clear uncertainty behavior, monitoring, and human escalation for
ambiguous or consequential cases.

## Files produced
- `llm_tester.py` — test runner.
- `results.csv` — timestamped prompts, expected results, actual responses, scores, statuses, and notes.
- `README.md` — this file, updated by the latest run.

## Limitations
- The model may produce different wording across runs.
- Some evaluators check for target numbers or uncertainty phrases and can miss subtle errors.
- Evidence quality, exact quotations, and possible hallucinations require human review.
- This script is an educational test harness, not a production-grade evaluation framework.
"""
    README_FILE.write_text(readme, encoding="utf-8")

    print("\n[Task 9/9] Results saved.")
    print(f"CSV: {RESULTS_FILE.resolve()}")
    print(f"README updated: {README_FILE.resolve()}")
    print(f"Overall scored-test accuracy: {overall:.1f}%")
    print("Scores by category:")
    for name, score in sorted(type_scores.items()):
        print(f"  - {name}: {score:.1f}%")
    print(f"Context accuracy: {before_pct:.1f}% before -> {after_pct:.1f}% after")
    print(f"Lowest-scoring category: {worst_test}")
    print("\nTake a screenshot of this terminal output for your submission.")


def main():
    try:
        run_tests()
    except urllib.error.URLError as exc:
        print("ERROR: Could not connect to Ollama at http://localhost:11434.")
        print("Make sure Ollama is running, then try again.")
        print(f"Details: {exc}")
        raise SystemExit(1)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
    except Exception as exc:
        print(f"ERROR: Test run failed: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
