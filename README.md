# Local LLM Tester

## Project overview
This project tests a local language model using Ollama. It covers Tasks 1–9: setup, response generation,
ground truth, context/evidence comparison, hallucination, consistency, prompt sensitivity, format validation,
and recording/comparing results.

## Model and setup
- **Model:** `llama3.2:3b`
- **Runner:** Ollama
- **Language:** Python 3
- **API:** `http://localhost:11434`

## Run the project
1. Install Ollama from https://ollama.com/download.
2. Make sure Ollama is running.
3. Pull the model if needed: `ollama pull llama3.2:3b`
4. Save `llm_tester.py` in a folder.
5. Run: `python llm_tester.py`
6. The script creates `results.csv` and refreshes this README with the measured scores.

No third-party Python packages are required.

## Scores from the latest run
- **Overall scored-test accuracy:** 73.3%
- **Ground truth:** 100.0%
- **Context/evidence before context:** 0.0%
- **Context/evidence after context:** 100.0%
- **Hallucination checks:** 100.0%
- **Consistency:** 100.0%
- **Prompt sensitivity:** 100.0%
- **Format:** 0.0%
- **Lowest-scoring test category:** Format (0.0%)

Scores are based on simple automated checks, not a comprehensive evaluation of language quality.
Review `results.csv` before drawing conclusions. The context score measures exact-answer matching on two
fictional policy questions before and after context is supplied.

## Mentor questions

### Which test did the model perform worst on, and why?
The lowest-scoring category in this run was **Format** at 0.0%.
This likely reflects the model outputs and the specific automated criteria used. Review the corresponding
rows in `results.csv`; a low score can also indicate a strict or imperfect heuristic.

### Did the context paragraph (Task 4) improve accuracy?
Before context: **0.0%**. After context: **100.0%**.
Yes, accuracy improved.
This is a small two-question experiment, so it should not be treated as a broad benchmark.

### Share one question where the model hallucinated. What did it make up, and how would you prevent it?
The configured heuristic did not flag a clear hallucination in this run. Do not claim a specific hallucination unless manual review of `results.csv` identifies one. The heuristic is imperfect and all flagged/unflagged answers should be reviewed.

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
