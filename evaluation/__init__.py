"""Phase 5: the evaluation suite.

Two kinds of measurement live here, and they are deliberately different in cost
and in trust:

- ``checker_eval`` — the grounding checker's own precision and recall on a
  labelled adversarial set. Fully offline, deterministic, free, and the
  higher-trust of the two.
- ``faithfulness`` — sentence-level faithfulness and answer relevance of real
  generated reports. Costs live Gemini calls and is the one place in this whole
  project where an LLM judges an LLM's output. Explicitly logged as lower-trust
  than the mechanical checker, and complementary to it rather than a replacement.

``EVALUATION.md`` at the repo root is the consolidated scorecard both feed into.
"""
