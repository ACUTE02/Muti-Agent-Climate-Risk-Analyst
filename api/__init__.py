"""Phase 6: the local API.

A thin wrapper over ``orchestrator.graph.analyse()`` — the same function the
tests exercise. There is deliberately no separate "demo" path: if the API
returns a report, it went through the real pipeline, the real tools and the real
grounding checker.
"""
