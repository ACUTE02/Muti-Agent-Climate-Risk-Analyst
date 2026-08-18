"""Heat Stress agent — the project's second risk type.

Independent of the Drought Agent by design: its own target, its own features, its
own models. It reuses that agent's *infrastructure* (Open-Meteo fetch, ONI series,
window builder, Ridge baseline, skill-score definitions) rather than its features,
so the two risk types stay comparable in method and separate in signal.
"""
