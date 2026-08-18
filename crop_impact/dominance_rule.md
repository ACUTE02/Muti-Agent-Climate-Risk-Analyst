# Dominant-risk decision rule

The rule `crop_impact/dominance.py` implements, written out so it can be reviewed
without reading code — the same reason `orchestrator/prompts/synthesis.md` is a
file rather than a Python string. **If the code and this document disagree, that
is a bug in one of them.** A test asserts the thresholds quoted here match
`crop_impact/config.py`.

## Why this is not an LLM's job

The standing scope decision (17 Aug 2026) requires the Crop Impact Agent to
"determine which risk factor is actually dominant before calculating yield loss"
rather than assuming drought always is. That is exactly the kind of question a
language model answers fluently from general agronomic knowledge whether or not
it has grounds to for *these* regions, *this* month and *this* project's measured
signals. So it is plain Python, unit-tested directly, with no model call in the
module — the same posture as `orchestrator/grounding.py`.

## Inputs

| Signal | Source | Nature |
|---|---|---|
| Drought | `forecast_drought_risk(region)` | **Forecast.** Predicted SPI-3 at t+1/t+2/t+3, each horizon carrying its own measured skill label. |
| Heat | `forecast_heat_stress_risk(region, month)` | **Observation only.** IMD heat wave day counts for a month that has already happened. There is no heat forecast — Heat Phases 1 and 1.1 measured 36 cells and found nothing above +0.0378 against a +0.1 bar. |

The asymmetry is the whole reason this rule is not a simple "compare two risk
scores" — the two signals are not the same kind of claim.

## The rule, in order

### 1. Season gate

If the month asked about is outside the crop's sensitive window, the result is
`none dominant`, regardless of how severe either signal is.

| Crop | Season | Sensitive window | Stage |
|---|---|---|---|
| Bajra (pearl millet) | Kharif | August–September | flowering and grain filling |
| Wheat | Rabi | February–March | grain filling (terminal heat window) |

Rationale: a severe SPI-3 in January does not reduce a bajra yield, because there
is no bajra in the ground. Attributing loss to a month the crop is not vulnerable
in would be a confident, well-formed, wrong answer.

### 2. Drought candidacy — gated by measured skill, not by value

Drought may be declared dominant **only** if all of the following hold:

- the requested horizon exists in the forecast, and
- the predicted SPI-3 is below the moderate threshold (`SPI_MODERATE = -1.0`;
  `severe` below `SPI_SEVERE = -1.5`), and
- **the horizon's label is exactly `validated`.**

The label gate is the single most important correctness property in this phase:

| Label | Meaning | May it decide dominance? |
|---|---|---|
| `validated` | measured skill above the project's +0.1 bar, replicated across four independent windows | **Yes** |
| `weak/directional` | positive but under the bar | No — may corroborate in the narrative, never decide |
| `no skill — shown for context only, do not rely on this figure` | at or below climatology | **Never**, whatever value it carries |

A t+3 SPI-3 of −2.5 therefore declares nothing. Five phases of evidence say that
number is worthless; letting it drive a crop assessment would launder it into
something that looks like a finding.

### 3. Heat candidacy — an observation, and only for the month observed

Heat is considered only if the heat tool's observation is **for the month being
asked about**. If the observation is for a different month, heat is reported as
*unknown* for the target month — never inferred, never extrapolated.

There are then two independent routes to candidacy, and the more severe wins:

| Route | Candidate at | Severe at |
|---|---|---|
| IMD heat wave day count | ≥ 3 heat wave days in the month | any *severe* heat wave day |
| Monthly mean Tmax departure from the 1981–2010 normal | ≥ +3.0 °C | ≥ +4.0 °C |

**Why the second route exists — a mismatch found by measurement, not assumed.**
IMD's plains heat wave criteria gate on Tmax ≥ 40 °C, which February and March at
these sites almost never reach. Measured directly across the record: the warmest
February at Jaipur (2006, mean Tmax **+5.35 °C** above normal) records **zero**
IMD heat wave days, and so does the record March of 2022 (+3.68 °C). The day
counter is a summer indicator; wheat's grain-filling window is Feb–March. Judging
terminal heat stress by heat wave days alone would report "no heat" for every
warm wheat season on record — a metric that passes because it cannot see the
thing it is being asked about, which is a failure shape this project has hit
before.

The **+4.0 °C severe threshold is not arbitrary**: it matches the `match_band` of
the one sourced coefficient in `yield_impact_table.json`, so that coefficient is
never applied outside the exposure range it was measured in. The **+3.0 °C
moderate threshold has no source behind it** and is flagged as a judgement call
in `config.py`, the same posture as `spi_to_risk_score` — a threshold for Phase 5
to calibrate, not a measured quantity.

Because heat is never a forecast, a request about a **future** month can never
return `heat` as dominant. That is a real limitation of the system, and the
returned reasoning says so in words rather than leaving it implicit.

### 4. Comparison and tie-break

If both qualify, the more severe wins. On a tie, **heat wins** — an observation is
a fact and a forecast is an estimate, so when the two look equally severe the fact
is taken as binding. The reasoning string states that this is what happened.

### 5. Outcomes

| Outcome | When |
|---|---|
| `drought` | drought qualified under §2 and outranked heat |
| `heat` | heat qualified under §3 and outranked drought |
| `none dominant` | signals were usable but nothing crossed a threshold, or §1 gated it |
| `insufficient data` | no usable drought signal (absent, or only a no-skill horizon) **and** no heat observation for the month |

`insufficient data` and `none dominant` are deliberately different answers. The
first means the system could not see; the second means it looked and found
nothing binding.

## What happens after the decision

The dominant risk and its severity are looked up in
`crop_impact/yield_impact_table.json`. That table contains one sourced
coefficient (wheat × heat) and three recorded gaps, and the lookup enforces each
coefficient's own exposure band — so a combination with no real source returns
"no sourced yield-impact estimate available" rather than a plausible number. The
LLM is handed whatever comes back and asked only to explain it.
