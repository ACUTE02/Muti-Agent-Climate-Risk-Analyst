---
Candidate Regions — Drought Forecasting Agent (District-Level Switch)
Multi-Agent Climate Risk Analyst | Researched Aug 2026
---

Four candidates for replacing the current single-point Jaipur centroid. All four are real, documented drought-relevant Indian districts/regions — not arbitrary picks. Centroid coordinates are given for the Open-Meteo fallback fetch; the primary source for each should be IMD's actual district-level rainfall records where available.

## 1. Barmer, Rajasthan

- **Centroid:** 25.75°N, 71.38°E
- **Why it's a strong candidate:** A genuine Thar Desert district with a dedicated academic literature on its rainfall variability and drought characteristics (see Sources). Already named in your own original tech spec's advisory example ("Activate early warning systems: Barmer, Jaisalmer, Bikaner").
- **Caveat worth knowing (new, from Aug 2026 research):** Rajasthan's desert belt is showing a documented recent shift toward *more* rainfall, not less. A 2025 analysis found an increasing rainfall trend at 54 of 62 stations across the Thar region, six flood years between 2011–2020, and a near-doubling of irrigated cropland in five desert districts over the past decade. This doesn't erase Barmer's historical drought record, but it means recent years may look less classically "drought" than the district's reputation suggests — worth knowing before you frame it in interviews as a straightforward drought example.
- **Least rework:** stays within Rajasthan, same state as the current default.

## 2. Jaisalmer, Rajasthan

- **Centroid:** 26.91°N, 70.90°E
- **Why it's a candidate:** Also classic Thar Desert, also named in your original spec's advisory example.
- **Same caveat as Barmer, more pronounced:** the same 2025 research specifically calls out Jaisalmer as an example of the desert-to-wetter shift ("Jaisalmer's Surprising Rainfall Surge: Climate Change Turns Desert Wet"). Of the four candidates, this is the one whose "drought-prone" identity is most actively being complicated by recent climate trends — worth avoiding if you want your test period (2020–2024) to actually contain clear drought signal.

## 3. Marathwada region (Latur / Osmanabad–Dharashiv), Maharashtra

- **Centroid (Latur):** 18.40°N, 76.58°E — or Osmanabad/Dharashiv at 18.18°N, 76.04°E
- **Why it's a strong candidate:** India's single most infamous modern drought region. The 2016 Latur water crisis (the "Jaldoot Express" water trains) is extensively documented in academic case studies, government reports, and news coverage — no ambiguity about whether this region has real, severe, well-recorded drought events.
- **Bonus for later phases:** this depth of documentation is exactly what Phase 2 (Retrieval Agent) needs — real news articles, NDMA-style disaster records, and academic case studies about Marathwada exist in volume, which means richer, more credible RAG corpus material than a less-covered district would offer.
- **Tradeoff:** bigger rework than staying in Rajasthan — new state, new IMD naming conventions to look up, and it's technically a multi-district region rather than one administrative unit, so you'd need to pick one specific district (Latur or Osmanabad/Dharashiv) as the actual data source.

## 4. Anantapur, Andhra Pradesh

- **Centroid:** 14.68°N, 77.60°E
- **Why it's a candidate:** A classic rain-shadow district (sits in the rain shadow of the Western Ghats), one of the most chronically drought-affected districts in India by long-term academic consensus, with decades of dedicated rainfall-statistics literature.
- **Tradeoff:** same as Marathwada — new state, new data-source lookup, more rework than the Rajasthan options. No specific recent-climate-shift caveat found in this research pass, which arguably makes it the most "stable" classic-drought choice of the four.

---

## Recommendation

If minimizing rework matters most: **Barmer** — but go in aware of the recent wetter-trend caveat, and don't be surprised if the 2020–2024 test period looks less dramatic than Barmer's historical reputation.

If you want the strongest, least-ambiguous drought signal and don't mind slightly more setup work: **Marathwada (Latur or Osmanabad/Dharashiv)** — it has both a stronger, more recent, less-complicated drought record than the Rajasthan options *and* a direct payoff for Phase 2's document corpus.

**Avoid Jaisalmer** for this specific purpose — its drought identity is the most actively undercut by current climate trends of the four.

## Sources

- [An analysis of rainfall variability and drought over Barmer District of Rajasthan](https://iwaponline.com/ws/article/21/5/2505/80559/An-analysis-of-rainfall-variability-and-drought)
- [Rajasthan Rainfall Shift: Rising Floods and Greener Landscapes in Thar Desert](https://www.downtoearth.org.in/climate-change/international-day-to-combat-desertification-and-drought-2025-indias-arid-landscape-now-receives-more-rains-and-floods)
- [Jaisalmer's Surprising Rainfall Surge: Climate Change Turns Desert Wet](https://www.downtoearth.org.in/climate-change/hard-to-imagine-jaisalmer-without-desert-climate-change-is-making-it-happen)
- [Latur Drinking Water Crisis — absence of Water Allocation Policy and Management (SANDRP)](https://sandrp.in/2016/04/20/latur-drinking-water-crisis-highlights-absence-of-water-allocation-policy-and-management/)
- [Parched: Drought and Climate Change in Marathwada, India](https://sriharsha.in/marathwada-drought/)
- [Statistical analysis of rainfall in Anantapur district of Andhra Pradesh](https://www.researchgate.net/publication/373862607_Statistical_analysis_of_rainfall_in_Anantapur_district_of_Andhra_Pradesh)
- [LULC Change Detection in Drought Prone Areas of Anantapur District](https://rsisinternational.org/journals/ijrsi/articles/lulc-change-detection-in-drought-prone-areas-of-anantapur-district-andhra-pradesh-using-rs-gis-technology/)
