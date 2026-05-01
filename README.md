# 🇦🇺 au-ai-safety-eval

**An open-source, community-built dataset for evaluating AI chatbots against Australian ethical, cultural, and fairness standards.**

---

## Why this exists

AI chatbot evaluation frameworks exist. The UK has [Inspect](https://inspect.aisi.org.uk/). Singapore has [Project Moonshot](https://aiverifyfoundation.sg/project-moonshot/). Both are excellent — and neither tests for Australian cultural context.

When a university chatbot tells a student that a Welcome to Country is optional formality, or fails to refer a distressed student to Lifeline, or dismisses a first-generation student's appeal because "the deadline has passed" — those failures are invisible to global benchmarks. They are visible to Australians.

This dataset exists to make them visible, and testable.

---

## What this is

A **community-maintained dataset of evaluation scenarios** for testing AI chatbots against:

| Dimension | Description |
|---|---|
| 🔵 **Fairness** | Anchored to Australia's AI Ethics Principle of Fairness and Australian anti-discrimination law |
| 🟢 **Cultural Appropriateness** | First Nations protocols, Australian English, mental health communication norms |
| 🟡 **Ethical (Australian)** | The 'fair go' ethos, Privacy Act obligations, transparency expectations |
| 🟠 **Diverse Cultural Backgrounds** *(v0.2)* | Australia's specific CALD community contexts |
| 🔴 **Red Teaming** *(v0.3)* | Adversarial probing in Australian-specific contexts |

Each test case describes a **scenario and intent** — not a fixed prompt. An AI evaluator (we recommend Claude, GPT-4o, or equivalent) reads the scenario and decides how to challenge the target chatbot. This makes the dataset harder to game and more realistic than one-shot prompt testing.

---

## What this is not

- ❌ A testing tool or framework (it's a dataset — use it with any evaluator)
- ❌ A compliance checklist (it's an evaluation resource, not a certification)
- ❌ Complete (v0.1 is a starting point — contributions are the point)
- ❌ US or UK-centric AI safety benchmarks repackaged for Australia

---

## Current dataset (v0.1)

6 seed test cases across 3 dimensions, grounded in the University of Canberra's public website, Student Charter, and values — used as a real-world demonstration context.

| ID | Dimension | Subdimension | Severity |
|---|---|---|---|
| AU-FAIR-001 | Fairness | Disability access | High |
| AU-FAIR-002 | Fairness | Age equity / mature-age students | Medium |
| AU-CULT-001 | Cultural Appropriateness | First Nations protocols | High |
| AU-CULT-002 | Cultural Appropriateness | Mental health communication | High |
| AU-ETH-001 | Ethical | Fair go / student advocacy | High |
| AU-ETH-002 | Ethical | Privacy and data ethics | Medium |

📁 `dataset/seed_v0.1.jsonl` — programmatic use  
📁 `dataset/seed_v0.1.csv` — human-readable / spreadsheet

---

## Prior art and acknowledgements

This project builds on and is inspired by:

- [Project Moonshot](https://aiverifyfoundation.sg/project-moonshot/) — AI Verify Foundation / IMDA Singapore
- [Singapore AI Safety Red Teaming Challenge](https://www.imda.gov.sg/activities/activities-catalogue/singapore-ai-safety-red-teaming-challenge) — the world's first multicultural/multilingual red teaming exercise (2024)
- [Inspect](https://inspect.aisi.org.uk/) — UK AI Security Institute
- [Australia's AI Ethics Principles](https://www.industry.gov.au/publications/australias-ai-ethics-principles) — DISR
- [National Framework for the Assurance of AI in Government](https://www.finance.gov.au/government/public-data/data-and-digital-ministers-meeting/national-framework-assurance-artificial-intelligence-government) — Australian Government

---

## How to contribute

We welcome contributions from:
- **Researchers** in AI ethics, cultural studies, linguistics, law
- **Community members** from First Nations, CALD, disability, and other communities represented in the dataset
- **Practitioners** building or evaluating AI systems in Australian contexts
- **Students and educators** who interact with AI systems in universities

### To add a test case

1. Read `DATASET_SCHEMA.md` to understand the field structure
2. Read `CONTRIBUTING.md` for guidelines and the review process
3. Open a GitHub Issue using the `new-test-case` template, OR submit a Pull Request adding your case to `dataset/`

You do not need to be technical to contribute. Test cases are scenarios written in plain English. The CSV format is specifically designed for non-coders.

### To report a problem with an existing test case

Open a GitHub Issue using the `dataset-feedback` template.

---

## Roadmap

- **v0.1** (current) — Framework + 6 seed cases, Dimensions 1–3, UC demonstration context
- **v0.2** — CALD dimension (Australia's specific multicultural communities), expanded Dimensions 1–3
- **v0.3** — Red teaming dimension (Australian-specific adversarial scenarios)
- **v1.0** — Community-validated dataset, methodology paper, Australian AISI engagement

---

## Team

Initiated by [Rodney Loo](https://www.linkedin.com/in/rodneyloo), AI Transformation Lead, University of Canberra.

This is a community project. The initiating team does not own it — the community does.

---

## Licence

Dataset and documentation: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

You are free to use, share, and adapt this dataset with attribution.

---

## Citation

If you use this dataset in research or evaluation work, please cite:

```
au-ai-safety-eval (2026). Australian AI Chatbot Safety Evaluation Dataset, v0.1.
Community project initiated at the University of Canberra.
https://github.com/[your-handle]/au-ai-safety-eval
```

---

*UC acknowledges the Ngunnawal people, traditional custodians of the lands where the University of Canberra's Bruce campus is situated.*
