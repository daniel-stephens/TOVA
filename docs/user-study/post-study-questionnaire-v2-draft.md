# Post-study questionnaire — proposed revision

**Draft:** `corpusbio-v2-draft`

**Collection:** Xylos Genomics / Bio-Vault, from `data/email_corpus (1).jsonl`

**Scope:** four written story questions, four factual statements, six interface
ratings, and two optional feedback prompts. This document is a content draft;
it has not replaced the running questionnaire.

The researcher answer key and coding guidance are in
[post-study-questionnaire-v2-key.md](post-study-questionnaire-v2-key.md).
Do not include that key in the participant form.

---

## Participant introduction

Thank you for exploring the emails. First, we will ask about your understanding
of the story. Then, we will ask about your experience with the interface.

Answer the story questions from memory, based on the emails you read. You do not
need to remember exact dates or quote the emails. You may select supporting
email IDs from the list of IDs and subject lines; the email bodies cannot be
reopened here.

All questions are optional. If you cannot remember something, you can say so or
choose **Not sure**. There are no right or wrong answers to the interface
questions.

Please do not include real people's personal or contact information. You may
refer to the fictional characters in the email collection by name.

## Part 1 — Explain what you found

Two to four sentences per answer are enough. Supporting email IDs are optional.

### 1. Motives — `motive`

What do Kaelen's messages suggest he hoped to gain from his dealings with
Victor? Describe two details from the emails that support your interpretation.

*Your answer:* [text box]

*Supporting email IDs, if remembered:* [citation picker]

### 2. Connecting accounts — `connections`

How did Gideon and Cassian explain the Bio-Vault outage? Which other emails
help you assess their explanations?

*Your answer:* [text box]

*Supporting email IDs, if remembered:* [citation picker]

### 3. Sequence of events — `timeline`

Describe three key events leading from the audit period to Xylos's collapse,
in the order they happened. Explain how one event contributed to the next.
Exact dates are not needed.

*Your answer:* [text box]

*Supporting email IDs, if remembered:* [citation picker]

### 4. Distinguishing problems — `threads`

How did the problem of blueprint theft differ from the problem of biological
instability? Describe evidence for each problem.

*Your answer:* [text box]

*Supporting email IDs, if remembered:* [citation picker]

## Part 2 — Check your understanding

For each statement, choose **True**, **False**, **Not sure**, or **Prefer not to
answer**. You may also leave it blank. Base your answer on what the emails said.

5. **`tf1`** — The next installment of investor funding depended on successful
   completion of the Bio-Vault audit.
6. **`tf2`** — Victor reported that the transferred blueprints were ready to use
   without further work.
7. **`tf3`** — Soren proposed destroying the current biological assets and
   starting again from the original material.
8. **`tf4`** — Marcus reported that, after the outage, the organisms' rate of
   change remained within Vance's predictions.

## Part 3 — Your experience with the interface

Think about the interface you just used. Choose the response that best describes
your experience. Leave any question blank if you prefer not to answer. Use the
extra options when you did not use a feature or cannot judge it.

| ID | Question | Response options |
| --- | --- | --- |
| `find_emails` | How easy or difficult was it to find emails relevant to a question you were investigating? | Very difficult / Difficult / Neither easy nor difficult / Easy / Very easy / Did not try |
| `theme_names` | How well did the theme names describe the emails grouped under them? | Very poorly / Poorly / Neither poorly nor well / Well / Very well / Did not use theme names / Cannot judge |
| `theme_summaries` | How accurately did the theme summaries reflect the emails you read? | Very inaccurately / Inaccurately / Neither accurately nor inaccurately / Accurately / Very accurately / Did not read summaries / Cannot judge |
| `label_control` | How easy or difficult was it to assign or change a document's label? | Very difficult / Difficult / Neither easy nor difficult / Easy / Very easy / Did not assign or change labels |
| `mental_effort` | How much mental effort did exploring the emails require? | Very little / Little / Moderate / Much / Very much / Cannot judge |
| `story_confidence` | How confident are you in your understanding of what happened at Xylos? | Not at all confident / Slightly confident / Moderately confident / Very confident / Extremely confident |

### 15. What helped — `helpful_example`

Which part of the interface, if any, helped you understand the emails? Give one
example of how you used it.

*Your answer:* [optional text box]

### 16. What to improve — `improvement_example`

What is one change that would have made this task easier? If possible, describe
a moment when that change would have helped.

*Your answer:* [optional text box]

**Submit answers** → thank-you / study completion page.

---

## Notes for implementing this draft

- Present the written questions before the factual statements to reduce the
  chance that the statements supply an answer to a written question. Keeping
  all items on one page does not eliminate this cueing; do not treat these as
  independent, unaided recall measures.
- Use the same core questions and scales across interface conditions. Keep
  feature-use responses separate from neutral ratings and skipped questions.
- Do not preselect responses. Preserve optional questions and citations.
- Keep the answer key on the server. Show no correctness feedback before the
  questionnaire is complete.
- Introduce a new instrument version when applying the revision. Retain the
  old version and scores for prior responses, and ensure an already open form
  is recorded against the wording it displayed.
- Update the participant study description when adding the interface section:
  the current consent describes only four factual and four written questions.
  This draft does not alter that consent text or claim approval of the expanded
  instrument.
- Export the six ratings and two feedback responses as separate fields. Keep
  story accuracy, self-reported confidence, effort, and interface ratings
  separate in analysis.
