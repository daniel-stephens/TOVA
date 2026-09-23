# Researcher key for the proposed post-study questionnaire

**Draft instrument:** `corpusbio-v2-draft`

**Participant wording:** [post-study-questionnaire-v2-draft.md](post-study-questionnaire-v2-draft.md)

This is a proposed coding guide, not a validated questionnaire or a result from
participant testing. The questions and key were checked against email text,
not the model's generated labels or summaries.

## Why replace the current content

The running questionnaire asks about Marcos, Priya, the Gene Integrity Detail
investigation, and an unsanctioned sample audit. None of those terms occurs in
the loaded 1,151-record collection. That collection concerns Xylos Genomics,
the Bio-Vault, the audit and funding pressure, blueprint extraction, and
biological instability. The proposed revision assesses those events directly.

The revised questions also avoid requiring an exact time gap across a corpus
with repeated and varied email versions. Written answers can demonstrate
reasoning without depending on recall of a particular timestamp or email ID.

## Source and reference convention

- Source file: `data/email_corpus (1).jsonl`.
- Source SHA-256: `31930ea501c0f377e91a4a72d4cf674f8ec41500ea97670d8d7a8f98fa4109a9`.
- Reviewed normalized file:
  `data/study-runs/corpusbio-study-20260923-015336-77e3/documents.json`.
- References below use original email IDs from the first 30 source records,
  with subject lines for disambiguation. These are not the generated `doc_…`
  database IDs. The first 30 normalized entries are `source_000001` through
  `source_000030`.
- The collection contains repeated IDs and parallel versions of the story.
  Accept another email that provides equivalent evidence; do not require
  participants to cite these specific representatives. Record the unique
  stored document ID for any citation selected in the application.

## Factual answer key — 0–4 correct

| Field | Answer | Evidence and interpretation |
| --- | --- | --- |
| `tf1` | True | Email 2, **RE: Funding Tranche Conditionality**, explicitly makes the second funding installment conditional on the audit and a signed stability report. Email 15, **Audit Finalization**, repeats the condition. |
| `tf2` | False | Email 25, **USELESS DATA**, contains Victor's complaint that the transferred sequences are fragmented, corrupted, and unusable. Kaelen's later claim in email 27 that his expertise could make them usable does not change what Victor reported. |
| `tf3` | True | Email 26, **The Only Solution**, proposes a total purge, destruction of the current assets, and a restart from the original seeds. The question concerns a proposal; it does not claim the purge was carried out. |
| `tf4` | False | Email 20, **Observations on Drift**, describes a rate of change beyond Vance's predictive models, with uncontrolled drift following the outage. |

Award one point per correct statement. Store incorrect answers, **Not sure**,
**Prefer not to answer**, and blank responses distinctly. Report the number
answered with the score so skipping is visible. Do not convert **Not sure** into
a factual answer.

The answer pattern happens to match the old key (true, false, true, false).
That does **not** make the instruments equivalent: their wording and subject
matter differ, so the revised instrument needs a new version.

## Written-answer coding guide — 0–3 per question

Each listed criterion is worth one point. Judge the substance of an answer,
not grammar, length, exact character spelling, or ability to quote an email.
Use `missing` for blank/declined responses; use `unsure` when the participant
explicitly cannot remember. An attempted answer that meets no criteria can
receive zero. Keep these response statuses alongside any score.

Citations are optional. Concrete evidence described in the response can earn
credit without an email ID. A citation by itself does not substitute for an
explanation. Equivalent evidence in another email version is acceptable.

### Q1 / `motive`

1. Identifies personal financial gain or bargaining for compensation as a
   supported motive for Kaelen's dealings with Victor.
2. Describes one supporting detail, such as an expected consulting fee, the
   offshore payment arrangement, or insistence on full payment for extraction.
3. Describes a second distinct supporting detail and connects it to the motive,
   such as the refused equity increase or the later demand for a larger fee.

**Representative evidence:** 8, **Technical Consulting**; 9, **Your
Contributions**; 13, **Closing the Deal**; 14, **The Window**; 27, **The Value
of Instability**.

Recognition or dissatisfaction may be defensible additional motives. Do not
require or reward an unsupported claim that the rejected equity request alone
proved why Kaelen acted. Seeking a benefit is distinct from ultimately
receiving it.

### Q2 / `connections`

1. Accurately distinguishes Gideon's account of a breach/security or containment
   failure from Cassian's account of a temporary outage or technical glitch.
2. Brings in relevant independent evidence: for example, Kaelen's planned
   blackout, his message that transfer had begun, or the later log comparison.
3. Explains why that evidence supports deliberate interference over the
   initial glitch explanation, while avoiding unsupported certainty about
   Cassian's knowledge or intent.

**Representative evidence:** 16, **CRITICAL: BIO-VAULT OFFLINE**; 17, **System
Issue**; 14, **The Window**; 18, **Curtains**; 22, **Patterns in the Crash**;
24, **SERVER ROOM NOW**.

Do not infer Cassian's complicity merely because he initially called the event
a glitch. The question tests comparing claims and evaluating evidence.

### Q3 / `timeline`

1. Identifies three events supported by the emails, including an audit-period
   event and an event associated with the company's collapse.
2. Places the events in a defensible sequence. A suitable example is audit and
   funding pressure → blackout/extraction and loss of control → investor
   withdrawal and bankruptcy proceedings.
3. Explains one supported connection between events, such as the blackout
   enabling extraction or asset loss contributing to investor withdrawal.

**Representative evidence:** 2, **RE: Funding Tranche Conditionality**; 7,
**URGENT: System Stability Report**; 14, **The Window**; 16, **CRITICAL:
BIO-VAULT OFFLINE**; 18, **Curtains**; 20, **Observations on Drift**; 28,
**Notice of Liquidation**.

Accept other supported sequences. Do not score exact dates, a precise duration,
or a claim that the theft was the sole cause of every subsequent problem.

### Q4 / `threads`

1. Explains blueprint theft as unauthorized extraction or transfer of the
   company's information/architecture.
2. Explains biological instability as changes in the organisms or genomic
   sequences and failure to keep those changes under control.
3. Gives evidence for both concerns and treats them as distinct findings,
   even if connected in the story.

**Representative evidence:** theft — 13, **Closing the Deal**; 14, **The
Window**; 18, **Curtains**. Instability — 19, **Umm... look at this lol**;
20, **Observations on Drift**; 21, **URGENT: Sequence Instability**; 26,
**The Only Solution**. Email 25, **USELESS DATA**, connects the problems through
Victor's report of the transferred material's condition.

Do not require a scientific diagnosis absent from the emails. An email author's
claim, an observed event, and a participant's inference should remain distinct.

## Interface feedback

These are custom diagnostic items. Do not label them SUS, NASA-TLX, a validated
trust measure, or a validated overall usability scale.

| Field | Intended construct | Scoring/interpretation |
| --- | --- | --- |
| `find_emails` | Perceived ease of finding relevant information | Five ordered responses, difficult to easy. |
| `theme_names` | Perceived fit between names and grouped emails | Five ordered responses, poorly to well. |
| `theme_summaries` | Perceived faithfulness of summaries to emails read | Five ordered responses, inaccurate to accurate. This is not an objective factuality audit. |
| `label_control` | Perceived ease of assigning/changing document labels | Five ordered responses, difficult to easy. Feature non-use is a separate response. |
| `mental_effort` | Effort required by the task as experienced | Five ordered responses, little to much. Higher values mean more effort; this does not isolate effort caused by the interface. |
| `story_confidence` | Self-reported confidence in story understanding | Five ordered responses, not at all to extremely. Keep separate from actual answer accuracy. |

For storage, encode the five substantive choices consistently as 1–5 in the
displayed order. Store feature non-use as `not_used`, inability to judge as
`cannot_judge`, and blanks as missing. Do not encode these as zero or as the
midpoint. Report each item separately with its answered and feature-use counts;
do not sum them into one score.

The two feedback prompts can be coded for features mentioned, helpful actions,
points of confusion, and requested changes. Preserve negative and mixed
experiences as well as positive ones.

## Review before applying

Pilot the wording for comprehension and scoring consistency. For written
answers, have researchers independently code the same small set of responses,
compare disagreements, and refine the criteria before the main collection.
Compare conditions using the same instrument version and task exposure.

The existing study description names four factual and four written questions.
The six ratings and two feedback prompts expand that instrument. Applying this
draft therefore includes updating the participant-facing description and
recording a new instrument version; the draft makes no claim about approval of
that change.
