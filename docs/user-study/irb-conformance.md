# IRB conformance: protocol → code

**Study:** TOVA CorpusBio · **IRB:** UMD package 1936243-1 (exempt path)
**PI:** Jordan Boyd-Graber · **Co-I:** Zongxia Li
**Sources:** "UMCP - IRB Initial Application - Part 1.pdf" (last edited 2026-08-10),
"TOVA CorpusBio - consent document revised.docx"

Where each clause of the approved protocol is enforced in the software. Written
so it can be handed to the PI, or to the IRB if they ask how the tool implements
the protocol.

> **Caveat on the source documents.** The PDF supplied is the *application*, not
> an approval letter, and it lists a consent waiver request as still-outstanding
> documentation; page 5 states recruitment cannot begin before final written
> approval. The consent file is also marked "revised". The build stamps
> `CONSENT_VERSION = "corpusbio-2026-08-30-revised"` onto every participant, so
> if the approved wording differs, the constant and the template change together
> and the export records exactly who saw which version.

## Consent

| Protocol clause | Where | Status |
| --- | --- | --- |
| "Implied consent by clicking to proceed" | `templates/study/consent.html`, `web/views.py:study_consent_accept` | Implemented. The click is the consent; no signed form. |
| Approved sign-up text shown before the task | `templates/study/consent.html` | Verbatim. `ConsentTests.test_renders_the_approved_text_verbatim` asserts all six approved paragraphs word for word against the rendered page, so a later paraphrase fails the build. |
| 18+, in and participating from the US | Attestation sentence immediately above the proceed button | Implemented, wording verbatim. |
| Consent precedes everything | `web/study_flow.py` `STAGE_CONSENT` heads `STAGE_ORDER`; `StudyFlowMiddleware` | Implemented. Nothing but the consent page is reachable first — a participant cannot even start the task timer. |
| "You may download or print a copy of this page" | Print affordance + `@media print` styles on the consent page | Implemented. |
| Investigator contact | Consent page, verbatim (name, address, email, phone) | Implemented. |
| IRB package number and contact | Consent page footer | Implemented. |
| Declining is possible | `web/views.py:study_consent_declined`, `templates/study/declined.html` | Implemented. No session is started; no "are you sure" friction. |
| Record of consent | `ParticipantSession.consented_at`, `.consent_version` | Implemented, idempotent — a double-submit cannot rewrite when consent happened. |

## Procedure

| Protocol clause | Where | Status |
| --- | --- | --- |
| ~600 synthetic emails, one assigned interface | Dashboard assignment per participant (`User.study_model_id`) | Implemented. **Verify the assigned dashboards point at the CorpusBio corpus** on the new pre-flight page. |
| "During a fixed 30-minute exploration period" | `STUDY_TASK_MINUTES` (default 30), enforced server-side in `StudyFlowMiddleware` | Implemented. **Open question — see below re: Finish Early.** |
| "You will be told the general kinds of questions … but not the exact questions" | `templates/instructions.html` | Implemented; four category cards, no actual questions. |
| "the tool will close" | Stage gating; the task routes become unreachable at `post_study` | Implemented, server-side. Killing JavaScript does not buy extra time. |
| "four true/false/not-sure questions and four written questions from memory" | `web/instruments.py` | Implemented. |
| "You may also select the email IDs that support your written answers" | `web/views.py:study_docs`, citation picker | Implemented. |
| "You will not be able to reopen the emails during the questionnaire" | `_study_base_documents` returns only `id`, `display_id`, `subject`, `date` — never body text | Implemented. The instructions copy previously contradicted this and has been corrected. **Open question — see below.** |
| "You may skip a question" | `web/views.py:study_post` | Implemented. Blank fields are recorded, not refused; an explicit "Prefer not to answer" is stored as `declined`, distinct from `unsure`. **This previously did not hold** — the form returned 400 on any blank. |
| Moderated one-on-one arm, researcher notes | `ParticipantSession.moderated`, `ResearcherNote` | Implemented. Notes are keyed to the participant's study code. |
| "No audio, video, screenshots, or screen recording" | Nothing in the application captures any | Holds. Enforced by researcher discipline during moderated sessions, not by code. |
| "approximately 30 to 60 minutes" total | 30 min task + questionnaire | Consistent. |
| Payment per the recruitment posting | `templates/study/done.html`; `PROLIFIC_COMPLETION_CODE` / `PROLIFIC_RETURN_URL` | Implemented. The page previously dead-ended, leaving no way to claim payment. |

## Data

| Protocol clause | Where | Status |
| --- | --- | --- |
| "Data will be linked only to a study-specific code" | `User.participant_code`, `User.export_id` | Implemented. Both CSV exports and the keyboard-vs-mouse rollup key on the code; no export emits an email address. Tests assert no `@` reaches the files. |
| Recruitment-platform account info not in the analytic dataset | Prolific PID never enters the database — participants log in as generated `p001`-style accounts | Implemented by construction. The PID→code mapping must be held outside the app. |
| "Please do not enter or say identifying information" | Notice beside the free-text boxes on the questionnaire | Implemented. |
| Records: questionnaire responses | `StudyResponse` + CSV export | Implemented. |
| Records: supporting email IDs | `StudyResponse.open_answers[*].citations` | Implemented. |
| Records: themes/topics/labels displayed or edited | `TopicRename`, active-learning labels | Implemented. |
| Records: AI interactions | `ChatMessage` | Implemented. |
| Records: application logs (time, emails opened, queries, retrieval results, label edits) | `InteractionEvent` + CSV export | Implemented. Captures no typed content: printable characters are recorded as the token `"char"`. |
| Records: moderated-session notes | `ResearcherNote` | Implemented. |
| "reviewed and anonymized before any public release" | Exports are already de-identified | Partly. Review before release remains a human step. |
| Content warning (misconduct, data manipulation, gene-therapy safety) | Consent page, verbatim, plus a surfaced content note | Implemented. |

## Open questions for the PI

These are judgement calls about the protocol, not software defects. Left as they
are rather than guessed at.

1. **"Fixed 30-minute exploration period" vs. the Finish Early button.** The
   button lets a participant end the task and move to the questionnaire early,
   which makes exposure time vary across participants. The consent's "you may
   stop participating at any time" covers withdrawal; it does not obviously
   describe curtailing the task and continuing. Unchanged pending a decision.
2. **Does theme-grouped subject-line browsing count as "from memory"?** The
   citation picker shows document IDs, subject lines and dates grouped by theme,
   but no email text. Defensible as a citation aid; it is the PI's call.
3. **Which consent version is approved**, and is there a final approval letter.
   See the caveat at the top.

## Outside the software

Recruitment posting and payment terms; the PID→code mapping procedure; data
retention and destruction; the no-recording discipline during moderated
sessions; review of the dataset before public release.
