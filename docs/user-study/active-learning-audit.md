# Active learning page: audit and design directions

`templates/active_learning_two.html` — 3,179 lines.

## What's there

A two-column layout, equal 6/6 split:

- **Left, "context":** a queue with a search box, a three-way view toggle
  (Recommended / By model label / Similar), status and label filters, a legend,
  bulk-select, and confidence bars.
- **Right, "review":** document ID, a full-text/summary toggle, the document
  body, a metadata row (label + confidence), suggested-label chips, a free-text
  label input with a datalist, a save button, and Previous/Skip in the footer.
- Plus a preview modal, a completion banner, and a no-model empty state.

Keyboard shortcuts already exist: `1`–`9` pick a suggestion, `Enter` saves and
advances, `←`/`→` go back and skip, `/` focuses search.

**Credit where due:** the empty, complete and error states are all handled; the
confidence bars and the human-vs-model legend are a genuinely good idea; and
bulk-labelling is there. This is not a bad page. The problems below are about
where attention goes and what the code costs to change, not about missing
features.

## Findings

### 1. Structure, style and behaviour are fused into one file

1,266 lines of CSS, ~190 of markup, ~1,716 of JS across 127 functions — all
inline in a Django template. Consequences: nothing is cacheable across page
loads, the JS cannot be linted or unit-tested, and any change means editing a
3,000-line file where a CSS tweak and an event handler are 1,500 lines apart.

This is the root problem. The layout issues below are symptoms — they are hard
to fix precisely *because* the file resists change.

### 2. The queue has equal billing with the task

`col-xl-6 col-lg-6` on both columns. But the task is *read this document, give
it a label*; the queue is navigation. Giving navigation half the screen means
the document — the thing that needs sustained reading — gets a narrow column,
and the eye has two equally-weighted places to land on arrival.

### 3. A label costs two deliberate actions

Pick a suggestion (click or number key), then press Save. For a queue of
hundreds of documents that doubles the interaction count. `Enter` already does
both, but see below.

### 4. The shortcuts were mislabelled and hidden

Fixed in `48cf586`: the hint advertised `1`–`3` while the handler accepts `1`–`9`;
it carried `aria-hidden="true"` so screen-reader users were never told the
shortcuts existed; and each chip announced "of 3" regardless of how many
suggestions came back.

This matters more here than it would elsewhere. **Keyboard-vs-mouse is a
measured outcome of this study.** If the keyboard path is undiscoverable, the
study measures discoverability rather than preference — the instrument biases
the result it is trying to record. The hint also sits low on the page, below the
suggestions, where someone reading top-to-bottom meets it only after they have
already reached for the mouse.

### 5. Three view modes compete in one control

Recommended / By model label / Similar are three different orderings of the same
queue, presented as equals. "Recommended" is the one active learning exists to
provide; the other two are refinements. Presenting them as three peers asks the
user to make a choice before they have any basis for making it.

## Directions

### A. Extract, don't redesign

Move the CSS and JS to `static/`, keep the markup identical. No visible change,
but the page becomes cacheable, lintable and testable, and every later change
gets cheaper. **Lowest risk, no design commitment.** This is a prerequisite for
B or C regardless, and the honest first move.

### B. Review-first (prototyped)

Give the document the dominant column and demote the queue to a narrower rail.
Move the label action directly beneath the document body, so read → label is one
vertical path rather than a diagonal across two equal columns. Make `Enter`
alone complete the loop, and put the shortcut legend at the top of the action
area where it is read before the first label rather than after.

Costs one action per document instead of two, and makes the keyboard path the
obvious one — which is the right default for a labelling queue, and removes the
discoverability confound from the study's measurement.

### C. Single-column focus mode

Drop the queue entirely into a toggle; one document at a time, full width,
shortcuts only. Fastest possible loop, but loses the queue overview that makes
"By model label" and "Similar" useful, and bulk-labelling has nowhere to live.
Worth considering as a *mode* rather than the default.

## What was prototyped

**B**, at `templates/active_learning_three.html` with `static/js/al3.js` and
`static/css/al3.css`, behind `TOVA_ENABLE_AL3` (off by default), at
`/active-learning-3`. `active_learning_two.html` is untouched, so the two can be
compared side by side against the same model.

It talks to the existing endpoints — no backend change. It is a proposal in
runnable form, not a migration.

**It deliberately does not implement everything**: bulk-labelling, the Similar
view and the preview modal are absent. It exists to make the layout and the
one-keystroke loop concrete enough to judge, not to replace the page.

## Before switching the study to it

The interface *is* the independent variable in this study. Changing it mid-study
splits the data across two instruments, and the IRB application describes
evaluating "different possibilities for the way a particular set of
computational techniques can be deployed" — swapping the tool underneath
participants is a protocol question, not just an engineering one. If any
participant has run on `active_learning_two`, this needs the PI's sign-off
before it becomes the default.
