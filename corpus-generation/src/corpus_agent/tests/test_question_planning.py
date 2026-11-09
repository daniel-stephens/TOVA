from corpus_agent.config import PlotPoint
from corpus_agent.phases.question_planning import (
    QuestionPlan,
    build_question_plan_prompt,
    validate_question_plan,
)


TOPICS = [
    PlotPoint(id="borrowed_melody", description="The score reuses a lullaby."),
    PlotPoint(id="shared_prop", description="A glass pear appears in both films."),
]


def _plan() -> QuestionPlan:
    return QuestionPlan.model_validate(
        {
            "corpus_id": "test",
            "questions": [
                {
                    "question": "Which documents carry the borrowed lullaby, and how does that topic meet the glass pear?",
                    "question_type": "topic_topic_connection",
                    "difficulty": 3,
                    "required_plot_points": ["borrowed_melody", "shared_prop"],
                    "answer_summary": "The lullaby and the pear are the same kept object.",
                    "evidence_slots": [
                        {
                            "slot_id": "q1-d1",
                            "topic_id": "borrowed_melody",
                            "writer_role": "composer",
                            "fact": "The feature score uses an old lullaby.",
                        },
                        {
                            "slot_id": "q1-d2",
                            "topic_id": "shared_prop",
                            "writer_role": "editor",
                            "fact": "A cracked glass pear sits on the table in the dailies.",
                        },
                    ],
                },
                {
                    "question": "Which document connects the lullaby topic to the score?",
                    "question_type": "document_topic_connection",
                    "difficulty": 2,
                    "required_plot_points": ["borrowed_melody"],
                    "answer_summary": "The composer memo carries the lullaby topic.",
                    "evidence_slots": [
                        {
                            "slot_id": "q2-d1",
                            "topic_id": "borrowed_melody",
                            "writer_role": "composer",
                            "fact": "Paul asks that the lullaby source stay unnamed.",
                        },
                        {
                            "slot_id": "q2-d2",
                            "topic_id": "borrowed_melody",
                            "writer_role": "editor",
                            "fact": "Lena finds the lullaby stem in the Night Window session.",
                        },
                    ],
                },
                {
                    "question": "How do the composer memo and the dailies note connect?",
                    "question_type": "document_document_connection",
                    "difficulty": 3,
                    "required_plot_points": ["borrowed_melody", "shared_prop"],
                    "answer_summary": "Together they show the kept object in sound and image.",
                    "evidence_slots": [
                        {
                            "slot_id": "q3-d1",
                            "topic_id": "borrowed_melody",
                            "writer_role": "composer",
                            "fact": "The cue is labeled only as 'orchard theme'.",
                        },
                        {
                            "slot_id": "q3-d2",
                            "topic_id": "shared_prop",
                            "writer_role": "editor",
                            "fact": "The pear is in frame during the orchard theme.",
                        },
                    ],
                },
            ],
        }
    )


def test_prompt_names_topics_before_documents_exist():
    prompt = build_question_plan_prompt([], TOPICS, "movie snippets")
    assert "before any document is written" in prompt
    assert "borrowed_melody" in prompt
    assert "document_topic_connection" in prompt


def test_plan_accepts_connection_questions():
    assert validate_question_plan(_plan(), {topic.id for topic in TOPICS}) == []


def test_plan_rejects_a_missing_connection_type():
    plan = _plan()
    plan.questions = plan.questions[:2]
    problems = validate_question_plan(plan, {topic.id for topic in TOPICS})
    assert any("document_document_connection" in problem for problem in problems)
