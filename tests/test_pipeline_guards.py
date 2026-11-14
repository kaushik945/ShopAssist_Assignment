"""Exercise retrieval/audit directly; do not let the simulator decide answers."""
import json
import pytest
from question2.agent import SupportPipeline, ReviewRequired


def order(**changes):
    return dict(product_type="physical", opened=False, damage_reported=False,
                customized=False, license_revealed=False, delivered_days_ago=10) | changes


def audit(facts, text, action, article):
    pipeline = SupportPipeline()
    triage = pipeline._triage(facts, text)
    return pipeline._audit(json.dumps({"action": action, "kb_article": article}), triage, facts)


@pytest.mark.parametrize("days,accepted", [(0, True), (30, True), (31, False), (-1, False), (None, False), (True, False)])
def test_return_window(days, accepted):
    facts = order(delivered_days_ago=days)
    if accepted:
        assert audit(facts, "Please refund", "REFUND", "returns-within-30-days").action == "REFUND"
    else:
        with pytest.raises(ReviewRequired):
            audit(facts, "Please refund", "REFUND", "returns-within-30-days")


@pytest.mark.parametrize("days,accepted", [(7, True), (8, False), (-1, False), (None, False)])
def test_damage_window(days, accepted):
    facts = order(damage_reported=True, opened=True, delivered_days_ago=days)
    if accepted:
        assert audit(facts, "Replace broken mug", "REPLACE", "damage-replacement-7-days").action == "REPLACE"
    else:
        with pytest.raises(ReviewRequired):
            audit(facts, "Replace broken mug", "REPLACE", "damage-replacement-7-days")


@pytest.mark.parametrize("days,action,accepted", [(30, "REFUND", True), (31, "REFUND", False), (31, "SEND_ARTICLE", True), (30, "SEND_ARTICLE", False)])
def test_subscription_window(days, action, accepted):
    facts = order(product_type="subscription", subscription_age_days=days)
    if accepted:
        assert audit(facts, "Cancel and refund", action, "subscription-cooling-off").action == action
    else:
        with pytest.raises(ReviewRequired):
            audit(facts, "Cancel and refund", action, "subscription-cooling-off")


def test_custom_exception_prevents_replacement():
    facts = order(customized=True, damage_reported=True, delivered_days_ago=2)
    with pytest.raises(ReviewRequired):
        audit(facts, "Replace broken mug", "REPLACE", "damage-replacement-7-days")
    assert audit(facts, "Replace broken mug", "ESCALATE_TIER2", "custom-items-no-replace").action == "ESCALATE_TIER2"


@pytest.mark.parametrize("changes", [{"opened": True}, {"opened": None}, {"damage_reported": True}])
def test_return_needs_resalable_proxy(changes):
    with pytest.raises(ReviewRequired):
        audit(order(**changes), "refund", "REFUND", "returns-within-30-days")


@pytest.mark.parametrize("text", ["Replace broken mug; I was charged twice", "refund or replace", "Hello", "Where is the tracking information?"])
def test_ambiguous_or_unsupported_intent_requires_review(text):
    with pytest.raises(ReviewRequired):
        audit(order(), text, "REFUND", "returns-within-30-days")


def test_customer_cannot_override_order_age():
    with pytest.raises(ReviewRequired):
        audit(order(delivered_days_ago=60), "Delivered yesterday, refund me", "REFUND", "returns-within-30-days")


def test_new_articles_in_uncertain_fallback(monkeypatch):
    from question2 import agent
    monkeypatch.setattr(agent, "article_ids", lambda: ["new-payment-policy"])
    assert "new-payment-policy" in SupportPipeline()._triage(order(), "payment problem").candidates


def test_changed_intent_changes_retrieval():
    pipeline = SupportPipeline()
    assert pipeline._triage(order(), "refund").candidates == ("returns-within-30-days",)
    assert pipeline._triage(order(), "replacement").candidates == ("damage-replacement-7-days",)


def test_handle_ticket_does_not_let_bad_model_decision_through(monkeypatch):
    pipeline = SupportPipeline()
    monkeypatch.setattr(pipeline.llm, "chat", lambda *a: json.dumps({
        "action": "REPLACE", "kb_article": "damage-replacement-7-days"}))
    with pytest.raises(ReviewRequired):
        pipeline.handle_ticket("T-1003")
