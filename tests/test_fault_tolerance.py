"""Tests verifying fault-tolerant hardening across Gemini call counter, SessionManager,
RedisLock, MuscleMapResolver, Worker conversation flow, and Telegram error alerts.
"""
import json
import httpx
import pytest

from app.clients.gemini import DailyCallCounter, GeminiClient, GeminiDailyCapExceeded
from app.memory.session import SessionManager
from app.metrics.muscle_mapping import muscle_groups_for
from app.metrics.muscle_resolver import MuscleMapResolver
from app.models.agent_io import (
    ConversationReply,
    ExerciseMuscleClassification,
    OrchestratorOutput,
)
from app.models.profile import AthleteProfile
from app.models.session import SessionState
from app.storage.redis import UpstashRedis, UpstashRedisError
from app.storage.redis_lock import RedisLock
from app.web import worker as worker_module
from tests.test_web_worker import (
    CURRENT_KEY,
    ENDPOINT_URL,
    NEXT_KEY,
    FakeExerciseHistory,
    FakeEventLog,
    FakeGeminiRunner,
    FakeProfileRepo,
    FakeStrava,
    FakeTelegram,
    NullCounter,
    _post,
    _sign,
)


def _exploding_redis(status_code: int = 403) -> UpstashRedis:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json={"error": "NOPERM this user has no permissions to run the 'incr' command"})

    return UpstashRedis(
        url="https://fake.upstash.io",
        token="read-only-tok",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


class DummyChoice(ExerciseMuscleClassification):
    pass


def test_daily_call_counter_fails_open_on_redis_http_error():
    redis = _exploding_redis(403)
    counter = DailyCallCounter(redis, cap=200)

    # Should not raise HTTPStatusError; returns 0 (fails open)
    count = counter.check_and_increment()
    assert count == 0


def test_gemini_client_proceeds_when_call_counter_redis_fails():
    class FakeModels:
        def generate_content(self, *, model, contents, config):
            class Resp:
                text = json.dumps({"primary": ["chest"], "secondary": ["triceps"], "confidence": 0.95})
            return Resp()

    class FakeRunner:
        models = FakeModels()

    redis = _exploding_redis(403)
    counter = DailyCallCounter(redis, cap=200)
    client = GeminiClient(FakeRunner(), counter)

    result = client.generate_structured(
        model="gemini-3.5-flash",
        prompt="classify",
        response_schema=ExerciseMuscleClassification,
    )
    assert result.primary == ["chest"]


def test_session_manager_does_not_raise_on_redis_failure():
    redis = _exploding_redis(403)
    sm = SessionManager(redis, ttl_seconds=1800)

    # load fails gracefully
    assert sm.load(123) is None

    # save fails gracefully
    state = SessionState(chat_id=123, expires_at=1000)
    sm.save(state)  # does not raise

    # forget fails gracefully
    sm.forget(123)  # does not raise


def test_redis_lock_yields_cleanly_on_redis_error():
    redis = _exploding_redis(403)
    lock = RedisLock(redis, "test_lock", ttl_seconds=5, max_attempts=2, retry_delay_s=0.01)

    # Should acquire (proceed) without throwing, and exit cleanly
    with lock:
        pass


def test_muscle_resolver_degrades_cleanly_on_transport_or_api_error():
    class FailingGemini:
        def generate_structured(self, **kwargs):
            req = httpx.Request("POST", "https://api.fake.com")
            raise httpx.HTTPStatusError("500 Internal Error", request=req, response=httpx.Response(500, request=req))

    class EmptyRepo:
        def get(self, key):
            return None
        def save(self, *args, **kwargs):
            pass

    resolver = MuscleMapResolver(EmptyRepo(), FailingGemini(), "fake-model")
    weights, found = resolver("Some Completely Unknown Exercise 999")
    assert found is False
    assert weights == {}


def test_handle_conversation_turn_dispatches_telegram_before_session_storage():
    events = []

    class SpyingTelegram(FakeTelegram):
        def send_markdown(self, chat_id, markdown, reply_markup=None):
            events.append(("telegram", markdown))
            return super().send_markdown(chat_id, markdown, reply_markup)

    class FailingSessionManager(SessionManager):
        def __init__(self):
            pass
        def get_or_create(self, chat_id, fallback_focus=None):
            return SessionState(chat_id=chat_id, expires_at=9999999999), False
        def save(self, state):
            events.append(("session_save", state))
            raise RuntimeError("Redis write failed on session save")

    runner = FakeGeminiRunner({
        OrchestratorOutput: OrchestratorOutput(
            intent="greeting",
            target_entity=None,
            needs_data=["none"],
            needs_chart=False,
            resolved_pronouns=[],
            route_to="conversation",
        ),
        ConversationReply: ConversationReply(reply_markdown="Hello there!"),
    })
    gemini = GeminiClient(runner, NullCounter())
    telegram = SpyingTelegram()

    deps = worker_module.WorkerDeps(
        owner_chat_id=111,
        training_goal_default="Hybrid",
        strava=FakeStrava({}),
        telegram=telegram,
        orchestrator=worker_module.OrchestratorAgent(gemini, model="m"),
        strength_analyst=worker_module.StrengthAnalystAgent(gemini, model="m"),
        endurance_analyst=worker_module.EnduranceAnalystAgent(gemini, model="m"),
        chart_agent=worker_module.ChartAgent(gemini, model="m"),
        conversation=worker_module.ConversationAgent(gemini, model="m"),
        session_manager=FailingSessionManager(),
        profile_repo=FakeProfileRepo(AthleteProfile()),
        exercise_history=FakeExerciseHistory(),
        event_log=FakeEventLog(),
    )

    worker_module._handle_conversation_turn("hello", 111, deps)

    # Check order: telegram sent first, then session save attempted
    assert len(events) == 2
    assert events[0][0] == "telegram"
    assert events[0][1] == "Hello there!"
    assert events[1][0] == "session_save"


def test_worker_notifies_telegram_user_on_unexpected_update_error():
    telegram = FakeTelegram()

    runner = FakeGeminiRunner({})
    gemini = GeminiClient(runner, NullCounter())

    class ExplodingOrchestrator(worker_module.OrchestratorAgent):
        def route(self, session_context, user_message):
            raise RuntimeError("Unexpected failure during routing")

    deps = worker_module.WorkerDeps(
        owner_chat_id=111,
        training_goal_default="Hybrid",
        strava=FakeStrava({}),
        telegram=telegram,
        orchestrator=ExplodingOrchestrator(gemini, model="m"),
        strength_analyst=worker_module.StrengthAnalystAgent(gemini, model="m"),
        endurance_analyst=worker_module.EnduranceAnalystAgent(gemini, model="m"),
        chart_agent=worker_module.ChartAgent(gemini, model="m"),
        conversation=worker_module.ConversationAgent(gemini, model="m"),
        session_manager=SessionManager(_exploding_redis(403), ttl_seconds=1800),
        profile_repo=FakeProfileRepo(AthleteProfile()),
        exercise_history=FakeExerciseHistory(),
        event_log=FakeEventLog(),
    )

    payload = {
        "kind": "telegram_update",
        "update": {
            "update_id": 1,
            "message": {
                "message_id": 1,
                "chat": {"id": 111},
                "text": "Tell me something",
            },
        },
        "event_hash": "hash-err",
    }
    resp = _post(payload, deps)
    assert resp.status == 200
    assert deps.event_log.failed == ["hash-err"]

    # Verify user received error notification on Telegram
    assert len(telegram.sent_markdown) == 1
    chat_id, text, _ = telegram.sent_markdown[0]
    assert chat_id == 111
    assert "error occurred while processing your request" in text


def test_new_dumbbell_muscle_mappings():
    # Incline Dumbbell Press
    w_inc, found_inc = muscle_groups_for("Incline Dumbbell Press")
    assert found_inc is True
    assert w_inc["chest"] == 1.0
    assert w_inc["shoulders"] == 0.5
    assert w_inc["triceps"] == 0.5

    # Dumbbell Bench Press
    w_db_bench, found_db_bench = muscle_groups_for("Dumbbell Bench Press")
    assert found_db_bench is True
    assert w_db_bench["chest"] == 1.0
    assert w_db_bench["shoulders"] == 0.5
    assert w_db_bench["triceps"] == 0.5

    # Dumbbell Shoulder Press
    w_db_sp, found_db_sp = muscle_groups_for("Dumbbell Shoulder Press")
    assert found_db_sp is True
    assert w_db_sp["shoulders"] == 1.0
    assert w_db_sp["triceps"] == 0.5
