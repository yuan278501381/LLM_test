# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""教学证据、结论边界与课程注册表回归测试。"""

import ast
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from dashboard.components.pedagogy import evidence_badge, get_result_claim
from dashboard.constants.course import (
    CLAIMS,
    CURRICULUM_DAG,
    EVIDENCE_DESCRIPTIONS,
    FORMATIVE_QUIZZES,
    LEARNING_LOOPS,
    LESSONS,
    ClaimKind,
    EvidenceLevel,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAGES_DIR = PROJECT_ROOT / "dashboard" / "pages"


def _declared_string_args(page: Path, function_name: str) -> list[str]:
    tree = ast.parse(page.read_text(encoding="utf-8"))
    declared = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != function_name or not node.args:
            continue
        first_arg = node.args[0]
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            declared.append(first_arg.value)
    return declared


def test_evidence_registry_is_complete_and_not_self_claimed_reproduction():
    assert set(LESSONS) == {f"M{i:02d}" for i in range(17)}
    assert set(EVIDENCE_DESCRIPTIONS) == set(EvidenceLevel)
    for lesson in LESSONS.values():
        assert lesson.evidence
        assert EvidenceLevel.PAPER_REPRODUCTION not in lesson.evidence
        assert lesson.references
        for ref in lesson.references:
            assert ref.author_or_organization
            assert ref.year > 0
            assert ref.stable_identifier.startswith("https://")
            assert ref.supports


def test_every_lesson_has_four_traceable_result_claims():
    assert len(CLAIMS) == 17 * 4
    assert len({claim.result_id for claim in CLAIMS.values()}) == len(CLAIMS)
    for lesson_id in LESSONS:
        lesson_claims = [claim for claim in CLAIMS.values() if claim.lesson_id == lesson_id]
        assert {claim.kind for claim in lesson_claims} == set(ClaimKind)
        for claim in lesson_claims:
            assert claim.statement and claim.conditions and claim.limitations and claim.sources
            assert claim.last_verified == "2026-08-22"
            assert get_result_claim(lesson_id, claim.result_id) is claim
            assert claim.evidence_level is not EvidenceLevel.PAPER_REPRODUCTION


def test_curriculum_dag_and_learning_loops_cover_every_lesson_without_cycles():
    expected = set(LESSONS)
    assert set(CURRICULUM_DAG) == expected
    assert set(LEARNING_LOOPS) == expected

    def ancestors(lesson_id: str, path: tuple[str, ...] = ()) -> set[str]:
        assert lesson_id not in path
        result: set[str] = set()
        for prerequisite in CURRICULUM_DAG[lesson_id]:
            assert prerequisite in expected
            result.add(prerequisite)
            result.update(ancestors(prerequisite, (*path, lesson_id)))
        return result

    for lesson_id, loop in LEARNING_LOOPS.items():
        ancestors(lesson_id)
        assert loop.diagnostic_question
        assert loop.minimum_experiment
        assert loop.counterexample_experiment
        assert loop.formative_assessment
        assert "适用条件" in loop.pass_criteria


def test_formative_quiz_registry_and_feedback_retry_flow():
    assert set(FORMATIVE_QUIZZES) == set(LESSONS)
    for quiz in FORMATIVE_QUIZZES.values():
        assert len(quiz.options) == 3
        assert len(set(quiz.options)) == 3
        assert 0 <= quiz.correct_index < 3

    at = AppTest.from_string(
        "from dashboard.components.pedagogy import render_formative_quiz\n"
        "render_formative_quiz('M01')\n"
    ).run()
    assert not at.exception
    assert not at.warning and not at.error and not at.success

    at.button[0].click().run()
    assert at.warning and "不会在作答前显示" in at.warning[0].value
    quiz = FORMATIVE_QUIZZES["M01"]
    wrong_index = next(i for i in range(3) if i != quiz.correct_index)
    at.radio[0].set_value(quiz.options[wrong_index]).run()
    at.button[0].click().run()
    assert at.error and quiz.diagnostic_feedback in at.error[0].value
    at.radio[0].set_value(quiz.options[quiz.correct_index]).run()
    at.button[0].click().run()
    assert at.success and quiz.correct_explanation in at.success[0].value


def test_unknown_result_claim_is_rejected():
    with pytest.raises(ValueError, match="未注册"):
        get_result_claim("M00", "missing")


def test_evidence_badges_are_machine_readable():
    for level in EvidenceLevel:
        badge = evidence_badge(level)
        assert f'data-evidence-level="{level.name}"' in badge
        assert level.value in badge


def test_unknown_evidence_level_is_rejected():
    with pytest.raises(KeyError):
        evidence_badge("NOT_A_LEVEL")  # type: ignore[arg-type]


def test_every_page_declares_exactly_one_matching_lesson():
    pages = sorted(PAGES_DIR.glob("*.py"))
    assert len(pages) == 17
    for page in pages:
        lesson_number = int(page.name.split("_", maxsplit=1)[0])
        expected = f"M{lesson_number:02d}"
        assert _declared_string_args(page, "render_lesson_evidence") == [expected], page.name
        assert _declared_string_args(page, "render_core_result_evidence") == [expected], page.name


def test_known_misleading_claims_do_not_return():
    transformer = (PAGES_DIR / "8_Transformer.py").read_text(encoding="utf-8")
    audio = (PAGES_DIR / "11_音频感知.py").read_text(encoding="utf-8")
    video = (PAGES_DIR / "12_视频与世界模型.py").read_text(encoding="utf-8")
    evaluation = (PAGES_DIR / "15_评估基准.py").read_text(encoding="utf-8")

    assert "残差连接能让深层网络绝不发生梯度消失" not in transformer
    assert "高层开始跨距跳跃关注语义相关词" not in transformer
    assert "Whisper 语音 Token 化" not in audio
    assert "SORA GENERATIVE PIPELINE" not in video
    assert "开放大语言模型权威天梯排行榜" not in evaluation
    assert "Claude 3.5 Sonnet / Opus 4.6" not in evaluation


def test_simulated_evaluation_is_labeled_next_to_results():
    source = (PAGES_DIR / "15_评估基准.py").read_text(encoding="utf-8")
    assert "mock_predict_fn" in source
    assert "SIMULATED PPL // 模拟困惑度" in source
    assert "所有数值均不是任何真实模型或正式 benchmark 的成绩" in source


def test_audio_api_states_continuous_feature_boundary():
    source = (PROJECT_ROOT / "nn_core" / "audio.py").read_text(encoding="utf-8")
    assert "class SpectrogramFramePatcher" in source
    assert "class AudioTokenizer(SpectrogramFramePatcher)" in source
    assert "不是离散 token id" in source


def test_world_model_states_missing_reverse_process():
    source = (PROJECT_ROOT / "nn_core" / "world_model.py").read_text(encoding="utf-8")
    assert "不含反向去噪网络" in source
