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
    assert set(LESSONS) == {f"M{i:02d}" for i in range(18)}
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
    assert len(CLAIMS) == 18 * 4
    assert len({claim.result_id for claim in CLAIMS.values()}) == len(CLAIMS)
    for lesson_id in LESSONS:
        lesson_claims = [claim for claim in CLAIMS.values() if claim.lesson_id == lesson_id]
        assert {claim.kind for claim in lesson_claims} == set(ClaimKind)
        for claim in lesson_claims:
            assert claim.statement and claim.conditions and claim.limitations and claim.sources
            assert claim.last_verified == "2026-08-23"
            assert get_result_claim(lesson_id, claim.result_id) is claim
            assert claim.evidence_level is not EvidenceLevel.PAPER_REPRODUCTION


def test_independent_claim_to_source_semantic_mapping():
    """使用独立人工审校映射表，消除'代码生成什么，测试断言什么'的自循环验证。"""
    audited_claim_sources = {
        "m00-formula": "The Matrix Calculus You Need For Deep Learning",
        "m00-result": "Numerical Optimization",
        "m00-history": "The Matrix Calculus You Need For Deep Learning",
        "m00-failure": "Numerical Optimization",
        "m01-formula": "The Perceptron",
        "m01-result": "The Perceptron",
        "m01-history": "The Perceptron",
        "m01-failure": "The Perceptron",
        "m02-formula": "Learning representations by back-propagating errors",
        "m02-result": "Learning representations by back-propagating errors",
        "m02-history": "Learning representations by back-propagating errors",
        "m02-failure": "Learning representations by back-propagating errors",
        "m03-formula": "Adam: A Method for Stochastic Optimization",
        "m03-result": "Adam: A Method for Stochastic Optimization",
        "m03-history": "Adam: A Method for Stochastic Optimization",
        "m03-failure": "Decoupled Weight Decay Regularization",
        "m04-formula": "Deep Learning",
        "m04-result": "Deep Learning",
        "m04-history": "Deep Learning",
        "m04-failure": "Deep Learning",
        "m05-formula": "Efficient Estimation of Word Representations",
        "m05-result": "Efficient Estimation of Word Representations",
        "m05-history": "Efficient Estimation of Word Representations",
        "m05-failure": "Visualizing Data using t-SNE",
        "m06-formula": "Long Short-Term Memory",
        "m06-result": "Long Short-Term Memory",
        "m06-history": "Learning Phrase Representations using RNN Encoder-Decoder",
        "m06-failure": "Learning long-term dependencies with gradient descent is difficult",
        "m07-formula": "Attention Is All You Need",
        "m07-result": "Attention Is All You Need",
        "m07-history": "Neural Machine Translation by Jointly Learning to Align and Translate",
        "m07-failure": "Attention is not Explanation",
        "m08-formula": "Attention Is All You Need",
        "m08-result": "Attention Is All You Need",
        "m08-history": "Layer Normalization",
        "m08-failure": "On Layer Normalization in the Transformer Architecture",
        "m09-formula": "Language Models are Unsupervised Multitask Learners",
        "m09-result": "Language Models are Unsupervised Multitask Learners",
        "m09-history": "Language Models are Unsupervised Multitask Learners",
        "m09-failure": "Language Models are Unsupervised Multitask Learners",
        "m10-formula": "Gradient-Based Learning Applied to Document Recognition",
        "m10-result": "An Image is Worth 16x16 Words",
        "m10-history": "Learning Transferable Visual Models From Natural Language Supervision",
        "m10-failure": "Visualizing and Understanding Convolutional Networks",
        "m11-formula": "A Tutorial on Short-Time Spectrum Analysis",
        "m11-result": "A Scale for the Measurement of the Psychological Magnitude Pitch",
        "m11-history": "Robust Speech Recognition via Large-Scale Weak Supervision",
        "m11-failure": "A Tutorial on Short-Time Spectrum Analysis",
        "m12-formula": "Denoising Diffusion Probabilistic Models",
        "m12-result": "Denoising Diffusion Probabilistic Models",
        "m12-history": "Scalable Diffusion Models with Transformers",
        "m12-failure": "Denoising Diffusion Probabilistic Models",
        "m13-formula": "BERT: Pre-training of Deep Bidirectional Transformers",
        "m13-result": "Training Compute-Optimal Large Language Models",
        "m13-history": "Neural Machine Translation of Rare Words with Subword Units",
        "m13-failure": "Training Compute-Optimal Large Language Models",
        "m14-formula": "LoRA: Low-Rank Adaptation of Large Language Models",
        "m14-result": "Direct Preference Optimization",
        "m14-history": "Training language models to follow instructions with human feedback",
        "m14-failure": "Proximal Policy Optimization Algorithms",
        "m15-formula": "Speech and Language Processing",
        "m15-result": "Measuring Massive Multitask Language Understanding",
        "m15-history": "Holistic Evaluation of Language Models",
        "m15-failure": "Training Verifiers to Solve Math Word Problems",
        "m16-formula": "Reinforcement Learning: An Introduction",
        "m16-result": "Q-learning",
        "m16-history": "Simple statistical gradient-following algorithms for connectionist reinforcement learning",
        "m16-failure": "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning",
        "m17-formula": "Efficient Streaming Language Models with Attention Sinks",
        "m17-result": "Lost in the Middle: How Language Models Use Long Contexts",
        "m17-history": "April 23, 2026: Claude Code Postmortem",
        "m17-failure": "The Reversal Curse: LLMs trained on 'A is B' fail to learn 'B is A'",
    }
    assert len(audited_claim_sources) == 72
    for claim_id, expected_title in audited_claim_sources.items():
        assert claim_id in CLAIMS, f"缺失 Claim ID: {claim_id}"
        actual_title = CLAIMS[claim_id].sources[0].title
        assert actual_title == expected_title, (
            f"Claim [{claim_id}] 来源绑定不匹配: expected={expected_title}, actual={actual_title}"
        )


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
        assert quiz.sources, "每道形成性测验必须显式绑定至少一份权威文献"
        assert quiz.conditions, "每道形成性测验必须注明严格适用条件"
        assert quiz.limitations, "每道形成性测验必须注明局限性与反例"

    # 关键题目科学语义人工核验
    assert (
        "解耦权重衰减" in FORMATIVE_QUIZZES["M04"].options[FORMATIVE_QUIZZES["M04"].correct_index]
    )
    assert (
        "雅可比矩阵连乘" in FORMATIVE_QUIZZES["M06"].options[FORMATIVE_QUIZZES["M06"].correct_index]
    )
    assert "Pre-LN" in FORMATIVE_QUIZZES["M08"].options[FORMATIVE_QUIZZES["M08"].correct_index]
    assert "心理声学" in FORMATIVE_QUIZZES["M11"].options[FORMATIVE_QUIZZES["M11"].correct_index]
    assert (
        "分词器词表粒度" in FORMATIVE_QUIZZES["M15"].options[FORMATIVE_QUIZZES["M15"].correct_index]
    )
    assert (
        "悲观下界估计" in FORMATIVE_QUIZZES["M16"].options[FORMATIVE_QUIZZES["M16"].correct_index]
    )

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
    assert len(pages) == 18
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
