# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""教学证据、结论边界与课程注册表回归测试。"""

import ast
from pathlib import Path

import pytest

from dashboard.components.pedagogy import evidence_badge
from dashboard.constants.course import EVIDENCE_DESCRIPTIONS, LESSONS, EvidenceLevel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAGES_DIR = PROJECT_ROOT / "dashboard" / "pages"


def _declared_lesson_ids(page: Path) -> list[str]:
    tree = ast.parse(page.read_text(encoding="utf-8"))
    declared = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "render_lesson_evidence" or not node.args:
            continue
        first_arg = node.args[0]
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            declared.append(first_arg.value)
    return declared


def test_evidence_registry_is_complete_and_not_self_claimed_reproduction():
    assert set(LESSONS) == {f"M{i:02d}" for i in range(16)}
    assert set(EVIDENCE_DESCRIPTIONS) == set(EvidenceLevel)
    for lesson in LESSONS.values():
        assert lesson.evidence
        assert EvidenceLevel.PAPER_REPRODUCTION not in lesson.evidence
        assert lesson.references


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
    assert len(pages) == 16
    for page in pages:
        lesson_number = int(page.name.split("_", maxsplit=1)[0])
        expected = f"M{lesson_number:02d}"
        assert _declared_lesson_ids(page) == [expected], page.name


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
