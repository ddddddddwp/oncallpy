import tomllib
from pathlib import Path

from app.config import Settings

ROOT = Path(__file__).resolve().parents[1]


def test_energyagent_is_the_canonical_product_name() -> None:
    settings = Settings(_env_file=None)
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    html = (ROOT / "static/index.html").read_text(encoding="utf-8")
    javascript = (ROOT / "static/app.js").read_text(encoding="utf-8")

    assert settings.app_name == "EnergyAgent"
    assert metadata["project"]["name"] == "energy-agent"
    assert metadata["project"]["description"] == (
        "EnergyAgent 智能能源运维平台 - 支持 RAG、AIOps、告警与值班协同"
    )
    assert "EnergyAgent" in html
    assert 'placeholder="问问 EnergyAgent"' in html
    assert "class EnergyAgentApp" in javascript
    assert "new EnergyAgentApp(" in javascript
    assert "this.messageInput.placeholder = '问问 EnergyAgent'" in javascript
    assert "SuperBizAgentApp" not in javascript
    assert "问问智能OnCall助手" not in javascript


def test_product_surfaces_do_not_present_legacy_branding() -> None:
    paths = [
        ROOT / "app/__init__.py",
        ROOT / "app/config.py",
        ROOT / "app/main.py",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        ROOT / "static/app.js",
        ROOT / "static/index.html",
        ROOT / "Makefile",
        ROOT / "README.md",
        ROOT / "docs/ONBOARDING.md",
        ROOT / "start-windows.bat",
        ROOT / "stop-windows.bat",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    assert "SuperBizAgent" not in text
    assert "oncallpy" not in text.lower()
