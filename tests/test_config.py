"""Configuration model regression tests."""
from __future__ import annotations

import json

import pytest

from src.config import AppPathDefaults, AppSettings, load_settings


def test_with_overrides_updates_target_section() -> None:
    base = AppSettings()
    updated = base.with_overrides({"retrieval": {"enable_hybrid": False}, "llm": {"llm_provider": "openai"}})

    assert updated.retrieval.enable_hybrid is False
    assert updated.llm.llm_provider == "openai"
    assert base.retrieval.enable_hybrid is True
    assert base.llm.llm_provider == "local"


def test_load_settings_supports_nested_config(tmp_path) -> None:
    cfg_path = tmp_path / "tekturo.config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "llm": {"llm_provider": "openai"},
                "retrieval": {"enable_hybrid": False},
            }
        ),
        encoding="utf-8",
    )

    settings = load_settings(config_file=str(cfg_path))

    assert settings.llm_provider == "openai"
    assert settings.enable_hybrid is False


def test_load_settings_rejects_unknown_config_key(tmp_path) -> None:
    cfg_path = tmp_path / "tekturo.config.json"
    cfg_path.write_text(json.dumps({"unknown_key": 123}), encoding="utf-8")

    with pytest.raises(RuntimeError):
        load_settings(config_file=str(cfg_path))


def test_default_paths_are_centralized() -> None:
    assert str(AppPathDefaults.home_dir()).endswith(".tekturo")
    assert AppPathDefaults.DEFAULT_COLLECTION == "tekturo_db"


def test_app_settings_rejects_flat_overrides() -> None:
    strict = AppSettings()

    with pytest.raises(ValueError):
        strict.with_overrides({"llm_provider": "openai"})


def test_app_settings_accepts_nested_overrides() -> None:
    strict = AppSettings()
    updated = strict.with_overrides({"llm": {"llm_provider": "openai"}})

    assert updated.llm.llm_provider == "openai"


def test_load_settings_rejects_flat_config_keys(tmp_path) -> None:
    cfg_path = tmp_path / "tekturo.config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "llm_provider": "openai",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError):
        load_settings(config_file=str(cfg_path))


def test_load_settings_allows_nested_config_keys(tmp_path) -> None:
    cfg_path = tmp_path / "tekturo.config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "llm": {"llm_provider": "openai"},
            }
        ),
        encoding="utf-8",
    )

    settings = load_settings(config_file=str(cfg_path))
    assert settings.llm.llm_provider == "openai"
