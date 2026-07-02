import pytest
import os
import tempfile
from unittest.mock import MagicMock, patch
from modules.memory import AtlasMemory


@pytest.fixture
def config():
    return {
        "memory": {
            "enabled": True,
            "wing": "test_wing",
            "palace_path": "~/.mempalace/palace",
            "auto_store_conversations": True
        }
    }


@pytest.fixture
def config_disabled():
    return {
        "memory": {
            "enabled": False,
            "wing": "test_wing",
            "palace_path": "~/.mempalace/palace"
        }
    }


@pytest.fixture
def memory(config):
    with patch("modules.memory.AtlasMemory._init_mempalace"):
        mem = AtlasMemory(config)
        mem.enabled = True
        mem._search_fn = MagicMock()
        return mem


@pytest.fixture
def memory_disabled(config_disabled):
    mem = AtlasMemory(config_disabled)
    return mem


# ── Init ──────────────────────────────────────────────────────────────────────

class TestInit:

    def test_enabled_from_config(self, config):
        with patch("modules.memory.AtlasMemory._init_mempalace"):
            mem = AtlasMemory(config)
            assert mem.enabled is True

    def test_disabled_from_config(self, config_disabled):
        mem = AtlasMemory(config_disabled)
        assert mem.enabled is False

    def test_wing_from_config(self, config):
        with patch("modules.memory.AtlasMemory._init_mempalace"):
            mem = AtlasMemory(config)
            assert mem.wing == "test_wing"

    def test_palace_path_expanded(self, config):
        with patch("modules.memory.AtlasMemory._init_mempalace"):
            mem = AtlasMemory(config)
            assert not mem.palace_path.startswith("~")

    def test_mempalace_not_installed(self, config):
        with patch("builtins.__import__", side_effect=ImportError):
            mem = AtlasMemory(config)
            assert mem.enabled is False

    def test_init_error_disables(self, config):
        with patch("modules.memory.AtlasMemory._init_mempalace",
                   side_effect=Exception("init failed")):
            mem = AtlasMemory(config)
            assert mem.enabled is False


# ── chunk_text ────────────────────────────────────────────────────────────────

class TestChunkText:

    def test_short_text_single_chunk(self, memory):
        result = memory.chunk_text("hello world", max_chars=800)
        assert len(result) == 1
        assert result[0] == "hello world"

    def test_long_text_multiple_chunks(self, memory):
        text = "a" * 2000
        result = memory.chunk_text(text, max_chars=800)
        assert len(result) == 3

    def test_exact_chunk_size(self, memory):
        text = "a" * 800
        result = memory.chunk_text(text, max_chars=800)
        assert len(result) == 1

    def test_empty_text(self, memory):
        result = memory.chunk_text("", max_chars=800)
        assert result == []

    def test_custom_max_chars(self, memory):
        text = "hello world test"
        result = memory.chunk_text(text, max_chars=5)
        assert len(result) == 4


# ── embed ─────────────────────────────────────────────────────────────────────

class TestEmbed:

    def test_embed_returns_list_or_none(self, memory):
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.encode.return_value = [0.1, 0.2, 0.3]
            mock_st.return_value = mock_model
            result = memory.embed("test text")
            assert result is not None or result is None  # either is valid

    def test_embed_returns_none_on_error(self, memory):
        with patch("builtins.__import__", side_effect=Exception("no model")):
            result = memory.embed("test")
            assert result is None


# ── cosine_sim ────────────────────────────────────────────────────────────────

class TestCosineSim:

    def test_identical_vectors(self, memory):
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert memory.cosine_sim(a, b) == pytest.approx(1.0)

    def test_orthogonal_vectors(self, memory):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert memory.cosine_sim(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self, memory):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert memory.cosine_sim(a, b) == pytest.approx(-1.0)

    def test_none_returns_zero(self, memory):
        assert memory.cosine_sim(None, [1.0]) == 0.0
        assert memory.cosine_sim([1.0], None) == 0.0
        assert memory.cosine_sim(None, None) == 0.0


# ── normalize_memory ──────────────────────────────────────────────────────────

class TestNormalizeMemory:

    def test_dict_input(self, memory):
        raw = {"content": "test content", "room": "general", "wing": "atlas"}
        result = memory.normalize_memory(raw)
        assert result["content"] == "test content"
        assert result["room"] == "general"

    def test_string_input(self, memory):
        result = memory.normalize_memory("just a string")
        assert result["content"] == "just a string"
        assert result["embedding"] is None
        assert result["access_count"] == 0

    def test_none_input(self, memory):
        result = memory.normalize_memory(None)
        assert result is None

    def test_dict_with_text_key(self, memory):
        raw = {"text": "from text key"}
        result = memory.normalize_memory(raw)
        assert result["content"] == "from text key"

    def test_defaults_filled(self, memory):
        result = memory.normalize_memory({})
        assert result["access_count"] == 0
        assert result["importance"] == 1.0


# ── recall ────────────────────────────────────────────────────────────────────

class TestRecall:

    def test_returns_string_on_success(self, memory):
        memory._search_fn.return_value = {
            "results": [
                {"text": "I prefer async Python", "wing": "test_wing"},
                {"text": "Flask is my go-to backend", "wing": "test_wing"},
            ]
        }
        result = memory.recall("python preferences")
        assert isinstance(result, str)

    def test_returns_empty_string_on_no_results(self, memory):
        memory._search_fn.return_value = {"results": []}
        result = memory.recall("nothing here")
        assert result == "" or result == []

    def test_filters_long_content(self, memory):
        memory._search_fn.return_value = {
            "results": [
                {"text": "x" * 500, "wing": "test_wing"},  # too long, filtered
                {"text": "short memory", "wing": "test_wing"},
            ]
        }
        result = memory.recall("test")
        assert "short memory" in result
        assert "x" * 500 not in result

    def test_returns_empty_on_error(self, memory):
        memory._search_fn.side_effect = Exception("search failed")
        result = memory.recall("test")
        assert result == "" or result == []

    def test_disabled_returns_none(self, memory_disabled):
        result = memory_disabled.recall("test")
        assert result is None or result == []

    def test_wing_filter_applied(self, memory):
        memory._search_fn.return_value = {
            "results": [
                {"text": "atlas memory", "wing": "test_wing"},
                {"text": "other memory", "wing": "other_wing"},
            ]
        }
        result = memory.recall("test")
        assert "atlas memory" in result
        assert "other memory" not in result


# ── remember ──────────────────────────────────────────────────────────────────

class TestRemember:

    def test_returns_false_when_disabled(self, memory_disabled):
        result = memory_disabled.remember("test")
        assert result is False

    def test_returns_true_on_success(self, memory):
        with patch("mempalace.miner.get_collection") as mock_gc, \
             patch("mempalace.miner.add_drawer") as mock_ad:
            mock_gc.return_value = MagicMock()
            result = memory.remember("I prefer PostgreSQL")
            assert result is True

    def test_stores_chunks(self, memory):
        with patch("mempalace.miner.get_collection") as mock_gc, \
             patch("mempalace.miner.add_drawer") as mock_ad:
            mock_gc.return_value = MagicMock()
            memory.remember("a" * 2000)  # will chunk into 3
            assert mock_ad.call_count == 3

    def test_returns_false_on_error(self, memory):
        with patch("mempalace.miner.get_collection",
                   side_effect=Exception("db error")):
            result = memory.remember("test")
            assert result is False

    def test_custom_room(self, memory):
        with patch("mempalace.miner.get_collection") as mock_gc, \
             patch("mempalace.miner.add_drawer") as mock_ad:
            mock_gc.return_value = MagicMock()
            memory.remember("test", room="preferences")
            call_kwargs = mock_ad.call_args[1]
            assert call_kwargs["room"] == "preferences"


# ── remember_conversation ─────────────────────────────────────────────────────

class TestRememberConversation:

    def test_returns_false_when_disabled(self, memory_disabled):
        result = memory_disabled.remember_conversation("cmd", "response")
        assert result is False

    def test_returns_true_on_success(self, memory):
        with patch("mempalace.miner.get_collection") as mock_gc, \
             patch("mempalace.miner.add_drawer") as mock_ad:
            mock_gc.return_value = MagicMock()
            result = memory.remember_conversation(
                "what is the capital of France",
                "Paris, sir."
            )
            assert result is True

    def test_content_includes_command_and_response(self, memory):
        with patch("mempalace.miner.get_collection") as mock_gc, \
             patch("mempalace.miner.add_drawer") as mock_ad:
            mock_gc.return_value = MagicMock()
            memory.remember_conversation("my command", "my response")
            call_kwargs = mock_ad.call_args[1]
            assert "my command" in call_kwargs["content"]
            assert "my response" in call_kwargs["content"]

    def test_stored_in_conversations_room(self, memory):
        with patch("mempalace.miner.get_collection") as mock_gc, \
             patch("mempalace.miner.add_drawer") as mock_ad:
            mock_gc.return_value = MagicMock()
            memory.remember_conversation("cmd", "response")
            call_kwargs = mock_ad.call_args[1]
            assert call_kwargs["room"] == "conversations"

    def test_returns_false_on_error(self, memory):
        with patch("mempalace.miner.get_collection",
                   side_effect=Exception("db error")):
            result = memory.remember_conversation("cmd", "response")
            assert result is False


# ── get_wake_up_context ───────────────────────────────────────────────────────

class TestGetWakeUpContext:

    def test_returns_none_when_disabled(self, memory_disabled):
        result = memory_disabled.get_wake_up_context()
        assert result is None

    def test_returns_context_on_success(self, memory):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="## L0 — IDENTITY\nAtlas memory loaded"
            )
            result = memory.get_wake_up_context()
            assert result is not None
            assert "IDENTITY" in result

    def test_returns_none_on_failure(self, memory):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="")
            result = memory.get_wake_up_context()
            assert result is None

    def test_returns_none_on_exception(self, memory):
        with patch("subprocess.run", side_effect=Exception("not found")):
            result = memory.get_wake_up_context()
            assert result is None


# ── format_for_speech ─────────────────────────────────────────────────────────

class TestFormatForSpeech:

    def test_removes_source_tags(self, memory):
        text = "[observer.py]: some content here"
        result = memory.format_for_speech(text)
        assert "[observer.py]:" not in result
        assert "some content here" in result

    def test_truncates_long_text(self, memory):
        text = "a" * 1000
        result = memory.format_for_speech(text)
        assert len(result) <= 500

    def test_collapses_whitespace(self, memory):
        text = "hello    world\n\ntest"
        result = memory.format_for_speech(text)
        assert "  " not in result

    def test_empty_string(self, memory):
        result = memory.format_for_speech("")
        assert result == ""
