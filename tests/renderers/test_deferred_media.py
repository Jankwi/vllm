# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
from types import SimpleNamespace

import pytest

import vllm.renderers.base as renderer_module
from vllm.multimodal.processing import ProcessorInputs
from vllm.renderers.hf import HfRenderer


class _Cache:
    def __init__(self, hits: set[str]) -> None:
        self.hits = hits
        self.probed: list[str] = []

    def is_cached_item(self, key: str) -> bool:
        self.probed.append(key)
        return key in self.hits


class _ProcessorInfo:
    model_id = "test-model"
    ctx = SimpleNamespace(
        get_mm_config=lambda: SimpleNamespace(mm_hasher_algorithm="sha256")
    )

    @staticmethod
    def parse_mm_data(data):
        return data


class _Processor:
    def __init__(self, cache: _Cache) -> None:
        self.cache = cache
        self.info = _ProcessorInfo()


def _setup_resolver(
    monkeypatch: pytest.MonkeyPatch,
    *,
    hits: set[str],
):
    fetched: list[str] = []

    class _Connector:
        def __init__(self, **_kwargs) -> None:
            pass

        def fetch_video(self, url: str, **_kwargs):
            fetched.append(url)
            return f"loaded:{url}"

    def get_mm_hashes(self, _model_id, _hash_algorithm):
        return {"video": list((self.mm_uuid_items or {})["video"])}

    monkeypatch.setattr(renderer_module, "MediaConnector", _Connector)
    monkeypatch.setattr(renderer_module, "get_video_processor_cls_name", lambda _: None)
    monkeypatch.setattr(ProcessorInputs, "get_mm_hashes", get_mm_hashes)

    renderer = object.__new__(HfRenderer)
    renderer.model_config = SimpleNamespace(
        multimodal_config=None,
        allowed_local_media_path="",
        allowed_media_domains=None,
    )
    cache = _Cache(hits)
    return renderer, _Processor(cache), cache, fetched


def test_early_lookup_is_opt_in(monkeypatch: pytest.MonkeyPatch):
    renderer, processor, cache, fetched = _setup_resolver(
        monkeypatch, hits={"video-id"}
    )
    monkeypatch.setenv("VLLM_EARLY_UUID_LOOKUPS", "0")

    mm_data, mm_uuids, _ = renderer._resolve_video_sources(
        [1],
        {"video": ["https://example.com/video.mp4"]},
        {"video": ["video-id"]},
        processor,
        None,
        {},
    )

    assert cache.probed == []
    assert fetched == ["https://example.com/video.mp4"]
    assert mm_data["video"] == ["loaded:https://example.com/video.mp4"]
    assert mm_uuids == {"video": ["video-id"]}


def test_early_lookup_resolves_only_misses(monkeypatch: pytest.MonkeyPatch):
    renderer, processor, cache, fetched = _setup_resolver(
        monkeypatch, hits={"cached-id"}
    )
    monkeypatch.setenv("VLLM_EARLY_UUID_LOOKUPS", "1")

    mm_data, mm_uuids, _ = renderer._resolve_video_sources(
        [1],
        {
            "video": [
                "https://example.com/cached.mp4",
                "https://example.com/missing.mp4",
            ]
        },
        {"video": ["cached-id", "missing-id"]},
        processor,
        None,
        {},
    )

    assert cache.probed == ["cached-id", "missing-id"]
    assert fetched == ["https://example.com/missing.mp4"]
    assert mm_data["video"] == [None, "loaded:https://example.com/missing.mp4"]
    assert mm_uuids == {"video": ["cached-id", "missing-id"]}


def test_url_uuid_derivation_preserves_explicit_uuid(
    monkeypatch: pytest.MonkeyPatch,
):
    video_url = "https://example.com/video.mp4"
    derived_uuid = f"url:{hashlib.sha256(video_url.encode()).hexdigest()}"
    renderer, processor, cache, _ = _setup_resolver(
        monkeypatch, hits={derived_uuid, "client-id"}
    )
    monkeypatch.setenv("VLLM_EARLY_UUID_LOOKUPS", "1")
    monkeypatch.setenv("VLLM_AUTO_DERIVE_UUID", "1")

    mm_data, mm_uuids, _ = renderer._resolve_video_sources(
        [1],
        {"video": [video_url]},
        None,
        processor,
        None,
        {},
    )

    assert cache.probed == [derived_uuid]
    assert mm_data["video"] == [None]
    assert mm_uuids == {"video": [derived_uuid]}

    cache.probed.clear()
    _, mm_uuids, _ = renderer._resolve_video_sources(
        [1],
        {"video": [video_url]},
        {"video": ["client-id"]},
        processor,
        None,
        {},
    )

    assert cache.probed == ["client-id"]
    assert mm_uuids == {"video": ["client-id"]}
