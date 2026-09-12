"""主机发现路径: descriptor, 配置校验与 zip 安装."""

from __future__ import annotations

import io
import shutil
import zipfile
from pathlib import Path
from typing import cast

import pytest
from amane.config import PluginConfig
from amane.plugin import (
    PLUGIN_API_VERSION,
    ContentType,
    EmptyPluginConfig,
    FilePlaybackTarget,
    HttpClient,
    PlaybackMediaFile,
    PlaybackPlugin,
    PlaybackProvider,
    PlaybackQuery,
    PluginContext,
    SourceCapability,
    WebClient,
)
from amane.plugins.manager import PluginManager
from amane.plugins.packaging import install_plugin_zip
from pydantic import ValidationError

PLUGIN_ID = "sqzw.local"
ROOT = Path(__file__).resolve().parents[1]


def dropin_modules() -> list[Path]:
    return sorted(ROOT.glob("*.py"))


def install_dropin(data_dir: Path) -> Path:
    dest = data_dir / "plugins" / "sources" / PLUGIN_ID
    dest.mkdir(parents=True)
    for path in dropin_modules():
        shutil.copy(path, dest / path.name)
    return dest


def build_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for path in dropin_modules():
            archive.write(path, arcname=path.name)
    return buffer.getvalue()


def plugin_context(tmp_path: Path) -> PluginContext:
    """本插件不使用 ``context``: 不联网, 不创建客户端, 也不写 ``data_dir``."""
    return PluginContext(
        source_id=PLUGIN_ID,
        http_client=cast(HttpClient, None),
        web_client=cast(WebClient, None),
        data_dir=tmp_path,
    )


def media_file(path: Path) -> PlaybackMediaFile:
    return PlaybackMediaFile(
        id=1,
        path=str(path),
        content_type=ContentType.CENSORED,
        library_id=1,
        library_path=str(path.parent),
    )


def test_discover_playback_plugin(tmp_path: Path) -> None:
    """目录名与 descriptor id 一致, 仅声明播放能力."""
    install_dropin(tmp_path)
    manager = PluginManager.discover(tmp_path)
    assert manager.failures == ()
    assert manager.has_playback_plugin(PLUGIN_ID)
    assert not manager.has_film_plugin(PLUGIN_ID)

    plugin = manager.get(PLUGIN_ID)
    assert isinstance(plugin, PlaybackPlugin)
    descriptor = plugin.descriptor()
    assert descriptor.id == PLUGIN_ID
    assert descriptor.api_version == PLUGIN_API_VERSION
    assert descriptor.supports(SourceCapability.PLAYBACK)
    assert not descriptor.supports(SourceCapability.FILM_METADATA)
    assert descriptor.version == "0.1.0"


def test_install_zip_keeps_entry_at_root(tmp_path: Path) -> None:
    """宿主接受根目录含 ``plugin.py`` 的 zip."""
    payload = build_zip()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert "plugin.py" in archive.namelist()

    assert install_plugin_zip(tmp_path, payload) == PLUGIN_ID
    manager = PluginManager.discover(tmp_path)
    assert manager.failures == ()
    assert manager.has_playback_plugin(PLUGIN_ID)


@pytest.mark.asyncio
async def test_provider_end_to_end(tmp_path: Path) -> None:
    """经宿主装载入口构造的 provider 能探测, 解析并转换同目录字幕."""
    install_dropin(tmp_path)
    manager = PluginManager.discover(tmp_path)
    provider = manager.build_playback_provider(
        PLUGIN_ID,
        context=plugin_context(tmp_path),
        config=PluginConfig(),
    )
    assert isinstance(provider, PlaybackProvider)

    library = tmp_path / "library"
    library.mkdir()
    video = library / "movie.mp4"
    video.write_bytes(b"0123456789")
    (library / "movie.srt").write_bytes("1\n00:00:01,000 --> 00:00:04,500\n中文字幕\n".encode("gbk"))
    payload = PlaybackQuery(metadata_id=1, number="ABC-001", files=(media_file(video),))

    offers = await provider.probe(payload)
    assert len(offers) == 1
    assert [track.id for track in offers[0].subtitles] == ["sidecar"]

    target = await provider.resolve(payload)
    assert isinstance(target, FilePlaybackTarget)
    assert target.path == video
    assert target.content_type == offers[0].content_type

    text = await provider.subtitle(payload, "sidecar")
    assert isinstance(text, str)
    assert "中文字幕" in text


@pytest.mark.asyncio
async def test_config_reaches_provider(tmp_path: Path) -> None:
    """``plugins.<id>.config`` 经宿主校验后进入 provider: 关闭字幕即不再声明轨道."""
    install_dropin(tmp_path)
    manager = PluginManager.discover(tmp_path)
    provider = manager.build_playback_provider(
        PLUGIN_ID,
        context=plugin_context(tmp_path),
        config=PluginConfig(config={"sidecar_subtitles": False}),
    )

    library = tmp_path / "library"
    library.mkdir()
    video = library / "movie.mp4"
    video.write_bytes(b"x")
    (library / "movie.srt").write_bytes(b"1\n00:00:01,000 --> 00:00:02,000\nHi\n")
    payload = PlaybackQuery(metadata_id=1, number="ABC-001", files=(media_file(video),))

    offers = await provider.probe(payload)
    assert [offer.subtitles for offer in offers] == [()]
    assert await provider.subtitle(payload, "sidecar") is None


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({}, (True, (".vtt", ".srt"))),
        ({"sidecar_subtitles": False}, (False, (".vtt", ".srt"))),
        # 后缀统一为小写并去除重复项.
        ({"sidecar_suffixes": [".SRT", ".srt", ".ass"]}, (True, (".srt", ".ass"))),
        ({"sidecar_suffixes": []}, (True, ())),
    ],
)
def test_config_defaults_and_normalization(
    tmp_path: Path,
    payload: dict[str, object],
    expected: tuple[bool, tuple[str, ...]],
) -> None:
    install_dropin(tmp_path)
    model = PluginManager.discover(tmp_path).get(PLUGIN_ID)
    assert isinstance(model, PlaybackPlugin)
    data = model.configuration_model().model_validate(payload).model_dump()
    assert (data["sidecar_subtitles"], tuple(data["sidecar_suffixes"])) == expected


@pytest.mark.parametrize(
    "payload",
    [
        {"unknown": 1},
        # ``.strm`` 的处理方式不开放配置: 该键已移除, 现在按未知字段拒绝.
        {"skip_strm": True},
        {"sidecar_subtitles": "maybe"},
        {"sidecar_suffixes": ["srt"]},
        {"sidecar_suffixes": ["../etc"]},
        {"sidecar_suffixes": [".vtt/x"]},
        {"sidecar_suffixes": [""]},
    ],
)
def test_config_rejects_invalid_values(tmp_path: Path, payload: dict[str, object]) -> None:
    install_dropin(tmp_path)
    plugin = PluginManager.discover(tmp_path).get(PLUGIN_ID)
    assert isinstance(plugin, PlaybackPlugin)
    with pytest.raises(ValidationError):
        plugin.configuration_model().model_validate(payload)


def test_build_playback_rejects_foreign_config(tmp_path: Path) -> None:
    install_dropin(tmp_path)
    plugin = PluginManager.discover(tmp_path).get(PLUGIN_ID)
    assert isinstance(plugin, PlaybackPlugin)
    with pytest.raises(TypeError):
        plugin.build_playback(plugin_context(tmp_path), EmptyPluginConfig())
