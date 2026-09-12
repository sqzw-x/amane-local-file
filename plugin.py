"""本地文件播放源插件入口. 目录名与 ``descriptor.id`` 必须同为 ``sqzw.local``."""

from __future__ import annotations

import re
from typing import override

from amane.plugin import (
    PlaybackPlugin,
    PlaybackProvider,
    PluginContext,
    SourceCapability,
    SourceDescriptor,
)
from pydantic import BaseModel, ConfigDict, field_validator

from .local import DEFAULT_SIDECAR_SUFFIXES, SOURCE_NAME, LocalFileProvider

PLUGIN_ID = "sqzw.local"
PLUGIN_VERSION = "0.1.0"

_SUFFIX_PATTERN = re.compile(r"^\.[a-z0-9]+$")


class LocalFileConfig(BaseModel):
    """插件配置.

    字段名不包含宿主脱敏使用的关键词 (凭据类): 本插件不需要任何凭据, 配置值应完整出现在
    插件页与配置快照中.
    """

    model_config = ConfigDict(extra="forbid")

    sidecar_subtitles: bool = True
    sidecar_suffixes: tuple[str, ...] = DEFAULT_SIDECAR_SUFFIXES

    @field_validator("sidecar_suffixes")
    @classmethod
    def _normalize_suffixes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """统一为小写并去除重复项; 非法后缀在配置保存时就被拒绝, 不留给探测阶段."""
        normalized: list[str] = []
        for item in value:
            suffix = item.strip().casefold()
            if not _SUFFIX_PATTERN.fullmatch(suffix):
                raise ValueError(f"字幕后缀必须以点开头且只含字母与数字: {item!r}")
            if suffix not in normalized:
                normalized.append(suffix)
        return tuple(normalized)


class Plugin(PlaybackPlugin):
    config_model = LocalFileConfig

    @classmethod
    @override
    def descriptor(cls) -> SourceDescriptor:
        return SourceDescriptor(
            id=PLUGIN_ID,
            name=SOURCE_NAME,
            version=PLUGIN_VERSION,
            # 仅声明播放能力: 本插件不提供影片元数据, 也不写入内容路由.
            capabilities=frozenset({SourceCapability.PLAYBACK}),
        )

    @override
    def build_playback(self, context: PluginContext, config: BaseModel) -> PlaybackProvider:
        if not isinstance(config, LocalFileConfig):
            raise TypeError("unexpected config type")
        return LocalFileProvider(
            sidecar_subtitles=config.sidecar_subtitles,
            sidecar_suffixes=config.sidecar_suffixes,
        )
