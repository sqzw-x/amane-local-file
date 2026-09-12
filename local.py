"""本地已入库文件的播放实现: 候选列举, 磁盘检查与同目录字幕.

主机只接受该条目已索引的文件路径, 并自行打开与输出码流, 因此这里不存在目录与库根策略:
实现只从 ``query.files`` 快照里取文件, 并把快照里的字面路径原样声明回去. 条目里的每个文件
各是一条流, 用户在列表里选哪一条就播哪一个; 不可播的候选同样列出来并说明原因, 否则一个坏
文件会像凭空消失. ``.strm`` 清单按同一规则处理: 本插件不解析清单正文, 它记录的地址不会被
打开, 因此它只是一条带原因的不可播候选. 唯一读取的字节是同目录字幕 —— 它必须转换成 WebVTT
正文由插件返回, 主机不读取本机字幕文件.

主机在事件循环上直接调用 ``probe`` / ``resolve`` / ``subtitle``, 因此这三个钩子内的文件系统
调用全部提交给线程执行: 媒体库位于网络挂载时, ``stat`` 与 ``read_bytes`` 会长时间阻塞, 阻塞
发生在事件循环线程上时, 整个服务端在该调用返回之前停止推进, 其它来源的探测一并超时. 同步
辅助函数保留原接口, 线程切换只在钩子边界发生.
"""

from __future__ import annotations

import asyncio
import re
import stat
from pathlib import Path
from typing import override

from amane.plugin import (
    FailureReason,
    FilePlaybackTarget,
    PlaybackMediaFile,
    PlaybackOffer,
    PlaybackProvider,
    PlaybackQuery,
    SourceError,
    SubtitleTrack,
    UpstreamPlaybackTarget,
)

SOURCE_NAME = "本地文件"
SIDECAR_TRACK_ID = "sidecar"
STRM_SUFFIX = ".strm"
DEFAULT_SIDECAR_SUFFIXES: tuple[str, ...] = (".vtt", ".srt")
SUBTITLE_ENCODINGS: tuple[str, ...] = ("utf-8-sig", "gbk")
# 字幕是同目录的小文本文件. 上限防止命名成字幕的超大文件占用探测预算.
MAX_SIDECAR_BYTES = 4 * 1024 * 1024
FALLBACK_MEDIA_TYPE = "video/mp4"
# 条目一个已索引文件都没有时的原因, 直接展示给终端用户.
NO_FILE_DETAIL = "该条目没有已索引的文件"

# 类型表固定写在这里, 不查询系统 mime.types: 同一份媒体库在不同机器上必须声明同样的类型,
# 且 ``mimetypes`` 会把 ``.m3u8`` 判成清单类型 —— 本插件声明的是 ``file`` 目标, 不是 HLS.
MEDIA_TYPES: dict[str, str] = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".avi": "video/x-msvideo",
    ".mov": "video/quicktime",
    ".wmv": "video/x-ms-wmv",
    ".flv": "video/x-flv",
    ".mpg": "video/mpeg",
    ".mpeg": "video/mpeg",
    ".ts": "video/mp2t",
    ".m2ts": "video/mp2t",
    ".rmvb": "video/vnd.rn-realvideo",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".wma": "audio/x-ms-wma",
}

SIDECAR_TRACK = SubtitleTrack(id=SIDECAR_TRACK_ID, label="外挂字幕")

_SRT_TIMESTAMP = re.compile(r"(\d{2,3}:\d{2}:\d{2}),(\d{3})")
_ASS_OVERRIDE = re.compile(r"\{\\[^}]*\}")


def media_type_for_path(path: Path) -> str:
    """按后缀给出码流类型; 未知后缀按 ``video/mp4`` 声明.

    ``probe`` 与 ``resolve`` 必须声明同一种类型, 因此两处都调用本函数.
    """
    return MEDIA_TYPES.get(path.suffix.casefold(), FALLBACK_MEDIA_TYPE)


def _natural_key(item: PlaybackMediaFile) -> tuple[tuple[int, int | str], ...]:
    """文件名的自然序键: 数字段按数值比较, 其余按小写文本.

    ``BONY-080-CD1.mp4`` 排在 ``BONY-080-CD2.mp4`` 前, ``CD2`` 排在 ``CD10`` 前.
    """
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part) for part in re.split(r"(\d+)", Path(item.path).name.casefold())
    )


def listed_candidates(query: PlaybackQuery) -> tuple[PlaybackMediaFile, ...]:
    """按文件名自然序列出候选文件.

    列表顺序同时决定「默认选中哪一条」, 因此它必须与用户对文件名的理解一致: 多片的条目按
    ``CD1`` / ``CD2`` 排列, 而不是按入库顺序 (入库顺序对用户不可见, 可能先给后半部). 文件名相同
    (或都不含数字) 时保持快照顺序, 排序稳定.
    """
    return tuple(sorted(query.files, key=_natural_key))


def preferred_candidates(query: PlaybackQuery) -> tuple[PlaybackMediaFile, ...]:
    """按偏好顺序列出候选文件, 最该播的排在最前.

    体积降序, ``size`` 缺失按 0 计; 同分按 id 兜底, 结果与快照顺序无关. 顺序只用于「没有指定文件
    时先试哪一个」, 不影响列表顺序.
    """
    ranked = sorted(listed_candidates(query), key=lambda item: (item.size or 0, item.id))
    return tuple(reversed(ranked))


def candidate_by_key(query: PlaybackQuery, key: str) -> PlaybackMediaFile | None:
    """按流的 key 找出该条目里的文件; 该条目没有这个 key 时返回 ``None``.

    本插件的 key 就是入库文件 id 的字符串形式, 因此用户选中的文件可以直接定位回快照条目.
    """
    return next((item for item in query.files if str(item.id) == key), None)


def is_strm_path(value: str) -> bool:
    """判断路径是否为 ``.strm`` 清单文件 (大小写不敏感).

    只按后缀判断, 不读取正文: 里面记录的是本机路径还是上游地址都不影响结论. 清单不是媒体,
    打开它只会得到文本, 因此它是一条带原因的不可播候选, 不做过滤.
    """
    return Path(value).suffix.casefold() == STRM_SUFFIX


def regular_file_size(path: Path) -> int | None:
    """返回普通文件的大小; 路径不存在, 不是普通文件或无法 stat 时返回 ``None``.

    只执行一次 ``stat``: 探测预算内不允许打开或读取媒体正文. 该函数阻塞所在线程, 钩子内只
    允许经 ``asyncio.to_thread`` 调用 (见模块 docstring).
    """
    try:
        info = path.stat()
    except OSError, ValueError:
        # ValueError: 路径含 NUL 等无法传给系统调用的字节.
        return None
    return info.st_size if stat.S_ISREG(info.st_mode) else None


def unplayable_detail(path: Path) -> str | None:
    """返回该文件不可播放的原因; 可播放时返回 ``None``.

    长度为 0 的文件是占位或残缺文件: 主机对它发出的任何 ``Range`` 都不可满足, 探测放行只会在
    播放时变成 416. ``.strm`` 清单是文本, 主机按码流输出只会得到这段文本, 因此同样在这里拒绝 ——
    清单记录了什么地址与本插件无关, 正文不被读取, 也不被解析. 原因里只写文件名 —— 它会直接进入
    播放源列表. 该函数阻塞所在线程, 钩子内只允许经 ``asyncio.to_thread`` 调用 (见模块 docstring).
    """
    size = regular_file_size(path)
    if size is None:
        return f"该条目索引的文件不存在: {path.name}"
    if is_strm_path(str(path)):
        return f"该条目索引的文件是 .strm 清单文件: {path.name}"
    if size == 0:
        return f"该条目索引的文件为空: {path.name}"
    return None


def decode_subtitle(raw: bytes) -> str | None:
    """按 UTF-8 与 GBK 顺序解码字幕正文.

    两种编码都失败时返回 ``None``. 不允许回退到不会失败的编码 (latin-1 之类): 那会把乱码当作
    正文交给浏览器, 用户看到的是错字而不是「无法读取」, 探测阶段也无从判断轨道是否可用.
    """
    for encoding in SUBTITLE_ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def read_sidecar(path: Path) -> str | None:
    """读取并解码一条字幕文件; 不存在, 超限或解码失败时返回 ``None``.

    该函数阻塞所在线程, 钩子内只允许经 ``asyncio.to_thread`` 调用 (见模块 docstring).
    """
    size = regular_file_size(path)
    if size is None or size > MAX_SIDECAR_BYTES:
        return None
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    return decode_subtitle(raw)


def sidecar_candidates(media_path: Path, suffixes: tuple[str, ...]) -> tuple[Path, ...]:
    """列出同目录字幕候选, 顺序即优先级.

    入库字面路径与解析后路径指向同一个文件时只保留字面路径. 两者不同 (符号链接, ``..``) 时
    两侧都列出, 字面路径优先: 用户按入库路径摆放的字幕应当胜出.
    """
    bases: list[Path] = []
    for base in (media_path, media_path.resolve()):
        if base not in bases:
            bases.append(base)
    return tuple(base.with_suffix(suffix) for base in bases for suffix in suffixes)


def find_sidecar(media_path: Path, suffixes: tuple[str, ...]) -> Path | None:
    """返回第一条可解码的同目录字幕; 没有则返回 ``None``.

    探测阶段就要判定可解码: 先声明轨道再在加载时报错, 会让用户看到一条读取必定失败的轨道.
    候选按顺序核对, 整组候选只允许提交一次线程切换 (见 ``_find_sidecar``); 该函数阻塞所在
    线程.
    """
    for candidate in sidecar_candidates(media_path, suffixes):
        if read_sidecar(candidate) is not None:
            return candidate
    return None


def strip_ass_overrides(text: str) -> str:
    """删除 ASS/SSA 覆盖代码 (``{\\an8}``, ``{\\pos(192,210)}``, ``{\\i1}`` 之类).

    只匹配花括号内以反斜杠开头的片段, 正文里的花括号文字 (``{note}``) 必须保留. 覆盖代码被
    删除后仅剩空白的行按空行处理: 只含空白的正文行在浏览器端渲染为一条空字幕, 仍会占据
    时间轴, 与「该行没有内容」的语义不符.
    """
    cleaned: list[str] = []
    for line in text.split("\n"):
        if "{\\" not in line:
            cleaned.append(line)
            continue
        without_codes = _ASS_OVERRIDE.sub("", line)
        cleaned.append("" if not without_codes.strip() else without_codes)
    return "\n".join(cleaned)


def to_webvtt(text: str, *, suffix: str) -> str:
    """把字幕正文转成 WebVTT.

    ``.srt`` 的时间戳用逗号分隔毫秒, 且允许三位小时数, 两者在浏览器端都是非法时间戳, 会导致
    整条字幕被丢弃, 因此必须改写. 覆盖代码的清洗在分支之前完成, 两条路径才会都生效.
    """
    body = strip_ass_overrides(text.lstrip("\ufeff"))
    lowered = suffix.casefold()
    if lowered == ".vtt":
        if body.lstrip().upper().startswith("WEBVTT"):
            return body
        return f"WEBVTT\n\n{body}"
    converted = _SRT_TIMESTAMP.sub(r"\1.\2", body)
    return f"WEBVTT\n\n{converted}"


class LocalFileProvider(PlaybackProvider):
    """声明条目已索引的本地文件由主机输出, 转换同目录字幕, 并在没有可播流时说明原因."""

    def __init__(
        self,
        *,
        sidecar_subtitles: bool = True,
        sidecar_suffixes: tuple[str, ...] = DEFAULT_SIDECAR_SUFFIXES,
    ) -> None:
        self._sidecar_subtitles = sidecar_subtitles
        self._sidecar_suffixes = sidecar_suffixes

    @override
    async def probe(self, query: PlaybackQuery) -> tuple[PlaybackOffer, ...]:
        """把该条目已索引的每个候选列成一条流; 不可播的候选也列出来并说明原因.

        列出候选而不是只列能播的那一个: 用户要看得到「这个文件在库里」以及它为什么暂时播不了,
        否则一个坏文件会像凭空消失. 顺序即文件名自然序, 默认选中的是第一条可播的.

        代价与候选数成正比: 每个候选一次 ``stat``, 每个可播候选再按后缀找一次同目录字幕, 因此
        条目文件很多、库又在网络挂载上时, 这一趟会吃掉整段探测预算. 对比不做候选数上限, 因为
        条目里的文件数是用户自己的库决定的.
        """
        candidates = listed_candidates(query)
        if not candidates:
            raise SourceError(FailureReason.NO_USABLE_METADATA, detail=NO_FILE_DETAIL)
        offers: list[PlaybackOffer] = []
        for item in candidates:
            path = Path(item.path)
            content_type = media_type_for_path(path)
            detail = await asyncio.to_thread(unplayable_detail, path)
            if detail is not None:
                offers.append(
                    PlaybackOffer(
                        key=str(item.id),
                        name=path.name,
                        content_type=content_type,
                        seekable=False,
                        unavailable=detail,
                    )
                )
                continue
            sidecar = await self._find_sidecar(path)
            offers.append(
                PlaybackOffer(
                    key=str(item.id),
                    name=path.name,
                    content_type=content_type,
                    subtitles=(SIDECAR_TRACK,) if sidecar is not None else (),
                )
            )
        return tuple(offers)

    @override
    async def resolve(self, query: PlaybackQuery) -> FilePlaybackTarget:
        chosen = await self._playable(query)
        path = Path(chosen.path)
        return FilePlaybackTarget(path=path, content_type=media_type_for_path(path))

    @override
    async def subtitle(self, query: PlaybackQuery, track_id: str) -> str | UpstreamPlaybackTarget | None:
        if track_id != SIDECAR_TRACK_ID:
            return None
        chosen = await self._playable(query)
        sidecar = await self._find_sidecar(Path(chosen.path))
        if sidecar is None:
            return None
        text = await asyncio.to_thread(read_sidecar, sidecar)
        if text is None:
            return None
        return to_webvtt(text, suffix=sidecar.suffix)

    async def _playable(self, query: PlaybackQuery) -> PlaybackMediaFile:
        """给出要执行的那一个文件并确认它此刻可播放; 没有时抛 ``SourceError``.

        没有可播流是 ``SourceError(NO_USABLE_METADATA)`` 而不是空结果: 空结果不带原因, 用户无从
        判断是文件没了还是插件坏了. 原因里只说文件名与怎么了, 不写完整路径.

        指定了 key 就只认它 —— 用户明确选中的文件不允许被换成另一个, 播不了就报它的原因. 没有指定
        时逐个核对偏好顺序里的候选, 取第一个可播放的: 占位文件、``.strm`` 清单与被删掉的文件留在索引
        里是常态, 它们不该挡掉同一条目里其它可播的文件; 全都不可播时报本该播的那一个的原因.
        ``stat`` 经 ``asyncio.to_thread`` 提交 (见模块 docstring).
        """
        if query.selected_key is not None:
            chosen = candidate_by_key(query, query.selected_key)
            if chosen is None:
                raise SourceError(FailureReason.NO_USABLE_METADATA, detail="所选文件不在该条目的索引中")
            detail = await asyncio.to_thread(unplayable_detail, Path(chosen.path))
            if detail is not None:
                raise SourceError(FailureReason.NO_USABLE_METADATA, detail=detail)
            return chosen
        candidates = preferred_candidates(query)
        if not candidates:
            raise SourceError(FailureReason.NO_USABLE_METADATA, detail=NO_FILE_DETAIL)
        preferred, *alternatives = candidates
        detail = await asyncio.to_thread(unplayable_detail, Path(preferred.path))
        if detail is None:
            return preferred
        for alternative in alternatives:
            if await asyncio.to_thread(unplayable_detail, Path(alternative.path)) is None:
                return alternative
        raise SourceError(FailureReason.NO_USABLE_METADATA, detail=detail)

    async def _find_sidecar(self, media_path: Path) -> Path | None:
        """返回第一条可解码的同目录字幕, 整组候选在同一次线程切换内按顺序核对.

        候选顺序即优先级. 不允许把候选拆成多次线程切换, 也不允许并发提交: 前者使每条候选各自
        排队, 后者使结果取决于完成顺序, 两者都会改变「字面路径优先」的取舍.
        """
        if not self._sidecar_subtitles or not self._sidecar_suffixes:
            return None
        return await asyncio.to_thread(find_sidecar, media_path, self._sidecar_suffixes)
