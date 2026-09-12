"""选片, 磁盘检查, 同目录字幕与 WebVTT 转换."""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections.abc import Awaitable
from pathlib import Path

import pytest
from amane.plugin import (
    ContentType,
    FailureReason,
    FilePlaybackTarget,
    PlaybackMediaFile,
    PlaybackQuery,
    SourceError,
)

import local
from local import (
    DEFAULT_SIDECAR_SUFFIXES,
    MAX_SIDECAR_BYTES,
    SIDECAR_TRACK_ID,
    LocalFileProvider,
    candidate_by_key,
    decode_subtitle,
    find_sidecar,
    is_strm_path,
    listed_candidates,
    media_type_for_path,
    preferred_candidates,
    read_sidecar,
    strip_ass_overrides,
    to_webvtt,
    unplayable_detail,
)

_SRT = "1\n00:00:01,000 --> 00:00:04,500\nHello\n\n"
_SRT_CHINESE = "1\n00:00:01,000 --> 00:00:04,500\n中文字幕\n"
_VTT = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nHi\n"
_UNDECODABLE = b"\xff\xfe\x9c\x80\x81"
_BLOCKING_SECONDS = 0.2
_HEARTBEAT_INTERVAL = 0.01


def write(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def media_file(path: Path, file_id: int, *, size: int | None = None) -> PlaybackMediaFile:
    return PlaybackMediaFile(
        id=file_id,
        path=str(path),
        size=size,
        content_type=ContentType.CENSORED,
        library_id=1,
        library_path=str(path.parent),
    )


def query(files: tuple[PlaybackMediaFile, ...], selected: str | None = None) -> PlaybackQuery:
    """``selected`` 是流的 key: 本插件的 key 即入库文件 id 的字符串形式."""
    return PlaybackQuery(metadata_id=1, number="ABC-001", files=files, selected_key=selected)


@pytest.mark.parametrize(
    ("names", "sizes", "expected_ids"),
    [
        (("a.mp4", "b.mp4"), (10, 20), (2, 1)),
        (("a.mp4", "b.mp4"), (None, 5), (2, 1)),
        (("a.mp4", "b.mp4"), (None, 0), (2, 1)),
        (("a.mp4", "b.mp4", "c.mp4"), (5, 5, 5), (3, 2, 1)),
        # ``.strm`` 不做过滤, 参与排序的方式与其它文件一致.
        (("a.strm", "b.mp4"), (20, 10), (1, 2)),
        ((), (), ()),
    ],
)
def test_preferred_candidates(
    tmp_path: Path,
    names: tuple[str, ...],
    sizes: tuple[int | None, ...],
    expected_ids: tuple[int, ...],
) -> None:
    """候选按体积降序, ``size`` 缺失按 0 计, 同分按 id 兜底.

    顺序只决定「没有指定文件时先试哪一个」.
    """
    files = tuple(
        media_file(write(tmp_path / name), index, size=size)
        for index, (name, size) in enumerate(zip(names, sizes, strict=True), start=1)
    )
    ranked = preferred_candidates(query(files))
    assert tuple(item.id for item in ranked) == expected_ids


@pytest.mark.parametrize(
    ("names", "expected_names"),
    [
        # 多片条目: CD1 在 CD2 前, 与入库顺序 (用例里按给出顺序递增) 相反.
        (("BONY-080-CD2.mp4", "BONY-080-CD1.mp4"), ("BONY-080-CD1.mp4", "BONY-080-CD2.mp4")),
        # 数字段按数值比较, 不是按文本.
        (("movie10.mp4", "movie2.mp4"), ("movie2.mp4", "movie10.mp4")),
        (("b.mp4", "a.mp4"), ("a.mp4", "b.mp4")),
        # ``.strm`` 同样进入列表: 它是一条带原因的不可播候选, 不再被隐藏.
        (("a.strm", "b.mp4"), ("a.strm", "b.mp4")),
        # 文件名相同 (或都不含数字) 时保持快照顺序: 排序稳定.
        (("same.mp4", "same.mp4"), ("same.mp4", "same.mp4")),
    ],
)
def test_listed_candidates_is_natural_name_order(
    tmp_path: Path,
    names: tuple[str, ...],
    expected_names: tuple[str, ...],
) -> None:
    """列表按文件名自然序, 与体积和入库顺序无关: 它决定默认选中哪一条, 必须与用户的读法一致."""
    files = tuple(
        media_file(write(tmp_path / name), index, size=100 - index) for index, name in enumerate(names, start=1)
    )
    listed = listed_candidates(query(files))
    assert tuple(Path(item.path).name for item in listed) == expected_names


@pytest.mark.parametrize(
    ("files", "key", "expected_name"),
    [
        ((("a.mp4", 1),), "1", "a.mp4"),
        ((("a.mp4", 1), ("b.mp4", 2)), "2", "b.mp4"),
        ((("a.mp4", 1),), "9", None),
    ],
)
def test_candidate_by_key(
    tmp_path: Path,
    files: tuple[tuple[str, int], ...],
    key: str,
    expected_name: str | None,
) -> None:
    """key 即入库文件 id 的字符串形式; 该条目没有这个 key 时返回 ``None``."""
    payload = query(tuple(media_file(write(tmp_path / name), file_id) for name, file_id in files))
    chosen = candidate_by_key(payload, key)
    assert (None if chosen is None else Path(chosen.path).name) == expected_name


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (b"x", None),
        (b"", "该条目索引的文件为空: movie.mp4"),
        (None, "该条目索引的文件不存在: movie.mp4"),
    ],
)
def test_unplayable_detail(tmp_path: Path, content: bytes | None, expected: str | None) -> None:
    """空文件与缺失文件各给出原因; 目录不算可播放的文件."""
    path = tmp_path / "movie.mp4"
    if content is not None:
        write(path, content)
    assert unplayable_detail(path) == expected


def test_unplayable_detail_rejects_directory(tmp_path: Path) -> None:
    (tmp_path / "movie.mp4").mkdir()
    assert unplayable_detail(tmp_path / "movie.mp4") == "该条目索引的文件不存在: movie.mp4"


@pytest.mark.parametrize(
    ("name", "content", "expected"),
    [
        # 清单不是媒体: 正文里写的是本机路径还是上游地址都不影响结论, 正文不被读取.
        ("movie.strm", b"/library/other.mp4", "该条目索引的文件是 .strm 清单文件: movie.strm"),
        ("movie.STRM", b"https://example.test/x.m3u8", "该条目索引的文件是 .strm 清单文件: movie.STRM"),
        ("movie.strm", b"", "该条目索引的文件是 .strm 清单文件: movie.strm"),
        # 文件已从磁盘消失时先报消失: 这比「它是清单」更值得用户处理.
        ("gone.strm", None, "该条目索引的文件不存在: gone.strm"),
    ],
)
def test_unplayable_detail_rejects_strm(tmp_path: Path, name: str, content: bytes | None, expected: str) -> None:
    """``.strm`` 清单与空文件、缺失文件一样, 是一条带原因的不可播候选."""
    path = tmp_path / name
    if content is not None:
        write(path, content)
    assert unplayable_detail(path) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("movie.mp4", "video/mp4"),
        ("movie.MKV", "video/x-matroska"),
        ("movie.ts", "video/mp2t"),
        ("movie.rmvb", "video/vnd.rn-realvideo"),
        ("track.mp3", "audio/mpeg"),
        ("track.FLAC", "audio/flac"),
        # 清单类型不能被声明为码流: file 目标由主机按字面路径输出, 不是 HLS.
        ("playlist.m3u8", "video/mp4"),
        ("manifest.mpd", "video/mp4"),
        ("movie.strm", "video/mp4"),
        ("no-extension", "video/mp4"),
    ],
)
def test_media_type_for_path(name: str, expected: str) -> None:
    assert media_type_for_path(Path("/library") / name) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [("a.strm", True), ("a.STRM", True), ("/x/y.STRM", True), ("a.mp4", False), ("strm", False), ("", False)],
)
def test_is_strm_path(value: str, expected: bool) -> None:
    assert is_strm_path(value) is expected


@pytest.mark.asyncio
async def test_probe_and_resolve_existing_file(tmp_path: Path) -> None:
    """命中的文件按快照里的字面路径回送, 探测与解析声明同一种类型."""
    video = write(tmp_path / "movie.mp3", b"0123456789")
    item = media_file(video, 7)
    provider = LocalFileProvider()
    offers = await provider.probe(query((item,)))
    assert len(offers) == 1
    offer = offers[0]
    assert offer.key == "7"
    assert offer.name == video.name
    assert offer.content_type == "audio/mpeg"
    assert offer.seekable is True
    assert offer.unavailable is None
    assert offer.subtitles == ()

    target = await provider.resolve(query((item,)))
    assert isinstance(target, FilePlaybackTarget)
    assert target.path == video
    assert target.content_type == offer.content_type


@pytest.mark.asyncio
async def test_resolve_keeps_literal_snapshot_path(tmp_path: Path) -> None:
    """符号链接照常可播: 回送的仍是快照里的字面路径, 主机按条目索引逐条核对."""
    real = write(tmp_path / "real" / "movie.mp4", b"x")
    link = tmp_path / "link" / "movie.mp4"
    link.parent.mkdir()
    link.symlink_to(real)

    target = await LocalFileProvider().resolve(query((media_file(link, 3),)))
    assert isinstance(target, FilePlaybackTarget)
    assert target.path == link
    assert target.path.resolve() == real.resolve()


def _record_blocking_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, threading.Thread]]:
    """记录 ``local`` 内阻塞辅助函数的调用线程与次数.

    ``check`` (逐候选核对) 与 ``find`` (找字幕) 是钩子级的入口, 次数可以精确断言; ``stat`` 与
    ``read`` 是它们内部的调用 (每份字幕候选各有一次), 只用于断言线程归属.
    """
    seen: list[tuple[str, threading.Thread]] = []
    real_check = local.unplayable_detail
    real_size = local.regular_file_size
    real_read = local.read_sidecar
    real_find = local.find_sidecar

    def traced_check(path: Path) -> str | None:
        seen.append(("check", threading.current_thread()))
        return real_check(path)

    def traced_size(path: Path) -> int | None:
        seen.append(("stat", threading.current_thread()))
        return real_size(path)

    def traced_read(path: Path) -> str | None:
        seen.append(("read", threading.current_thread()))
        return real_read(path)

    def traced_find(media_path: Path, suffixes: tuple[str, ...]) -> Path | None:
        seen.append(("find", threading.current_thread()))
        return real_find(media_path, suffixes)

    monkeypatch.setattr(local, "unplayable_detail", traced_check)
    monkeypatch.setattr(local, "regular_file_size", traced_size)
    monkeypatch.setattr(local, "read_sidecar", traced_read)
    monkeypatch.setattr(local, "find_sidecar", traced_find)
    return seen


def _hook_calls(seen: list[tuple[str, threading.Thread]]) -> list[str]:
    return [label for label, _ in seen if label in {"check", "find"}]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("hook", "expected"),
    [("probe", ["check", "find"]), ("resolve", ["check"]), ("subtitle", ["check", "find"])],
)
async def test_blocking_calls_run_off_event_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hook: str,
    expected: tuple[str, ...],
) -> None:
    """三个钩子内的阻塞文件系统调用必须落在工作线程.

    主机在事件循环上直接调用 ``probe`` / ``resolve`` / ``subtitle``; 调用留在事件循环线程上
    时, 整个服务端在该调用返回之前停止推进, 媒体库位于网络挂载时其它来源的探测一并超时.
    """
    video = write(tmp_path / "movie.mp4")
    write(tmp_path / "movie.srt", _SRT.encode("utf-8"))
    provider = LocalFileProvider()
    payload = query((media_file(video, 1),))
    loop_thread = threading.current_thread()
    seen = _record_blocking_calls(monkeypatch)

    if hook == "probe":
        assert await provider.probe(payload) != ()
    elif hook == "resolve":
        assert await provider.resolve(payload) is not None
    elif hook == "subtitle":
        assert await provider.subtitle(payload, SIDECAR_TRACK_ID) is not None
    else:
        raise AssertionError(hook)

    assert _hook_calls(seen) == expected
    assert [label for label, thread in seen if thread is loop_thread] == []


@pytest.mark.asyncio
async def test_probe_costs_one_stat_per_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """探测代价与候选数成正比: 每个候选一次 ``stat``, 不可播的候选不再找字幕.

    这是探测预算下唯一可依赖的约束, 因此在这里钉住次数, 而不是只断言「调用发生在工作线程」.
    """
    playable = write(tmp_path / "movie.mp4")
    write(tmp_path / "movie.srt", _SRT.encode("utf-8"))
    placeholder = write(tmp_path / "placeholder.mp4", b"")
    payload = query((media_file(playable, 1), media_file(placeholder, 2)))
    seen = _record_blocking_calls(monkeypatch)

    offers = await LocalFileProvider().probe(payload)
    assert [offer.key for offer in offers] == ["1", "2"]
    assert _hook_calls(seen) == ["check", "find", "check"]


@pytest.mark.asyncio
async def test_blocking_stat_does_not_stall_event_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """探测期间的阻塞 ``stat`` 不允许暂停事件循环.

    心跳协程在阻塞窗口内的推进次数即证据: 调用留在事件循环线程上时, 心跳在整个阻塞窗口内
    无法恢复执行.
    """
    video = write(tmp_path / "movie.mp4")
    real_size = local.regular_file_size

    def slow_size(path: Path) -> int | None:
        time.sleep(_BLOCKING_SECONDS)
        return real_size(path)

    monkeypatch.setattr(local, "regular_file_size", slow_size)
    ticks = 0

    async def heartbeat() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(_HEARTBEAT_INTERVAL)
            ticks += 1

    pump = asyncio.create_task(heartbeat())
    try:
        offers = await LocalFileProvider(sidecar_subtitles=False).probe(query((media_file(video, 1),)))
    finally:
        pump.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pump

    assert offers != ()
    assert ticks >= 3


def _no_stream_query(tmp_path: Path, case: str) -> PlaybackQuery:
    if case == "no-files":
        return query(())
    if case == "missing":
        return query((media_file(tmp_path / "gone.mp4", 1),))
    if case == "directory":
        (tmp_path / "movie.mp4").mkdir(parents=True)
        return query((media_file(tmp_path / "movie.mp4", 1),))
    if case == "empty":
        return query((media_file(write(tmp_path / "movie.mp4", b""), 1),))
    if case == "selected-elsewhere":
        return query((media_file(write(tmp_path / "movie.mp4"), 1),), selected="99")
    if case == "selected-strm":
        item = media_file(write(tmp_path / "movie.strm", b"https://example.test/x.m3u8"), 1)
        return query((item,), selected="1")
    if case == "selected-empty":
        return query((media_file(write(tmp_path / "movie.mp4", b""), 1),), selected="1")
    if case == "selected-missing":
        return query((media_file(tmp_path / "gone.mp4", 1),), selected="1")
    if case == "nul-byte":
        # 含 NUL 的路径无法传给系统调用, ``stat`` 抛 ``ValueError``.
        return query((media_file(tmp_path / "na\x00me.mp4", 1),))
    raise AssertionError(case)


# 整个来源都没有东西可列时抛 ``SourceError``; 有候选但都不可播时改为逐条列出并说明.
_NO_CANDIDATE_CASES = [
    ("no-files", "该条目没有已索引的文件"),
]

_SELECTED_CASES = [
    ("selected-elsewhere", "所选文件不在该条目的索引中"),
    ("selected-strm", "该条目索引的文件是 .strm 清单文件: movie.strm"),
    ("selected-empty", "该条目索引的文件为空: movie.mp4"),
    ("selected-missing", "该条目索引的文件不存在: gone.mp4"),
    ("nul-byte", "该条目索引的文件不存在: na\x00me.mp4"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("case", "detail"), _NO_CANDIDATE_CASES)
async def test_no_candidate_reports_reason(tmp_path: Path, case: str, detail: str) -> None:
    """一个候选都没有时三个钩子都报同一条原因, 而不是不带原因的空结果.

    原因直接进入播放源列表, 用户据此判断是文件没了、是空文件, 还是选错了文件.
    """
    provider = LocalFileProvider()
    payload = _no_stream_query(tmp_path, case)

    async def check(call: Awaitable[object]) -> None:
        with pytest.raises(SourceError) as excinfo:
            await call
        assert excinfo.value.reason is FailureReason.NO_USABLE_METADATA
        assert excinfo.value.detail == detail

    await check(provider.probe(payload))
    await check(provider.resolve(payload))
    await check(provider.subtitle(payload, SIDECAR_TRACK_ID))


@pytest.mark.asyncio
@pytest.mark.parametrize(("case", "detail"), _SELECTED_CASES)
async def test_selected_candidate_is_never_replaced(tmp_path: Path, case: str, detail: str) -> None:
    """用户明确选中的文件播不了就报它的原因, 不允许改用同条目里其它文件."""
    provider = LocalFileProvider()
    payload = _no_stream_query(tmp_path, case)

    async def check(call: Awaitable[object]) -> None:
        with pytest.raises(SourceError) as excinfo:
            await call
        assert excinfo.value.detail == detail

    await check(provider.resolve(payload))
    await check(provider.subtitle(payload, SIDECAR_TRACK_ID))


@pytest.mark.asyncio
async def test_probe_lists_every_candidate(tmp_path: Path) -> None:
    """每个候选一条流: 可播的带字幕与类型, 不可播的照常列出并说明原因.

    坏文件不该像凭空消失 —— 用户要看得到它在库里、以及它为什么暂时播不了.
    """
    playable = write(tmp_path / "movie.mp4", b"0123456789")
    write(tmp_path / "movie.srt", _SRT.encode("utf-8"))
    placeholder = write(tmp_path / "placeholder.mp4", b"")
    manifest = write(tmp_path / "manifest.strm", b"/library/other.mp4")
    files = (
        media_file(playable, 1),
        media_file(placeholder, 2),
        media_file(tmp_path / "gone.mp4", 3),
        media_file(manifest, 4),
    )

    offers = await LocalFileProvider().probe(query(files))
    # 列表顺序 = 文件名自然序 (gone < manifest < movie < placeholder), 与入库顺序无关.
    assert [(offer.key, offer.name, offer.unavailable) for offer in offers] == [
        ("3", "gone.mp4", "该条目索引的文件不存在: gone.mp4"),
        ("4", "manifest.strm", "该条目索引的文件是 .strm 清单文件: manifest.strm"),
        ("1", "movie.mp4", None),
        ("2", "placeholder.mp4", "该条目索引的文件为空: placeholder.mp4"),
    ]
    assert offers[0].seekable is False
    assert offers[1].seekable is False
    assert [track.id for track in offers[2].subtitles] == [SIDECAR_TRACK_ID]


@pytest.mark.asyncio
async def test_resolve_picks_preferred_playable_candidate(tmp_path: Path) -> None:
    """没有指定 key 时逐个核对偏好顺序, 取第一个可播放的.

    ``size`` 缺失时偏好顺序按 id 兜底, 因此可播的文件排在最后才轮得到它.
    """
    playable = write(tmp_path / "movie.mp4", b"0123456789")
    placeholder = write(tmp_path / "placeholder.mp4", b"")
    files = (media_file(tmp_path / "gone.mp4", 3), media_file(placeholder, 2), media_file(playable, 1))
    target = await LocalFileProvider().resolve(query(files))
    assert target.path == playable


@pytest.mark.asyncio
async def test_resolve_honours_selected_key(tmp_path: Path) -> None:
    """指定了 key 就播那一个, 即使偏好顺序里还有别的可播文件."""
    first = write(tmp_path / "movie.mp4", b"0123456789")
    second = write(tmp_path / "other.mp4", b"0123456789")
    files = (media_file(first, 1), media_file(second, 2))
    target = await LocalFileProvider().resolve(query(files, selected="1"))
    assert target.path == first
    assert (await LocalFileProvider().resolve(query(files))).path == second


@pytest.mark.asyncio
async def test_all_candidates_unplayable_reports_preferred(tmp_path: Path) -> None:
    """全部不可播放时报本该播的那一个的原因: 偏好顺序里第一个."""
    write(tmp_path / "placeholder.mp4", b"")
    payload = query((media_file(tmp_path / "gone.mp4", 1), media_file(tmp_path / "placeholder.mp4", 2)))
    with pytest.raises(SourceError) as excinfo:
        await LocalFileProvider().resolve(payload)
    assert excinfo.value.detail == "该条目索引的文件为空: placeholder.mp4"


@pytest.mark.asyncio
async def test_strm_is_listed_as_unplayable(tmp_path: Path) -> None:
    """``.strm`` 不做过滤: 它是一条带原因的不可播候选, 正文记录的地址不会被打开.

    长度不为 0 的普通文件在探测阶段本来是「可播」的, 因此这条原因必须由后缀判定给出, 否则用户
    点下去才失败, 主机按码流输出的只会是清单文本.
    """
    item = media_file(write(tmp_path / "movie.strm", b"/library/other.mp4"), 1)
    provider = LocalFileProvider()
    offers = await provider.probe(query((item,)))
    assert [(offer.key, offer.name, offer.unavailable) for offer in offers] == [
        ("1", "movie.strm", "该条目索引的文件是 .strm 清单文件: movie.strm"),
    ]
    with pytest.raises(SourceError) as excinfo:
        await provider.resolve(query((item,)))
    assert excinfo.value.detail == "该条目索引的文件是 .strm 清单文件: movie.strm"


@pytest.mark.asyncio
async def test_strm_does_not_block_playable_candidates(tmp_path: Path) -> None:
    """同条目里有可播文件时, ``.strm`` 不挡掉它: 没有指定 key 时逐个核对偏好顺序."""
    manifest = write(tmp_path / "movie.strm", b"/library/other.mp4")
    playable = write(tmp_path / "movie-cd2.mp4", b"0123456789")
    files = (media_file(manifest, 1, size=999), media_file(playable, 2, size=1))
    assert (await LocalFileProvider().resolve(query(files))).path == playable


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sidecar", "payload", "declared", "expected_body"),
    [
        ("movie.srt", _SRT.encode("utf-8"), True, "00:00:01.000 --> 00:00:04.500"),
        ("movie.srt", _SRT_CHINESE.encode("gbk"), True, "中文字幕"),
        ("movie.vtt", _VTT.encode("utf-8"), True, "WEBVTT"),
        ("movie.srt", _UNDECODABLE, False, None),
        # 只有同 stem 的字幕算外挂: ``movie.zh.srt`` 属于别的文件.
        ("movie.zh.srt", _SRT.encode("utf-8"), False, None),
        ("other.srt", _SRT.encode("utf-8"), False, None),
    ],
)
async def test_sidecar_track(
    tmp_path: Path,
    sidecar: str,
    payload: bytes,
    declared: bool,
    expected_body: str | None,
) -> None:
    """探测阶段就判定字幕能否解码: 读不出来的文件不声明轨道, 也不返回正文."""
    write(tmp_path / sidecar, payload)
    item = media_file(write(tmp_path / "movie.mp4"), 1)
    provider = LocalFileProvider()
    offers = await provider.probe(query((item,)))
    assert len(offers) == 1
    offer = offers[0]
    if not declared:
        assert offer.subtitles == ()
        assert await provider.subtitle(query((item,)), SIDECAR_TRACK_ID) is None
        return
    assert [track.id for track in offer.subtitles] == [SIDECAR_TRACK_ID]
    text = await provider.subtitle(query((item,)), SIDECAR_TRACK_ID)
    assert isinstance(text, str)
    assert text.startswith("WEBVTT")
    assert expected_body is not None
    assert expected_body in text


@pytest.mark.asyncio
async def test_sidecar_without_media_file_is_absent(tmp_path: Path) -> None:
    """媒体文件已从磁盘消失时连字幕一起不可用: 那条流不可播, 也不声明轨道."""
    write(tmp_path / "movie.srt", _SRT.encode("utf-8"))
    item = media_file(tmp_path / "movie.mp4", 1)
    provider = LocalFileProvider()
    offers = await provider.probe(query((item,)))
    assert len(offers) == 1
    assert offers[0].unavailable == "该条目索引的文件不存在: movie.mp4"
    assert offers[0].subtitles == ()
    with pytest.raises(SourceError):
        await provider.subtitle(query((item,)), SIDECAR_TRACK_ID)


@pytest.mark.asyncio
async def test_unknown_track_id_returns_none(tmp_path: Path) -> None:
    write(tmp_path / "movie.srt", _SRT.encode("utf-8"))
    item = media_file(write(tmp_path / "movie.mp4"), 1)
    assert await LocalFileProvider().subtitle(query((item,)), "embedded") is None


@pytest.mark.asyncio
async def test_sidecar_directory_is_not_a_track(tmp_path: Path) -> None:
    """与媒体同名的目录不算字幕: 声明轨道之后读取必然失败, 因此探测阶段就不声明."""
    (tmp_path / "movie.srt").mkdir(parents=True)
    item = media_file(write(tmp_path / "movie.mp4"), 1)
    provider = LocalFileProvider()

    offers = await provider.probe(query((item,)))
    assert [offer.subtitles for offer in offers] == [()]
    assert await provider.subtitle(query((item,)), SIDECAR_TRACK_ID) is None


@pytest.mark.asyncio
async def test_provider_prefers_literal_path_sidecar(tmp_path: Path) -> None:
    """钩子侧同样按字面路径优先返回字幕: 整组候选在同一次线程切换内按顺序核对."""
    real = tmp_path / "real"
    other = tmp_path / "other"
    target = write(real / "movie.mp4")
    other.mkdir()
    (other / "movie.mp4").symlink_to(target)
    write(other / "movie.srt", b"1\n00:00:01,000 --> 00:00:02,000\nliteral\n")
    write(real / "movie.srt", b"1\n00:00:01,000 --> 00:00:02,000\nresolved\n")
    item = media_file(other / "movie.mp4", 1)
    provider = LocalFileProvider()

    offers = await provider.probe(query((item,)))
    assert [[track.id for track in offer.subtitles] for offer in offers] == [[SIDECAR_TRACK_ID]]
    text = await provider.subtitle(query((item,)), SIDECAR_TRACK_ID)
    assert isinstance(text, str)
    assert "literal" in text
    assert "resolved" not in text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("suffixes", "declared"),
    [
        (DEFAULT_SIDECAR_SUFFIXES, True),
        ((".vtt",), False),
        ((), False),
    ],
)
async def test_sidecar_suffix_whitelist(tmp_path: Path, suffixes: tuple[str, ...], declared: bool) -> None:
    write(tmp_path / "movie.srt", _SRT.encode("utf-8"))
    item = media_file(write(tmp_path / "movie.mp4"), 1)
    provider = LocalFileProvider(sidecar_suffixes=suffixes)
    offers = await provider.probe(query((item,)))
    assert [bool(offer.subtitles) for offer in offers] == [declared]


@pytest.mark.asyncio
async def test_sidecar_subtitles_can_be_disabled(tmp_path: Path) -> None:
    write(tmp_path / "movie.srt", _SRT.encode("utf-8"))
    item = media_file(write(tmp_path / "movie.mp4"), 1)
    provider = LocalFileProvider(sidecar_subtitles=False)
    offers = await provider.probe(query((item,)))
    assert [offer.subtitles for offer in offers] == [()]
    assert await provider.subtitle(query((item,)), SIDECAR_TRACK_ID) is None


def test_oversized_sidecar_is_not_declared(tmp_path: Path) -> None:
    """命名成字幕的超大文件不进入探测: 读取它会占用整段探测预算."""
    oversized = write(tmp_path / "movie.srt", b"x" * (MAX_SIDECAR_BYTES + 1))
    assert read_sidecar(oversized) is None
    assert find_sidecar(write(tmp_path / "movie.mp4"), DEFAULT_SIDECAR_SUFFIXES) is None


def test_literal_directory_sidecar_wins(tmp_path: Path) -> None:
    """入库字面路径与解析后路径不同时, 字面路径所在目录的字幕优先."""
    real = tmp_path / "real"
    other = tmp_path / "other"
    target = write(real / "movie.mp4")
    other.mkdir()
    (other / "movie.mp4").symlink_to(target)
    write(other / "movie.srt", b"1\n00:00:01,000 --> 00:00:02,000\nliteral\n")
    write(real / "movie.srt", b"1\n00:00:01,000 --> 00:00:02,000\nresolved\n")

    assert find_sidecar(other / "movie.mp4", DEFAULT_SIDECAR_SUFFIXES) == other / "movie.srt"


def test_resolved_directory_sidecar_is_found(tmp_path: Path) -> None:
    """字面路径所在目录没有字幕时, 继续检查解析后路径所在目录."""
    real = tmp_path / "real"
    other = tmp_path / "other"
    target = write(real / "movie.mp4")
    other.mkdir()
    (other / "movie.mp4").symlink_to(target)
    resolved = write(real / "movie.srt", _SRT.encode("utf-8"))

    assert find_sidecar(other / "movie.mp4", DEFAULT_SIDECAR_SUFFIXES) == resolved


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("中文字幕".encode("gbk"), "中文字幕"),
        ("中文字幕".encode(), "中文字幕"),
        ("\ufeff中文字幕".encode(), "中文字幕"),
        (b"WEBVTT\n", "WEBVTT\n"),
        (_UNDECODABLE, None),
        (b"", ""),
    ],
)
def test_decode_subtitle(raw: bytes, expected: str | None) -> None:
    assert decode_subtitle(raw) == expected


@pytest.mark.parametrize(
    ("text", "suffix", "expected"),
    [
        (_VTT, ".vtt", _VTT),
        ("00:00:01.000 --> 00:00:02.000\nHi\n", ".vtt", "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nHi\n"),
        ("\ufeffWEBVTT\n\n00:00:01.000 --> 00:00:02.000\nHi\n", ".vtt", _VTT),
        (
            "1\n00:00:01,000 --> 00:00:04,500\nHello\n\n",
            ".srt",
            "WEBVTT\n\n1\n00:00:01.000 --> 00:00:04.500\nHello\n\n",
        ),
        # SRT 允许三位小时数; 逗号与三位小时数在浏览器端都是非法时间戳.
        (
            "1\n100:00:01,000 --> 100:00:04,500\nHello\n\n",
            ".srt",
            "WEBVTT\n\n1\n100:00:01.000 --> 100:00:04.500\nHello\n\n",
        ),
    ],
)
def test_to_webvtt(text: str, suffix: str, expected: str) -> None:
    assert to_webvtt(text, suffix=suffix) == expected


def test_to_webvtt_removes_srt_commas() -> None:
    """逗号时间戳在浏览器端非法, 转换后正文里不允许再出现逗号."""
    assert "," not in to_webvtt("1\n00:00:01,000 --> 00:00:04,500\nHello\n", suffix=".srt")


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("{\\an8}", ""),
        ("{\\an8}顶部对齐", "顶部对齐"),
        ("顶部对齐{\\an8}", "顶部对齐"),
        ("前段{\\an8}后段", "前段后段"),
        ("{\\an8}   ", ""),
        ("{\\i1}斜体{\\i0}与{\\pos(192,210)}定位", "斜体与定位"),
        ("{note}", "{note}"),
        ("{\\an8}{note}", "{note}"),
        ("花括号文字 {中文} 保留", "花括号文字 {中文} 保留"),
        ("未闭合{\\an8 保留原文", "未闭合{\\an8 保留原文"),
    ],
)
def test_strip_ass_overrides(body: str, expected: str) -> None:
    """只删除花括号内以反斜杠开头的片段, 正文里的花括号文字必须保留."""
    assert strip_ass_overrides(body) == expected


@pytest.mark.parametrize(
    ("text", "suffix", "expected"),
    [
        (
            "1\n00:00:01,000 --> 00:00:04,500\n{\\an8}顶部\n",
            ".srt",
            "WEBVTT\n\n1\n00:00:01.000 --> 00:00:04.500\n顶部\n",
        ),
        (
            "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n{\\an8}Hi\n",
            ".vtt",
            "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nHi\n",
        ),
        ("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n{\\an8}\n", ".vtt", "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n\n"),
    ],
)
def test_to_webvtt_converts_and_cleans(text: str, suffix: str, expected: str) -> None:
    """已带 WEBVTT 头的 ``.vtt`` 原样返回, 覆盖代码的清洗同样必须生效."""
    assert to_webvtt(text, suffix=suffix) == expected
