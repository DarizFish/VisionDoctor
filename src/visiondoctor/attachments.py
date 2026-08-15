from __future__ import annotations

import io
import json
import stat
import threading
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

import pypdf.filters
from pypdf import PdfReader
from pypdf.errors import PyPdfError


class AttachmentAnalysisError(ValueError):
    """An attachment cannot be analyzed completely within the safe contract."""


@dataclass(frozen=True)
class ExtractedAttachment:
    text: str
    metadata: dict[str, Any]


_MAX_EXTRACTED_CHARS = 40_000
_MAX_JSON_DEPTH = 64
_MAX_JSON_NODES = 100_000
_MAX_PDF_PAGES = 100
_MAX_PDF_STREAM_OUTPUT_BYTES = 20 * 1024 * 1024
_MAX_ZIP_ENTRIES = 100
_MAX_ZIP_ENTRY_BYTES = 4 * 1024 * 1024
_MAX_ZIP_TOTAL_BYTES = 20 * 1024 * 1024
_MAX_ZIP_RATIO = 100
_MAX_SKIPPED_NAMES = 20
_PDF_LOCK = threading.Lock()
_ZIP_TEXT_SUFFIXES = {
    ".c",
    ".cc",
    ".cfg",
    ".conf",
    ".cpp",
    ".csv",
    ".h",
    ".hpp",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".log",
    ".md",
    ".properties",
    ".ps1",
    ".py",
    ".sh",
    ".toml",
    ".ts",
    ".tsv",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}


def analyze_attachment(name: str, media_type: str, payload: bytes) -> ExtractedAttachment:
    if media_type == "text/plain":
        text = _decode_text(payload, name=name)
        kind = "log" if PurePosixPath(name.lower()).suffix == ".log" else "text"
        return _finish(
            text,
            kind=kind,
            description_prefix="已读取文本附件",
            limitations=[],
        )
    if media_type == "application/json":
        return _analyze_json(payload, name=name)
    if media_type == "application/pdf":
        return _analyze_pdf(payload, name=name)
    if media_type == "application/zip":
        return _analyze_zip(payload, name=name)
    raise AttachmentAnalysisError(f"附件 {name} 没有可用的内容分析器")


def _decode_text(payload: bytes, *, name: str) -> str:
    encoding = (
        "utf-16"
        if payload.startswith((b"\xff\xfe", b"\xfe\xff"))
        else "utf-8-sig"
    )
    try:
        text = payload.decode(encoding, errors="strict")
    except UnicodeDecodeError as exc:
        raise AttachmentAnalysisError(
            f"附件 {name} 不是 UTF-8 或带字节序标记的 UTF-16 文本"
        ) from exc
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if "\x00" in text:
        raise AttachmentAnalysisError(f"附件 {name} 含有二进制空字符，不能按文本分析")
    if not text.strip():
        raise AttachmentAnalysisError(f"附件 {name} 没有可分析的文字内容")
    return text


def _analyze_json(payload: bytes, *, name: str) -> ExtractedAttachment:
    canonical = _canonical_json(payload, name=name)
    return _finish(
        canonical,
        kind="json",
        description_prefix="已校验并读取 JSON",
        limitations=[],
    )


def _canonical_json(payload: bytes, *, name: str) -> str:
    text = _decode_text(payload, name=name)

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AttachmentAnalysisError(f"附件 {name} 的 JSON 含有重复字段：{key}")
            result[key] = value
        return result

    try:
        value = json.loads(text, object_pairs_hook=reject_duplicate_keys)
    except AttachmentAnalysisError:
        raise
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise AttachmentAnalysisError(f"附件 {name} 不是有效 JSON") from exc
    _validate_json_size(value, name=name)
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _validate_json_size(value: Any, *, name: str) -> None:
    count = 0
    stack: list[tuple[Any, int]] = [(value, 1)]
    while stack:
        current, depth = stack.pop()
        count += 1
        if count > _MAX_JSON_NODES:
            raise AttachmentAnalysisError(f"附件 {name} 的 JSON 节点数量超过安全上限")
        if depth > _MAX_JSON_DEPTH:
            raise AttachmentAnalysisError(f"附件 {name} 的 JSON 嵌套层级超过安全上限")
        if isinstance(current, dict):
            stack.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            stack.extend((item, depth + 1) for item in current)


def _analyze_pdf(payload: bytes, *, name: str) -> ExtractedAttachment:
    if b"%PDF-" not in payload[:1024]:
        raise AttachmentAnalysisError(f"附件 {name} 的内容与 PDF 类型不匹配")
    try:
        with _PDF_LOCK:
            previous_limit = pypdf.filters.ZLIB_MAX_OUTPUT_LENGTH
            pypdf.filters.ZLIB_MAX_OUTPUT_LENGTH = _MAX_PDF_STREAM_OUTPUT_BYTES
            try:
                reader = PdfReader(io.BytesIO(payload), strict=True)
                if reader.is_encrypted:
                    raise AttachmentAnalysisError(f"附件 {name} 已加密，不能读取内容")
                page_count = len(reader.pages)
                if page_count == 0:
                    raise AttachmentAnalysisError(f"附件 {name} 没有页面")
                if page_count > _MAX_PDF_PAGES:
                    raise AttachmentAnalysisError(
                        f"附件 {name} 超过 {_MAX_PDF_PAGES} 页的安全上限"
                    )
                pages: list[str] = []
                for number, page in enumerate(reader.pages, start=1):
                    page_text = (page.extract_text() or "").strip()
                    if page_text:
                        pages.append(f"===== 第 {number} 页 =====\n{page_text}")
            finally:
                pypdf.filters.ZLIB_MAX_OUTPUT_LENGTH = previous_limit
    except AttachmentAnalysisError:
        raise
    except (PyPdfError, OSError, RuntimeError, ValueError, TypeError) as exc:
        raise AttachmentAnalysisError(f"附件 {name} 无法安全解析为 PDF") from exc
    if not pages:
        raise AttachmentAnalysisError(
            f"附件 {name} 没有可搜索文字；请将需要查看的页面导出为图片"
        )
    return _finish(
        "\n\n".join(pages),
        kind="pdf",
        description_prefix=f"已从 {page_count} 页 PDF 中提取可搜索文字",
        limitations=["只读取可搜索文字，不分析页面图形、扫描图像或精确版式关系"],
        extra={"page_count": page_count},
    )


def _analyze_zip(payload: bytes, *, name: str) -> ExtractedAttachment:
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload), mode="r", allowZip64=False)
    except (zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise AttachmentAnalysisError(f"附件 {name} 不是有效 ZIP 压缩包") from exc
    with archive:
        entries = archive.infolist()
        if len(entries) > _MAX_ZIP_ENTRIES:
            raise AttachmentAnalysisError(
                f"附件 {name} 超过 {_MAX_ZIP_ENTRIES} 个条目的安全上限"
            )
        total_size = 0
        seen_paths: set[str] = set()
        supported: list[zipfile.ZipInfo] = []
        skipped: list[str] = []
        for entry in entries:
            normalized = _safe_zip_path(entry.filename, archive_name=name)
            path_identity = normalized.casefold()
            if path_identity in seen_paths:
                raise AttachmentAnalysisError(f"附件 {name} 含有重复路径：{normalized}")
            seen_paths.add(path_identity)
            if entry.flag_bits & 0x1:
                raise AttachmentAnalysisError(f"附件 {name} 含有加密条目，不能分析")
            unix_mode = entry.external_attr >> 16
            if unix_mode and stat.S_ISLNK(unix_mode):
                raise AttachmentAnalysisError(f"附件 {name} 含有符号链接，不能分析")
            if entry.is_dir():
                continue
            total_size += entry.file_size
            if total_size > _MAX_ZIP_TOTAL_BYTES:
                raise AttachmentAnalysisError(f"附件 {name} 解压后总大小超过 20 MB")
            if entry.file_size > _MAX_ZIP_ENTRY_BYTES:
                raise AttachmentAnalysisError(
                    f"附件 {name} 中的 {normalized} 解压后超过 4 MB"
                )
            ratio = entry.file_size / max(entry.compress_size, 1)
            if entry.file_size > 1_000_000 and ratio > _MAX_ZIP_RATIO:
                raise AttachmentAnalysisError(f"附件 {name} 的压缩比例超过安全上限")
            suffix = PurePosixPath(normalized.lower()).suffix
            if suffix in _ZIP_TEXT_SUFFIXES:
                supported.append(entry)
            else:
                skipped.append(normalized)
        if not supported:
            raise AttachmentAnalysisError(f"附件 {name} 中没有支持的文本文件")
        sections: list[str] = []
        for entry in supported:
            normalized = _safe_zip_path(entry.filename, archive_name=name)
            try:
                with archive.open(entry, mode="r") as stream:
                    content = stream.read(_MAX_ZIP_ENTRY_BYTES + 1)
            except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
                raise AttachmentAnalysisError(
                    f"附件 {name} 中的 {normalized} 无法安全读取"
                ) from exc
            if len(content) > _MAX_ZIP_ENTRY_BYTES or len(content) != entry.file_size:
                raise AttachmentAnalysisError(
                    f"附件 {name} 中的 {normalized} 解压大小不可信"
                )
            if PurePosixPath(normalized.lower()).suffix == ".json":
                extracted = _canonical_json(content, name=f"{name}/{normalized}")
            else:
                extracted = _decode_text(content, name=f"{name}/{normalized}")
            sections.append(f"===== {normalized} =====\n{extracted}")
    limitations: list[str] = []
    if skipped:
        visible_skipped = skipped[:_MAX_SKIPPED_NAMES]
        suffix = "" if len(skipped) <= _MAX_SKIPPED_NAMES else " 等"
        limitations.append(
            "未读取不支持的压缩包条目：" + "、".join(visible_skipped) + suffix
        )
    return _finish(
        "\n\n".join(sections),
        kind="zip",
        description_prefix=f"已读取压缩包中的 {len(supported)} 个文本文件",
        limitations=limitations,
        force_partial=bool(skipped),
        extra={
            "entry_count": len(entries),
            "analyzed_entry_count": len(supported),
            "skipped_entry_count": len(skipped),
        },
    )


def _safe_zip_path(value: str, *, archive_name: str) -> str:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or normalized.startswith("/")
        or ":" in normalized
        or ".." in path.parts
        or "\x00" in normalized
    ):
        raise AttachmentAnalysisError(f"附件 {archive_name} 含有不安全路径")
    return path.as_posix()


def _finish(
    text: str,
    *,
    kind: str,
    description_prefix: str,
    limitations: list[str],
    force_partial: bool = False,
    extra: dict[str, Any] | None = None,
) -> ExtractedAttachment:
    excerpt, total_characters, truncated = _bounded_excerpt(text)
    all_limitations = list(limitations)
    if truncated:
        omitted = total_characters - len(excerpt)
        all_limitations.append(
            f"内容超过单附件分析上限，已保留开头和结尾，省略 {omitted} 个字符"
        )
    metadata: dict[str, Any] = {
        "status": "partial" if force_partial or truncated else "ready",
        "kind": kind,
        "description": f"{description_prefix}，纳入诊断 {len(excerpt)} 个字符",
        "total_characters": total_characters,
        "included_characters": len(excerpt),
        "truncated": truncated,
        "limitations": all_limitations,
    }
    metadata.update(extra or {})
    return ExtractedAttachment(text=excerpt, metadata=metadata)


def _bounded_excerpt(text: str) -> tuple[str, int, bool]:
    total = len(text)
    if total <= _MAX_EXTRACTED_CHARS:
        return text, total, False
    marker = "\n\n[中间内容因长度限制已省略]\n\n"
    available = _MAX_EXTRACTED_CHARS - len(marker)
    head = available * 2 // 3
    tail = available - head
    return text[:head] + marker + text[-tail:], total, True
