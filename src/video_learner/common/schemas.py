"""持久化数据使用整数微秒，保留媒体的原始时间信息。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Track(Record):
    file: str
    index: int
    kind: Literal["video", "audio"]
    codec: str
    time_base: str
    start_pts: int
    start_us: int
    duration_us: int
    prefix_bytes: int = 0
    width: int | None = None
    height: int | None = None
    rate: str | None = None
    sample_rate: int | None = None
    channels: int | None = None


class Source(Record):
    schema_version: int = 1
    title: str
    adapter: Literal["local", "bilibili-cache"]
    tracks: list[Track]
    duration_us: int
    origin_us: int
    subtitles: list[str] = Field(default_factory=list)
    metadata_complete: bool | None = None
    opened: bool = True
    sampled_decode: bool = False
    full_verified: bool = False
    diagnostics: list[str] = Field(default_factory=list)
    bvid: str | None = None


class TranscriptSegment(Record):
    id: str
    start_us: int = Field(ge=0)
    end_us: int = Field(gt=0)
    text: str
    origin: Literal["asr", "srt", "vtt"]
    alignment: Literal["sentence", "audio_window"] = "sentence"
    chunk_id: str | None = None
    raw_start_us: int | None = None
    raw_end_us: int | None = None


class FrameEvidence(Record):
    id: str
    at_us: int = Field(ge=0)
    requested_us: int = Field(ge=0)
    pts: int
    time_base: str
    origin_us: int
    path: str
    original_path: str
    crop: tuple[int, int, int, int] | None = None
    parent_id: str | None = None
    sha256: str | None = None
    original_sha256: str | None = None


class ReviewItem(Record):
    reason: str
    start_us: int
    end_us: int
    block_id: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class DraftBlock(Record):
    kind: Literal["text", "figure"]
    body: str = Field(max_length=20000)
    category: Literal["original", "ai_addition", "uncertain"]
    evidence_ids: list[str]
    frame_id: str | None


class Draft(Record):
    title: str = Field(min_length=1, max_length=200)
    blocks: list[DraftBlock] = Field(min_length=1, max_length=100)
    review: list[str] = Field(max_length=100)


class NoteBlock(DraftBlock):
    id: str
    chapter_id: str
    sync_status: Literal["generated", "manual_unverified"] = "generated"


class Chapter(Record):
    id: str
    title: str
    start_us: int
    end_us: int
    status: Literal["pending", "completed", "failed"] = "pending"
    blocks: list[NoteBlock] = Field(default_factory=list)


class Notebook(Record):
    schema_version: int = 1
    title: str
    start_us: int
    end_us: int
    source: Source
    chapters: list[Chapter]
    transcript: list[TranscriptSegment]
    frames: list[FrameEvidence]
    review: list[ReviewItem] = Field(default_factory=list)
    sync_status: Literal["generated", "manual_unverified"] = "generated"
