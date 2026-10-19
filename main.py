"""
Gemini Live Audio Backend Server.

A FastAPI-based WebSocket server that proxies audio between clients and
Google's Gemini Live API for real-time voice interactions.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
import logging
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, AsyncGenerator

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

if TYPE_CHECKING:
    from collections.abc import MutableMapping


# =============================================================================
# CONFIGURATION
# =============================================================================

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    google_api_key: str = Field(..., description="Google API key for Gemini access")
    gemini_model: str = Field(
        default="gemini-2.5-flash-native-audio-preview-12-2025",
        description="Gemini model identifier"
    )
    sample_rate: int = Field(default=24000, ge=8000, le=48000)
    silence_duration_ms: int = Field(default=50, ge=50, le=2000)
    generation_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    text_model: str = Field(
        default="gemini-2.5-flash",
        description="Text model used to generate lesson content"
    )
    instruction_file: Path = Field(default=Path("instructions.txt"))
    phrases_file: Path = Field(default=Path("phrases.json"))
    static_dir: Path = Field(default=Path("static"))

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


settings = Settings()


# =============================================================================
# VOICE OPTIONS
# =============================================================================

@dataclass(frozen=True)
class VoiceOption:
    """Represents a Gemini voice option."""
    name: str
    gender: str
    style: str


AVAILABLE_VOICES: tuple[VoiceOption, ...] = (
    VoiceOption("Zephyr", "Female", "Bright"),
    VoiceOption("Puck", "Male", "Upbeat"),
    VoiceOption("Charon", "Male", "Informative"),
    VoiceOption("Kore", "Female", "Firm"),
    VoiceOption("Fenrir", "Male", "Excitable"),
    VoiceOption("Leda", "Female", "Youthful"),
    VoiceOption("Orus", "Male", "Firm"),
    VoiceOption("Aoede", "Female", "Breezy"),
    VoiceOption("Callirrhoe", "Female", "Easy-going"),
    VoiceOption("Autonoe", "Female", "Bright"),
    VoiceOption("Enceladus", "Male", "Breathy"),
    VoiceOption("Iapetus", "Male", "Clear"),
    VoiceOption("Umbriel", "Male", "Easy-going"),
    VoiceOption("Algieba", "Male", "Smooth"),
    VoiceOption("Despina", "Female", "Smooth"),
    VoiceOption("Erinome", "Female", "Clear"),
    VoiceOption("Algenib", "Male", "Gravelly"),
    VoiceOption("Achernar", "Female", "Soft"),
    VoiceOption("Achird", "Male", "Friendly"),
    VoiceOption("Gacrux", "Male", "Mature"),
    VoiceOption("Laomedeia", "Female", "Upbeat"),
    VoiceOption("Sadachbia", "Female", "Lively"),
    VoiceOption("Sadaltager", "Male", "Knowledgeable"),
    VoiceOption("Schedar", "Female", "Even"),
    VoiceOption("Zubenelgenubi", "Male", "Casual"),
)

VOICE_NAME_SET: frozenset[str] = frozenset(v.name for v in AVAILABLE_VOICES)


# =============================================================================
# STRUCTURED LOGGING
# =============================================================================

class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add session_id if present
        if hasattr(record, "session_id"):
            log_data["session_id"] = record.session_id
        
        # Add extra fields if present
        if hasattr(record, "extra"):
            log_data.update(record.extra)
        
        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


def setup_logging() -> logging.Logger:
    """Configure structured JSON logging."""
    logger = logging.getLogger("GeminiLiveBackend")
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler with JSON formatting
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    
    return logger


logger = setup_logging()


# =============================================================================
# METRICS COLLECTION
# =============================================================================

@dataclass
class Metrics:
    """In-memory metrics container."""
    
    start_time: float = field(default_factory=time.time)
    total_sessions: int = 0
    total_audio_bytes_sent: int = 0
    total_audio_bytes_received: int = 0
    total_gemini_requests: int = 0
    gemini_latencies_ms: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def record_gemini_latency(self, latency_ms: float) -> None:
        """Record a Gemini API response latency."""
        self.gemini_latencies_ms.append(latency_ms)
    
    def get_latency_stats(self) -> dict:
        """Calculate latency statistics."""
        if not self.gemini_latencies_ms:
            return {"avg_ms": 0, "p95_ms": 0, "count": 0}
        
        latencies = sorted(self.gemini_latencies_ms)
        avg = sum(latencies) / len(latencies)
        p95_idx = int(len(latencies) * 0.95)
        p95 = latencies[min(p95_idx, len(latencies) - 1)]
        
        return {
            "avg_ms": round(avg, 2),
            "p95_ms": round(p95, 2),
            "count": len(latencies)
        }
    
    def to_dict(self, active_connections: int) -> dict:
        """Export all metrics as a dictionary."""
        uptime_seconds = time.time() - self.start_time
        return {
            "uptime_seconds": round(uptime_seconds, 1),
            "active_connections": active_connections,
            "total_sessions": self.total_sessions,
            "audio": {
                "bytes_sent": self.total_audio_bytes_sent,
                "bytes_received": self.total_audio_bytes_received,
            },
            "gemini": {
                "total_requests": self.total_gemini_requests,
                "latency": self.get_latency_stats(),
            }
        }


metrics = Metrics()


# =============================================================================
# SCHOOL LESSONS (explanation / questions / story / welcome)
# =============================================================================

SEGMENT_SEPARATOR = "---"
ALLOWED_DURATIONS: tuple[int, ...] = (5, 10, 15)

# Output audio from Gemini Live is 24 kHz, 16-bit mono PCM.
OUTPUT_AUDIO_BYTES_PER_SECOND = 24000 * 2


@dataclass(frozen=True)
class LessonMode:
    """Describes one lesson activity type."""
    key: str
    label: str
    # Seconds of silence to wait (after the robot finishes speaking a part
    # we asked for) before moving on to the next part.
    gap_after_part: float
    # How the robot must behave during the whole activity (system prompt).
    behaviour: str
    # What the robot must do with each part we feed it.
    part_action: str


LESSON_MODES: dict[str, LessonMode] = {
    "explain": LessonMode(
        key="explain",
        label="شرح للمادة",
        gap_after_part=1.5,
        behaviour=(
            "- أنت معلم روبوت تشرح الدرس للطلاب شرحًا ممتعًا ومبسطًا.\n"
            "- اشرح بأسلوب منطوق حيوي، بجمل قصيرة وأمثلة من حياة الطلاب.\n"
            "- إذا ورد في الجزء سؤال تفاعلي فاطرحه على الطلاب ثم توقف لتسمع إجابتهم."
        ),
        part_action="اشرح هذا الجزء الآن بأسلوبك المنطوق الممتع، ثم توقف عن الكلام.",
    ),
    "questions": LessonMode(
        key="questions",
        label="أسئلة",
        gap_after_part=12.0,
        behaviour=(
            "- أنت تدير مسابقة أسئلة ممتعة للطلاب حول الدرس.\n"
            "- اطرح سؤالًا واحدًا في كل مرة، ولا تذكر الإجابة قبل أن يحاول الطلاب.\n"
            "- عندما يجيب طالب: إن كانت إجابته صحيحة فامدحه بحماس، وإن كانت خاطئة "
            "فشجّعه بلطف ووضّح الإجابة الصحيحة باختصار."
        ),
        part_action="اطرح هذا السؤال على الطلاب بوضوح دون ذكر الإجابة، ثم توقف وانتظر إجابتهم.",
    ),
    "story": LessonMode(
        key="story",
        label="قصة وحوار",
        gap_after_part=2.0,
        behaviour=(
            "- أنت راوٍ مبدع تحكي للطلاب قصة تعليمية فيها حوار بين شخصيات.\n"
            "- غيّر نبرة صوتك قليلًا مع كل شخصية ليكون الحوار حيًّا، ولا تقرأ أسماء "
            "المتحدثين كأنها عناوين بل اربطها بالسرد.\n"
            "- إذا انتهى الجزء بسؤال للطلاب فاطرحه ثم توقف لتسمع آراءهم."
        ),
        part_action="احكِ هذا الجزء من القصة الآن بأسلوب مشوّق، ثم توقف عن الكلام.",
    ),
    "welcome": LessonMode(
        key="welcome",
        label="ترحيب وتفاعل",
        gap_after_part=5.0,
        behaviour=(
            "- أنت روبوت مرح يرحّب بالطلاب ويتفاعل معهم ويحفّزهم على الدرس.\n"
            "- كن ودودًا ومتحمسًا، واطلب من الطلاب المشاركة والإجابة، واستمع لهم "
            "وعلّق على ما يقولونه بإيجابية."
        ),
        part_action="قدّم هذه الفقرة الآن بحماس وتفاعل مع الطلاب، ثم توقف لتسمع تفاعلهم.",
    ),
}

# Silence to wait after the robot answered a student (not a part we asked for).
GAP_AFTER_REPLY_SECONDS = 2.5
# If the model has not started speaking this long after we sent a part, resend it.
LESSON_PROMPT_TIMEOUT_SECONDS = 25.0
LESSON_PROMPT_PREFIX = "[تحكم الحصة]"


def lesson_part_count(mode: str, duration: int) -> int:
    """How many parts to generate so the activity roughly fills the duration."""
    if mode == "questions":
        return max(4, round(duration * 1.5))
    return max(3, duration)


def build_generation_prompt(mode: str, subject: str, lesson: str, duration: int) -> str:
    """Prompt for the text model that writes the lesson content."""
    parts = lesson_part_count(mode, duration)
    words = duration * 110
    sep_rule = (
        f"افصل بين كل جزء والذي يليه بسطر مستقل يحتوي فقط على {SEGMENT_SEPARATOR} "
        "ولا تضع أي ترقيم أو عناوين للأجزاء."
    )
    topic = f"درس «{lesson}» في مادة «{subject}»"

    if mode == "explain":
        return (
            f"اكتب نص شرح منطوق لـ{topic}، يلقيه روبوت معلم على طلاب المدرسة، "
            f"ومدته نحو {duration} دقائق (قرابة {words} كلمة).\n"
            f"قسّم الشرح إلى {parts} أجزاء متتابعة، كل جزء نحو دقيقة واحدة من الكلام.\n"
            "ابدأ بتمهيد قصير جذاب يثير فضول الطلاب، ثم اشرح أفكار الدرس الأساسية "
            "بالتدريج مع أمثلة من حياة الطالب اليومية، واجعل في بعض الأجزاء سؤالًا "
            "تفاعليًا قصيرًا للطلاب، واختم بملخص لأهم ما تعلموه.\n"
            + sep_rule
        )
    if mode == "questions":
        return (
            f"اكتب {parts} سؤالًا عن {topic} لمسابقة يديرها روبوت مع طلاب المدرسة "
            f"خلال نحو {duration} دقائق.\n"
            "نوّع بين أسئلة الاختيار من متعدد (اذكر الخيارات داخل نص السؤال) وأسئلة "
            "الصواب والخطأ والأسئلة القصيرة، وتدرّج من السهل إلى الأصعب.\n"
            "اكتب كل سؤال في جزء مستقل بهذا الشكل بالضبط:\n"
            "السؤال: نص السؤال\n"
            "الإجابة: الإجابة الصحيحة مع توضيح قصير جدًا\n"
            + sep_rule
        )
    if mode == "story":
        return (
            f"اكتب قصة تعليمية قصيرة فيها حوار بين شخصيات (مثل طالبين ومعلم أو "
            f"روبوت) توضّح أفكار {topic} لطلاب المدرسة، ويحكيها روبوت في نحو "
            f"{duration} دقائق (قرابة {words} كلمة).\n"
            f"قسّم القصة إلى {parts} أجزاء متتابعة، كل جزء نحو دقيقة. اكتب الحوار "
            "بذكر اسم الشخصية ثم نقطتين ثم كلامها. اجعل بعض الأجزاء تنتهي بسؤال "
            "للطلاب عن رأيهم أو عمّا سيحدث بعد ذلك، واختم القصة بالدرس المستفاد.\n"
            + sep_rule
        )
    if mode == "welcome":
        return (
            f"اكتب فقرات ترحيب وتفاعل يقدّمها روبوت لطلاب المدرسة في بداية حصة "
            f"مادة «{subject}» عن درس «{lesson}»، "
            f"لمدة نحو {duration} دقائق.\n"
            f"قسّمها إلى {parts} فقرات متتابعة، كل فقرة نحو دقيقة. ضمّنها: ترحيبًا "
            "حارًّا يعرّف فيه الروبوت بنفسه، ونشاطًا سريعًا لكسر الجمود، وأسئلة "
            "تحفيزية عن موضوع الدرس يجيب عنها الطلاب، ومعلومة مدهشة مرتبطة بالدرس، "
            "وتحديًا بسيطًا، وعبارات تشجيع، واختم بتهيئة الطلاب لبدء الدرس.\n"
            + sep_rule
        )
    raise ValueError(f"Unknown lesson mode: {mode}")


GENERATION_SYSTEM_PROMPT = (
    "أنت مصمم محتوى تعليمي خبير في المناهج المدرسية. تكتب بالعربية الفصحى "
    "المبسطة المناسبة لطلاب المدارس، بدقة علمية وأسلوب ممتع ومشجّع.\n"
    "النص سيُنطق بصوت روبوت، لذا:\n"
    "- لا تستخدم أي تنسيق Markdown (لا #، ولا **، ولا جداول، ولا رموز تعبيرية).\n"
    "- لا تكتب مقدمة أو تعليقًا خارج المحتوى المطلوب.\n"
    "- اكتب الأرقام والرموز الرياضية بطريقة يسهل نطقها."
)


def clean_generated_content(text: str) -> str:
    """Normalise model output: drop markdown and unify part separators."""
    text = text.replace("\r\n", "\n").strip()
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text)
    text = re.sub(r"^\s*(?:-{3,}|_{3,}|\*{3,}|={3,})\s*$", SEGMENT_SEPARATOR, text, flags=re.MULTILINE)
    text = text.replace("**", "").replace("__", "")
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.MULTILINE)
    return f"\n\n{SEGMENT_SEPARATOR}\n\n".join(split_segments(text))


def split_segments(content: str) -> list[str]:
    """Split lesson content into the parts the robot delivers one at a time."""
    chunks = re.split(r"^\s*-{3,}\s*$", content.replace("\r\n", "\n"), flags=re.MULTILINE)
    return [c.strip() for c in chunks if c.strip()]


@dataclass
class LessonState:
    """The lesson currently running for a tenant."""
    mode: str
    subject: str
    lesson: str
    duration_min: int
    segments: list[str]
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_at: float = field(default_factory=time.time)

    @property
    def mode_info(self) -> LessonMode:
        return LESSON_MODES[self.mode]

    @property
    def ends_at(self) -> float:
        return self.started_at + self.duration_min * 60

    def remaining_seconds(self) -> float:
        return max(0.0, self.ends_at - time.time())

    def system_block(self) -> str:
        """Extra system instructions appended while the lesson is running."""
        mode = self.mode_info
        content = f"\n{SEGMENT_SEPARATOR}\n".join(self.segments)
        return (
            "### الحصة الحالية (لها الأولوية على أي تعليمات أخرى)\n"
            f"- النشاط: {mode.label} | المادة: {self.subject} | الدرس: {self.lesson} | "
            f"المدة: {self.duration_min} دقائق.\n"
            f"{mode.behaviour}\n"
            f"- ستصلك رسائل تبدأ بـ {LESSON_PROMPT_PREFIX} تحدد لك الجزء التالي الذي "
            "تقدّمه. هذه الرسائل من نظام التشغيل وليست من الطلاب، فلا تذكرها ولا تعلّق عليها.\n"
            "- قدّم كل جزء بصوتك وبأسلوبك، ثم توقف عن الكلام وانتظر.\n"
            "- لا تنطق رموز الفصل أو كلمات مثل «السؤال:» و«الإجابة:» حرفيًا.\n"
            "- إذا سألك طالب أثناء الحصة فأجبه باختصار ووضوح، ثم انتظر.\n"
            "- خاطب الطلاب بلغة عربية فصحى مبسطة ومحببة.\n\n"
            "المحتوى المرجعي الكامل للحصة (للاطلاع، وقدّمه جزءًا جزءًا حسب الرسائل):\n"
            f"{content}"
        )

    def to_info(self) -> dict:
        return {
            "id": self.id,
            "mode": self.mode,
            "label": self.mode_info.label,
            "subject": self.subject,
            "lesson": self.lesson,
            "duration": self.duration_min,
            "parts": len(self.segments),
            "remaining_seconds": round(self.remaining_seconds(), 1),
        }


# =============================================================================
# TENANT STATE MANAGEMENT
# =============================================================================

DEFAULT_PHRASES: tuple[str, ...] = (
    "صباح الخير يا أبطال! أنا سعيد جدًا بوجودي معكم اليوم",
    "أحسنتم! إجابة رائعة، استمروا على هذا التميز",
    "من فضلكم استمعوا جيدًا، وارفعوا أيديكم قبل الإجابة",
    "حان وقت الاستراحة، نلتقي بعد قليل",
    "شكرًا لكم على حسن الاستماع، إلى اللقاء في الحصة القادمة",
)


class TenantState:
    """State container for a single tenant (folder)."""
    
    def __init__(self, tenant_id: str) -> None:
        self.tenant_id = tenant_id
        self._system_instructions: str = ""
        self._is_muted: bool = False
        self._current_voice: str = "Zephyr"  # Default voice
        self._connections: MutableMapping[WebSocket, asyncio.Queue[dict]] = {}
        self.lesson: LessonState | None = None
    
    @property
    def base_dir(self) -> Path:
        """Folder holding this tenant's files (project root for the main tenant)."""
        return Path(".") if self.tenant_id == TenantManager.DEFAULT_TENANT else Path(self.tenant_id)
    
    @property
    def instruction_file(self) -> Path:
        if self.tenant_id == TenantManager.DEFAULT_TENANT:
            return settings.instruction_file
        return self.base_dir / "instructions.txt"
    
    @property
    def phrases_file(self) -> Path:
        if self.tenant_id == TenantManager.DEFAULT_TENANT:
            return settings.phrases_file
        return self.base_dir / settings.phrases_file.name
    
    def effective_instructions(self) -> str:
        """System instructions sent to Gemini (base + running lesson, if any)."""
        if self.lesson is None:
            return self._system_instructions
        return f"{self._system_instructions}\n\n{self.lesson.system_block()}"
    
    def load_phrases(self) -> list[str]:
        """Read the quick-phrase list, falling back to the defaults."""
        try:
            data = json.loads(self.phrases_file.read_text(encoding="utf-8"))
            phrases = [str(p).strip() for p in data.get("phrases", []) if str(p).strip()]
            return phrases
        except FileNotFoundError:
            return list(DEFAULT_PHRASES)
        except Exception as e:
            logger.error(f"Failed to read phrases file: {e}")
            return list(DEFAULT_PHRASES)
    
    def save_phrases(self, phrases: list[str]) -> None:
        self.phrases_file.write_text(
            json.dumps({"phrases": phrases}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    
    @property
    def system_instructions(self) -> str:
        return self._system_instructions
    
    @system_instructions.setter
    def system_instructions(self, value: str) -> None:
        self._system_instructions = value
        logger.info("System instructions updated", extra={"extra": {"tenant": self.tenant_id, "length": len(value)}})
    
    @property
    def is_muted(self) -> bool:
        return self._is_muted

    @is_muted.setter
    def is_muted(self, value: bool) -> None:
        self._is_muted = value
        logger.info("Mute state changed", extra={"extra": {"tenant": self.tenant_id, "muted": value}})
    
    @property
    def current_voice(self) -> str:
        return self._current_voice
    
    @current_voice.setter
    def current_voice(self, value: str) -> None:
        if value not in VOICE_NAME_SET:
            raise ValueError(f"Invalid voice: {value}")
        self._current_voice = value
        logger.info("Voice changed", extra={"extra": {"tenant": self.tenant_id, "voice": value}})
    
    @property
    def connection_count(self) -> int:
        return len(self._connections)
    
    def register(self, websocket: WebSocket) -> asyncio.Queue[dict]:
        queue: asyncio.Queue[dict] = asyncio.Queue()
        self._connections[websocket] = queue
        metrics.total_sessions += 1
        logger.info("Connection registered", extra={"extra": {"tenant": self.tenant_id, "active": self.connection_count}})
        return queue
    
    def unregister(self, websocket: WebSocket) -> None:
        if websocket in self._connections:
            del self._connections[websocket]
            logger.info("Connection unregistered", extra={"extra": {"tenant": self.tenant_id, "active": self.connection_count}})
    
    async def broadcast(self, command: dict) -> None:
        for queue in self._connections.values():
            await queue.put(command)
    
    def load_instructions_from_file(self, path: Path) -> None:
        """Load system instructions from file."""
        if not path.exists():
            raise FileNotFoundError(f"Instruction file not found: {path}")
        
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            raise ValueError(f"Instruction file is empty: {path}")
        
        self._system_instructions = content
        logger.info("Loaded instructions from file", extra={"extra": {"tenant": self.tenant_id, "path": str(path), "length": len(content)}})


class TenantManager:
    """Manages all tenant states."""
    
    DEFAULT_TENANT = "main"
    
    def __init__(self) -> None:
        self._tenants: dict[str, TenantState] = {}
        self._genai_client: genai.Client | None = None
    
    @property
    def genai_client(self) -> genai.Client:
        """Lazy-initialized singleton Gemini client (shared across all tenants)."""
        if self._genai_client is None:
            self._genai_client = genai.Client(api_key=settings.google_api_key)
            logger.info("Initialized Gemini client")
        return self._genai_client
    
    def get_tenant(self, tenant_id: str) -> TenantState:
        """Get or create a tenant state."""
        if tenant_id not in self._tenants:
            self._tenants[tenant_id] = TenantState(tenant_id)
            self._load_tenant_instructions(tenant_id)
        return self._tenants[tenant_id]
    
    def _load_tenant_instructions(self, tenant_id: str) -> None:
        """Load instructions for a tenant from file."""
        tenant = self._tenants[tenant_id]
        
        if tenant_id == self.DEFAULT_TENANT:
            # Main tenant uses root instructions.txt
            instruction_path = settings.instruction_file
        else:
            # Other tenants use {folder}/instructions.txt
            instruction_path = Path(tenant_id) / "instructions.txt"
        
        if instruction_path.exists():
            tenant.load_instructions_from_file(instruction_path)
        else:
            # Fall back to main instructions
            if settings.instruction_file.exists():
                tenant.load_instructions_from_file(settings.instruction_file)
                logger.info(f"Tenant {tenant_id} using main instructions (no {instruction_path})")
    
    def get_all_connection_count(self) -> int:
        """Get total connections across all tenants."""
        return sum(t.connection_count for t in self._tenants.values())
    
    def is_valid_tenant(self, tenant_id: str) -> bool:
        """Check if a tenant folder exists."""
        if tenant_id == self.DEFAULT_TENANT:
            return True
        folder = Path(tenant_id)
        return folder.is_dir() and not tenant_id.startswith(".") and tenant_id not in ("venv", "__pycache__", "static")


tenant_manager = TenantManager()


# Backward compatibility: app_state points to main tenant
@property
def _get_app_state_instructions() -> str:
    return tenant_manager.get_tenant(TenantManager.DEFAULT_TENANT).system_instructions


class ApplicationState:
    """Backward-compatible application state (delegates to main tenant)."""
    
    @property
    def system_instructions(self) -> str:
        return tenant_manager.get_tenant(TenantManager.DEFAULT_TENANT).system_instructions
    
    @system_instructions.setter
    def system_instructions(self, value: str) -> None:
        tenant_manager.get_tenant(TenantManager.DEFAULT_TENANT).system_instructions = value
    
    @property
    def is_muted(self) -> bool:
        return tenant_manager.get_tenant(TenantManager.DEFAULT_TENANT).is_muted

    @is_muted.setter
    def is_muted(self, value: bool) -> None:
        tenant_manager.get_tenant(TenantManager.DEFAULT_TENANT).is_muted = value
    
    @property
    def current_voice(self) -> str:
        return tenant_manager.get_tenant(TenantManager.DEFAULT_TENANT).current_voice
    
    @current_voice.setter
    def current_voice(self, value: str) -> None:
        tenant_manager.get_tenant(TenantManager.DEFAULT_TENANT).current_voice = value
    
    @property
    def genai_client(self) -> genai.Client:
        return tenant_manager.genai_client
    
    def load_instructions_from_file(self, path: Path) -> None:
        tenant_manager.get_tenant(TenantManager.DEFAULT_TENANT).load_instructions_from_file(path)


app_state = ApplicationState()


# =============================================================================
# CONNECTION MANAGEMENT (Backward compatible - delegates to main tenant)
# =============================================================================

class ConnectionManager:
    """Backward-compatible connection manager (delegates to main tenant)."""
    
    @property
    def connection_count(self) -> int:
        return tenant_manager.get_tenant(TenantManager.DEFAULT_TENANT).connection_count
    
    def register(self, websocket: WebSocket) -> asyncio.Queue[dict]:
        return tenant_manager.get_tenant(TenantManager.DEFAULT_TENANT).register(websocket)
    
    def unregister(self, websocket: WebSocket) -> None:
        tenant_manager.get_tenant(TenantManager.DEFAULT_TENANT).unregister(websocket)
    
    async def broadcast(self, command: dict) -> None:
        await tenant_manager.get_tenant(TenantManager.DEFAULT_TENANT).broadcast(command)


connection_manager = ConnectionManager()


# =============================================================================
# GEMINI AUDIO SESSION
# =============================================================================

class GeminiAudioSession:
    """Encapsulates a single user's audio session with Gemini Live API."""
    
    def __init__(self, websocket: WebSocket, command_queue: asyncio.Queue[dict], session_id: str, tenant: TenantState) -> None:
        self.websocket = websocket
        self.command_queue = command_queue
        self.session_id = session_id
        self.tenant = tenant
        self._client = tenant_manager.genai_client
        self._restart_event = asyncio.Event()
        self._client_disconnected = False
        self._tasks: list[asyncio.Task] = []
        self._session_start = time.time()
        # --- Lesson progress (kept across Gemini reconnects) ---
        self._lesson_id: str | None = None
        self._lesson_part_idx = 0            # next part to deliver
        self._lesson_done = False
        self._closing_sent = False
        self._next_step_at: float | None = None
        self._resume_pending = False
        self._awaiting_prompted_turn = False  # model turn we asked for is pending
        self._prompt_sent_at = 0.0
        self._in_model_turn = False
        self._reply_since_part = False       # a student-driven reply happened
        self._playback_until = 0.0           # estimated end of client playback
    
    def _log(self, level: int, msg: str, **extra) -> None:
        """Log with session correlation ID."""
        log_extra = {"session_id": self.session_id[:8]}
        if extra:
            log_extra["extra"] = extra
        logger.log(level, msg, extra=log_extra)
    
    async def run(self) -> None:
        """Main session loop with automatic reconnection on Gemini errors only."""
        while not self._client_disconnected:
            try:
                self._restart_event.clear()
                await self._run_session()
            except WebSocketDisconnect:
                self._log(logging.INFO, "Client disconnected normally")
                self._client_disconnected = True
                break
            except ExceptionGroup as eg:
                if self._is_client_disconnect(eg):
                    self._log(logging.INFO, "Client disconnected")
                    self._client_disconnected = True
                    break
                
                for exc in eg.exceptions:
                    if not isinstance(exc, (WebSocketDisconnect, asyncio.CancelledError)):
                        self._log(logging.ERROR, f"Task error: {type(exc).__name__}: {exc}")
                
                if not self._client_disconnected:
                    self._log(logging.INFO, "Reconnecting to Gemini in 1 second...")
                    await asyncio.sleep(1)
            except Exception as e:
                if self._client_disconnected:
                    break
                self._log(logging.ERROR, f"Session error: {type(e).__name__}: {e}")
                self._log(logging.INFO, "Reconnecting to Gemini in 1 second...")
                await asyncio.sleep(1)
    
    def _is_client_disconnect(self, eg: ExceptionGroup) -> bool:
        """Check if any exception in the group indicates client disconnect."""
        for exc in eg.exceptions:
            if isinstance(exc, WebSocketDisconnect):
                return True
            if isinstance(exc, RuntimeError) and "disconnect" in str(exc).lower():
                return True
        return False
    
    async def _run_session(self) -> None:
        """Execute a single Gemini session."""
        if self._client_disconnected:
            return
            
        config = self._build_config()
        self._tasks.clear()
        
        connect_start = time.time()
        async with self._client.aio.live.connect(
            model=settings.gemini_model, 
            config=config
        ) as session:
            connect_latency = (time.time() - connect_start) * 1000
            metrics.record_gemini_latency(connect_latency)
            metrics.total_gemini_requests += 1
            self._log(logging.INFO, "Connected to Gemini", latency_ms=round(connect_latency, 1))
            
            # Send current mute state to client on connect
            if self.tenant.is_muted:
                try:
                    await self.websocket.send_json({"type": "mute", "muted": True})
                except Exception:
                    pass
            
            self._sync_lesson_on_connect()
            
            async with asyncio.TaskGroup() as tg:
                self._tasks.append(tg.create_task(self._receive_audio(session)))
                self._tasks.append(tg.create_task(self._send_audio(session)))
                self._tasks.append(tg.create_task(self._handle_commands(session)))
                self._tasks.append(tg.create_task(self._lesson_driver(session)))
    
    def _build_config(self) -> types.LiveConnectConfig:
        """Build the Gemini Live connection configuration."""
        return types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self.tenant.current_voice
                    )
                )
            ),
            generation_config=types.GenerationConfig(
                temperature=settings.generation_temperature,
                thinking_config=types.ThinkingConfig(include_thoughts=False)
            ),
            system_instruction=types.Content(
                parts=[types.Part(text=self.tenant.effective_instructions())]
            ),
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    silence_duration_ms=settings.silence_duration_ms
                )
            )
        )
    
    async def _receive_audio(self, session) -> None:
        """Receive and forward audio from Gemini to the client."""
        try:
            while not self._restart_event.is_set() and not self._client_disconnected:
                turn = session.receive()
                async for response in turn:
                    if self._client_disconnected:
                        return
                    await self._process_response(response)
        except asyncio.CancelledError:
            pass
    
    async def _process_response(self, response) -> None:
        """Process a single response from Gemini."""
        if self._client_disconnected:
            return
            
        content = response.server_content
        if not content:
            return
        
        if content.model_turn:
            self._on_model_output()
            if not self.tenant.is_muted:
                for part in content.model_turn.parts:
                    if part.inline_data:
                        try:
                            data = part.inline_data.data
                            metrics.total_audio_bytes_sent += len(data)
                            self._track_playback(len(data))
                            await self.websocket.send_bytes(data)
                        except Exception:
                            self._client_disconnected = True
                            return
        
        if content.turn_complete or content.interrupted:
            self._on_turn_complete()
    
    # -------------------------------------------------------------------------
    # Lesson driver: feeds the lesson to the robot one part at a time
    # -------------------------------------------------------------------------
    
    def _active_lesson(self) -> LessonState | None:
        lesson = self.tenant.lesson
        if lesson is None or lesson.id != self._lesson_id or self._lesson_done:
            return None
        return lesson
    
    def _sync_lesson_on_connect(self) -> None:
        """Called after every (re)connection to Gemini."""
        lesson = self.tenant.lesson
        self._in_model_turn = False
        if lesson is None:
            self._lesson_id = None
            self._next_step_at = None
            return
        if lesson.id != self._lesson_id:
            # A new lesson: start from the first part.
            self._lesson_id = lesson.id
            self._lesson_part_idx = 0
            self._lesson_done = False
            self._closing_sent = False
            self._resume_pending = False
            self._reply_since_part = False
            self._playback_until = 0.0
        elif not self._lesson_done:
            # Same lesson after a dropped connection: redo the unfinished part.
            if self._awaiting_prompted_turn:
                if self._closing_sent:
                    self._closing_sent = False
                else:
                    self._lesson_part_idx = max(0, self._lesson_part_idx - 1)
            self._resume_pending = True
        else:
            return
        self._awaiting_prompted_turn = False
        self._next_step_at = max(time.time(), self._playback_until) + 0.8
    
    def _track_playback(self, num_bytes: int) -> None:
        now = time.time()
        self._playback_until = max(now, self._playback_until) + num_bytes / OUTPUT_AUDIO_BYTES_PER_SECOND
    
    def _on_model_output(self) -> None:
        if self._in_model_turn:
            return
        self._in_model_turn = True
        if not self._awaiting_prompted_turn:
            # The robot is answering a student (or an admin phrase): hold the lesson.
            self._reply_since_part = True
            self._next_step_at = None
    
    def _on_turn_complete(self) -> None:
        self._in_model_turn = False
        lesson = self._active_lesson()
        prompted = self._awaiting_prompted_turn
        self._awaiting_prompted_turn = False
        if lesson is None:
            return
        if prompted and self._closing_sent:
            self._lesson_done = True
            self._next_step_at = None
            self._log(logging.INFO, "Lesson finished")
            return
        if prompted:
            self._reply_since_part = False
            gap = lesson.mode_info.gap_after_part
        else:
            gap = GAP_AFTER_REPLY_SECONDS
        self._next_step_at = max(time.time(), self._playback_until) + gap
    
    async def _lesson_driver(self, session) -> None:
        try:
            while not self._restart_event.is_set() and not self._client_disconnected:
                await asyncio.sleep(0.25)
                lesson = self._active_lesson()
                if lesson is None or self.tenant.is_muted:
                    continue
                now = time.time()
                # Watchdog: the model never answered our prompt -> send it again.
                if (self._awaiting_prompted_turn and not self._in_model_turn
                        and now - self._prompt_sent_at > LESSON_PROMPT_TIMEOUT_SECONDS):
                    self._log(logging.WARNING, "Lesson prompt timed out, resending")
                    self._awaiting_prompted_turn = False
                    if self._closing_sent:
                        self._closing_sent = False
                    else:
                        self._lesson_part_idx = max(0, self._lesson_part_idx - 1)
                    self._next_step_at = now
                if self._next_step_at is None or now < self._next_step_at or self._in_model_turn:
                    continue
                self._next_step_at = None
                await self._send_lesson_step(session, lesson)
        except asyncio.CancelledError:
            pass
    
    def _build_lesson_step(self, lesson: LessonState) -> str:
        mode = lesson.mode_info
        total = len(lesson.segments)
        idx = self._lesson_part_idx
        unanswered = (
            mode.key == "questions" and idx > 0 and not self._reply_since_part
            and not self._resume_pending
        )
        lead = []
        if self._resume_pending:
            lead.append("انقطع الاتصال لحظيًا. تابع الحصة بسلاسة دون أن تذكر الانقطاع.")
        if unanswered:
            lead.append("لم تصلك إجابة على السؤال السابق: اذكر إجابته الصحيحة باختصار وبأسلوب مشجّع أولًا.")
        
        time_up = lesson.remaining_seconds() <= 0
        if idx >= total or time_up:
            self._closing_sent = True
            closing = (
                "انتهى وقت الحصة. " if time_up and idx < total else "انتهت أجزاء الحصة. "
            )
            if mode.key == "questions":
                closing += "أعلن انتهاء المسابقة، وامدح مشاركة الطلاب، وودّعهم بلطف."
            elif mode.key == "welcome":
                closing += "اختم بعبارة تحفيزية قصيرة تهيئ الطلاب لبدء الدرس."
            else:
                closing += "اختم الآن بملخص قصير جدًا لأهم الأفكار، وكلمة تشجيع، ووداع لطيف."
            return f"{LESSON_PROMPT_PREFIX} " + " ".join(lead + [closing])
        
        part = lesson.segments[idx]
        self._lesson_part_idx += 1
        remaining_min = max(1, round(lesson.remaining_seconds() / 60))
        if idx == 0 and not self._resume_pending:
            opening = "ابدأ الحصة الآن مباشرة دون مقدمات عن نفسك إلا إذا طلب الجزء ذلك."
        else:
            opening = f"الوقت المتبقي نحو {remaining_min} دقيقة."
        header = f"الجزء {idx + 1} من {total}."
        return (
            f"{LESSON_PROMPT_PREFIX} " + " ".join(lead + [opening, header, mode.part_action])
            + f"\n\n{part}"
        )
    
    async def _send_lesson_step(self, session, lesson: LessonState) -> None:
        prompt = self._build_lesson_step(lesson)
        self._resume_pending = False
        self._awaiting_prompted_turn = True
        self._prompt_sent_at = time.time()
        self._log(logging.INFO, "Lesson step", part=self._lesson_part_idx, closing=self._closing_sent)
        await session.send_client_content(
            turns=[types.Content(role="user", parts=[types.Part(text=prompt)])],
            turn_complete=True,
        )
    
    async def _send_audio(self, session) -> None:
        """Forward audio from the client to Gemini."""
        try:
            while not self._restart_event.is_set() and not self._client_disconnected:
                data = await self.websocket.receive_bytes()
                metrics.total_audio_bytes_received += len(data)
                await session.send_realtime_input(
                    media=types.Blob(
                        data=data, 
                        mime_type=f"audio/pcm;rate={settings.sample_rate}"
                    )
                )
        except WebSocketDisconnect:
            self._client_disconnected = True
            raise
        except asyncio.CancelledError:
            pass
    
    async def _handle_commands(self, session) -> None:
        """Process admin commands from the command queue."""
        try:
            while not self._restart_event.is_set() and not self._client_disconnected:
                command = await self._wait_for_command()
                if command is None:
                    return
                await self._execute_command(command, session)
        except asyncio.CancelledError:
            pass
    
    async def _wait_for_command(self) -> dict | None:
        """Wait for either a command or restart signal."""
        cmd_task = asyncio.create_task(self.command_queue.get())
        restart_task = asyncio.create_task(self._restart_event.wait())
        tasks = [cmd_task, restart_task]
        
        try:
            done, _ = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED
            )
            
            if restart_task in done:
                return None
            
            return cmd_task.result()
        finally:
            # Always clean up tasks, even if we're cancelled by TaskGroup
            for task in tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
    
    async def _execute_command(self, command: dict, session) -> None:
        """Execute a single admin command."""
        cmd_type = command.get("type")
        
        if cmd_type == "kill":
            await self._handle_kill_command()
        elif cmd_type == "speak":
            await self._handle_speak_command(command.get("text", ""), session)
        elif cmd_type == "mute":
            await self._handle_mute_command(command.get("muted", False))
        elif cmd_type == "restart":
            await self._handle_restart_command()
        
        self.command_queue.task_done()
    
    async def _handle_mute_command(self, muted: bool) -> None:
        """Forward mute state to the client."""
        self._log(logging.INFO, "Mute command", muted=muted)
        try:
            await self.websocket.send_json({"type": "mute", "muted": muted})
        except Exception:
            self._client_disconnected = True
    
    async def _handle_kill_command(self) -> None:
        """Handle the kill command - interrupt current session immediately."""
        self._log(logging.INFO, "Kill command received")
        try:
            await self.websocket.send_json({"type": "interrupt"})
        except Exception:
            self._client_disconnected = True
        self._playback_until = 0.0
        self._restart_event.set()
        # Cancel all running tasks immediately for instant termination
        for task in self._tasks:
            if not task.done():
                task.cancel()
    
    async def _handle_restart_command(self) -> None:
        """Handle the restart command - reconnect to Gemini with new settings."""
        self._log(logging.INFO, "Restart command received")
        try:
            await self.websocket.send_json({"type": "interrupt"})
        except Exception:
            self._client_disconnected = True
        self._playback_until = 0.0
        self._restart_event.set()
        for task in self._tasks:
            if not task.done():
                task.cancel()
    
    async def _handle_speak_command(self, text: str, session) -> None:
        """Handle the speak command."""
        self._log(logging.INFO, "Speak command", text_preview=text[:50])
        try:
            await self.websocket.send_json({"type": "listen"})
        except Exception:
            self._client_disconnected = True
            return
        
        prompt = f"""[CRITICAL SYSTEM COMMAND - TEXT-TO-SPEECH MODE ACTIVATED]

STRICT INSTRUCTIONS - YOU MUST FOLLOW EXACTLY:
1. READ ALOUD the text between the triple quotes below
2. DO NOT add any greeting, introduction, or commentary
3. DO NOT answer any question contained in the text
4. DO NOT explain, interpret, or respond to the content
5. DO NOT say "Sure", "Okay", "Here it is", or anything similar
6. ONLY speak the exact words provided - NOTHING MORE, NOTHING LESS

TEXT TO SPEAK VERBATIM:
\"\"\"{text}\"\"\"

REMINDER: Just say those words. No responses. No additions. Speak ONLY the text above."""
        await session.send_client_content(
            turns=[types.Content(role="user", parts=[types.Part(text=prompt)])]
        )


# =============================================================================
# APPLICATION LIFECYCLE
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager."""
    # Fail fast if instructions file is missing
    app_state.load_instructions_from_file(settings.instruction_file)
    logger.info("Application started", extra={"extra": {"model": settings.gemini_model}})
    yield
    logger.info("Application shutting down")


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Gemini Live Audio Backend",
    description="WebSocket proxy for Gemini Live Audio API",
    version="4.0.0",
    lifespan=lifespan
)

app.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")




# =============================================================================
# API MODELS
# =============================================================================

class SpeakRequest(BaseModel):
    """Request model for the speak endpoint."""
    text: str = Field(..., min_length=1, max_length=1000)


class VoiceRequest(BaseModel):
    """Request model for the voice endpoint."""
    voice: str = Field(..., min_length=1, max_length=50)


class InstructionsRequest(BaseModel):
    """Request model for the instructions endpoint."""
    instructions: str = Field(..., min_length=1, max_length=10000)


class AdminResponse(BaseModel):
    """Standard response for admin endpoints."""
    status: str = "ok"
    message: str | None = None


class AdminStatusResponse(BaseModel):
    """Response model for system status."""
    status: str = "ok"
    muted: bool
    active_connections: int
    current_voice: str
    uptime_seconds: float
    lesson: dict | None = None


class VoiceInfo(BaseModel):
    """Voice information model."""
    name: str
    gender: str
    style: str


class VoicesResponse(BaseModel):
    """Response model for voices list."""
    voices: list[VoiceInfo]
    current: str


class InstructionsResponse(BaseModel):
    """Response model for instructions endpoint."""
    status: str = "ok"
    instructions: str


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = "ok"


# =============================================================================
# HEALTH & METRICS ENDPOINTS
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Simple health check endpoint for deployment monitoring."""
    return HealthResponse()


@app.get("/metrics")
async def get_metrics() -> JSONResponse:
    """Get current system metrics."""
    return JSONResponse(content=metrics.to_dict(connection_manager.connection_count))


# =============================================================================
# WEBSOCKET ENDPOINTS
# =============================================================================

@app.websocket("/ws/audio")
async def audio_websocket(websocket: WebSocket) -> None:
    """WebSocket endpoint for bidirectional audio streaming (main tenant)."""
    await websocket.accept()
    session_id = str(uuid.uuid4())
    tenant = tenant_manager.get_tenant(TenantManager.DEFAULT_TENANT)
    logger.info("New WebSocket connection", extra={"session_id": session_id[:8], "extra": {"tenant": "main"}})
    
    command_queue = tenant.register(websocket)
    session = GeminiAudioSession(websocket, command_queue, session_id, tenant)
    
    try:
        await session.run()
    finally:
        tenant.unregister(websocket)
        logger.info("Session ended", extra={"session_id": session_id[:8]})


@app.websocket("/{tenant_id}/ws/audio")
async def tenant_audio_websocket(websocket: WebSocket, tenant_id: str) -> None:
    """WebSocket endpoint for bidirectional audio streaming (tenant-specific)."""
    # Validate tenant
    if not tenant_manager.is_valid_tenant(tenant_id):
        await websocket.close(code=4004, reason="Invalid tenant")
        return
    
    await websocket.accept()
    session_id = str(uuid.uuid4())
    tenant = tenant_manager.get_tenant(tenant_id)
    logger.info("New WebSocket connection", extra={"session_id": session_id[:8], "extra": {"tenant": tenant_id}})
    
    command_queue = tenant.register(websocket)
    session = GeminiAudioSession(websocket, command_queue, session_id, tenant)
    
    try:
        await session.run()
    finally:
        tenant.unregister(websocket)
        logger.info("Session ended", extra={"session_id": session_id[:8]})


# =============================================================================
# ADMIN ENDPOINTS
# =============================================================================

@app.post("/admin/kill", response_model=AdminResponse)
async def admin_kill() -> AdminResponse:
    """Interrupt all active sessions (and stop any running lesson)."""
    _main_tenant().lesson = None
    await connection_manager.broadcast({"type": "kill"})
    return AdminResponse(message=f"Kill sent to {connection_manager.connection_count} clients")


@app.post("/admin/speak", response_model=AdminResponse)
async def admin_speak(request: SpeakRequest) -> AdminResponse:
    """Make all connected robots speak the specified text."""
    await connection_manager.broadcast({"type": "speak", "text": request.text})
    return AdminResponse(message=f"Speak command sent to {connection_manager.connection_count} clients")


@app.post("/admin/mute", response_model=AdminStatusResponse)
async def admin_toggle_mute() -> AdminStatusResponse:
    """Toggle the global mute state."""
    app_state.is_muted = not app_state.is_muted
    # Broadcast mute state to all connected clients
    await connection_manager.broadcast({"type": "mute", "muted": app_state.is_muted})
    return _status_for(_main_tenant())


@app.get("/admin/status", response_model=AdminStatusResponse)
async def admin_status() -> AdminStatusResponse:
    """Get current system status."""
    return _status_for(_main_tenant())


@app.get("/admin/voices", response_model=VoicesResponse)
async def admin_get_voices() -> VoicesResponse:
    """Get all available voices."""
    return VoicesResponse(
        voices=[VoiceInfo(name=v.name, gender=v.gender, style=v.style) for v in AVAILABLE_VOICES],
        current=app_state.current_voice
    )


@app.post("/admin/voice", response_model=AdminResponse)
async def admin_set_voice(request: VoiceRequest) -> AdminResponse:
    """Set the active voice for all sessions."""
    if request.voice not in VOICE_NAME_SET:
        return AdminResponse(status="error", message=f"Invalid voice: {request.voice}")
    
    app_state.current_voice = request.voice
    # Restart all sessions to use the new voice
    await connection_manager.broadcast({"type": "restart"})
    return AdminResponse(message=f"Voice changed to {request.voice}, restarting {connection_manager.connection_count} sessions")


@app.get("/admin/instructions", response_model=InstructionsResponse)
async def admin_get_instructions() -> InstructionsResponse:
    """Get current system instructions."""
    return InstructionsResponse(instructions=app_state.system_instructions)


@app.post("/admin/instructions", response_model=AdminResponse)
async def admin_set_instructions(request: InstructionsRequest) -> AdminResponse:
    """Update system instructions for all sessions."""
    app_state.system_instructions = request.instructions
    # Save to file for persistence
    settings.instruction_file.write_text(request.instructions, encoding="utf-8")
    # Restart all sessions to use the new instructions
    await connection_manager.broadcast({"type": "restart"})
    return AdminResponse(message=f"Instructions updated, restarting {connection_manager.connection_count} sessions")


# =============================================================================
# TENANT-SPECIFIC ADMIN ENDPOINTS
# =============================================================================

def _get_tenant_or_404(tenant_id: str) -> TenantState:
    """Get tenant or raise 404."""
    from fastapi import HTTPException
    if not tenant_manager.is_valid_tenant(tenant_id):
        raise HTTPException(status_code=404, detail=f"Tenant not found: {tenant_id}")
    return tenant_manager.get_tenant(tenant_id)


@app.post("/{tenant_id}/admin/kill", response_model=AdminResponse)
async def tenant_admin_kill(tenant_id: str) -> AdminResponse:
    """Interrupt all active sessions for a specific tenant."""
    tenant = _get_tenant_or_404(tenant_id)
    tenant.lesson = None
    await tenant.broadcast({"type": "kill"})
    return AdminResponse(message=f"Kill sent to {tenant.connection_count} clients")


@app.post("/{tenant_id}/admin/speak", response_model=AdminResponse)
async def tenant_admin_speak(tenant_id: str, request: SpeakRequest) -> AdminResponse:
    """Make all connected robots speak the specified text for a specific tenant."""
    tenant = _get_tenant_or_404(tenant_id)
    await tenant.broadcast({"type": "speak", "text": request.text})
    return AdminResponse(message=f"Speak command sent to {tenant.connection_count} clients")


@app.post("/{tenant_id}/admin/mute", response_model=AdminStatusResponse)
async def tenant_admin_toggle_mute(tenant_id: str) -> AdminStatusResponse:
    """Toggle the mute state for a specific tenant."""
    tenant = _get_tenant_or_404(tenant_id)
    tenant.is_muted = not tenant.is_muted
    await tenant.broadcast({"type": "mute", "muted": tenant.is_muted})
    return _status_for(tenant)


@app.get("/{tenant_id}/admin/status", response_model=AdminStatusResponse)
async def tenant_admin_status(tenant_id: str) -> AdminStatusResponse:
    """Get current system status for a specific tenant."""
    tenant = _get_tenant_or_404(tenant_id)
    return _status_for(tenant)


@app.get("/{tenant_id}/admin/voices", response_model=VoicesResponse)
async def tenant_admin_get_voices(tenant_id: str) -> VoicesResponse:
    """Get all available voices for a specific tenant."""
    tenant = _get_tenant_or_404(tenant_id)
    return VoicesResponse(
        voices=[VoiceInfo(name=v.name, gender=v.gender, style=v.style) for v in AVAILABLE_VOICES],
        current=tenant.current_voice
    )


@app.post("/{tenant_id}/admin/voice", response_model=AdminResponse)
async def tenant_admin_set_voice(tenant_id: str, request: VoiceRequest) -> AdminResponse:
    """Set the active voice for a specific tenant."""
    tenant = _get_tenant_or_404(tenant_id)
    if request.voice not in VOICE_NAME_SET:
        return AdminResponse(status="error", message=f"Invalid voice: {request.voice}")
    
    tenant.current_voice = request.voice
    await tenant.broadcast({"type": "restart"})
    return AdminResponse(message=f"Voice changed to {request.voice}, restarting {tenant.connection_count} sessions")


@app.get("/{tenant_id}/admin/instructions", response_model=InstructionsResponse)
async def tenant_admin_get_instructions(tenant_id: str) -> InstructionsResponse:
    """Get current system instructions for a specific tenant."""
    tenant = _get_tenant_or_404(tenant_id)
    return InstructionsResponse(instructions=tenant.system_instructions)


@app.post("/{tenant_id}/admin/instructions", response_model=AdminResponse)
async def tenant_admin_set_instructions(tenant_id: str, request: InstructionsRequest) -> AdminResponse:
    """Update system instructions for a specific tenant."""
    tenant = _get_tenant_or_404(tenant_id)
    tenant.system_instructions = request.instructions
    # Save to tenant's instruction file for persistence
    tenant.instruction_file.write_text(request.instructions, encoding="utf-8")
    await tenant.broadcast({"type": "restart"})
    return AdminResponse(message=f"Instructions updated, restarting {tenant.connection_count} sessions")


# =============================================================================
# QUICK PHRASES & LESSONS (shared implementation for main + tenant routes)
# =============================================================================

class PhrasesRequest(BaseModel):
    phrases: list[str] = Field(..., max_length=200)


class PhrasesResponse(BaseModel):
    status: str = "ok"
    phrases: list[str]


class LessonGenerateRequest(BaseModel):
    mode: str
    subject: str = Field(..., min_length=1, max_length=120)
    lesson: str = Field(..., min_length=1, max_length=200)
    duration: int


class LessonGenerateResponse(BaseModel):
    status: str = "ok"
    message: str | None = None
    content: str = ""
    parts: int = 0


class LessonStartRequest(LessonGenerateRequest):
    content: str = Field(..., min_length=1, max_length=40000)


def _validate_lesson_request(req: LessonGenerateRequest) -> str | None:
    if req.mode not in LESSON_MODES:
        return "نوع النشاط غير صحيح"
    if req.duration not in ALLOWED_DURATIONS:
        return "المدة يجب أن تكون 5 أو 10 أو 15 دقيقة"
    if not req.subject.strip() or not req.lesson.strip():
        return "الرجاء كتابة اسم المادة واسم الدرس"
    return None


def _phrases_get(tenant: TenantState) -> PhrasesResponse:
    return PhrasesResponse(phrases=tenant.load_phrases())


def _phrases_save(tenant: TenantState, req: PhrasesRequest) -> PhrasesResponse:
    cleaned: list[str] = []
    for p in req.phrases:
        p = " ".join(str(p).split())
        if p and p not in cleaned:
            cleaned.append(p[:1000])
    tenant.save_phrases(cleaned)
    return PhrasesResponse(phrases=cleaned)


async def _lesson_generate(req: LessonGenerateRequest) -> LessonGenerateResponse:
    error = _validate_lesson_request(req)
    if error:
        return LessonGenerateResponse(status="error", message=error)
    prompt = build_generation_prompt(req.mode, req.subject.strip(), req.lesson.strip(), req.duration)
    try:
        response = await tenant_manager.genai_client.aio.models.generate_content(
            model=settings.text_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=GENERATION_SYSTEM_PROMPT,
                temperature=0.8,
            ),
        )
        content = clean_generated_content(response.text or "")
    except Exception as e:
        logger.error(f"Lesson generation failed: {type(e).__name__}: {e}")
        return LessonGenerateResponse(status="error", message="تعذّر توليد المحتوى، حاول مرة أخرى")
    if not content:
        return LessonGenerateResponse(status="error", message="لم يُرجع النموذج محتوى، حاول مرة أخرى")
    return LessonGenerateResponse(content=content, parts=len(split_segments(content)))


async def _lesson_start(tenant: TenantState, req: LessonStartRequest) -> AdminResponse:
    error = _validate_lesson_request(req)
    if error:
        return AdminResponse(status="error", message=error)
    segments = split_segments(req.content)
    if not segments:
        return AdminResponse(status="error", message="محتوى الحصة فارغ")
    tenant.lesson = LessonState(
        mode=req.mode,
        subject=req.subject.strip(),
        lesson=req.lesson.strip(),
        duration_min=req.duration,
        segments=segments,
    )
    logger.info("Lesson started", extra={"extra": {"tenant": tenant.tenant_id, "mode": req.mode, "parts": len(segments)}})
    # Reconnect sessions so the lesson becomes part of the system instructions.
    await tenant.broadcast({"type": "restart"})
    if tenant.connection_count == 0:
        return AdminResponse(message="تم تجهيز الحصة، وستبدأ فور اتصال الروبوت")
    return AdminResponse(message=f"بدأت الحصة على الروبوت ({tenant.connection_count} جهاز متصل)")


async def _lesson_stop(tenant: TenantState) -> AdminResponse:
    if tenant.lesson is None:
        return AdminResponse(message="لا توجد حصة جارية")
    tenant.lesson = None
    await tenant.broadcast({"type": "restart"})
    return AdminResponse(message="تم إنهاء الحصة")


def _status_for(tenant: TenantState) -> AdminStatusResponse:
    return AdminStatusResponse(
        muted=tenant.is_muted,
        active_connections=tenant.connection_count,
        current_voice=tenant.current_voice,
        uptime_seconds=round(time.time() - metrics.start_time, 1),
        lesson=tenant.lesson.to_info() if tenant.lesson else None,
    )


def _main_tenant() -> TenantState:
    return tenant_manager.get_tenant(TenantManager.DEFAULT_TENANT)


@app.get("/admin/phrases", response_model=PhrasesResponse)
async def admin_get_phrases() -> PhrasesResponse:
    return _phrases_get(_main_tenant())


@app.put("/admin/phrases", response_model=PhrasesResponse)
async def admin_save_phrases(request: PhrasesRequest) -> PhrasesResponse:
    return _phrases_save(_main_tenant(), request)


@app.post("/admin/lesson/generate", response_model=LessonGenerateResponse)
async def admin_lesson_generate(request: LessonGenerateRequest) -> LessonGenerateResponse:
    return await _lesson_generate(request)


@app.post("/admin/lesson/start", response_model=AdminResponse)
async def admin_lesson_start(request: LessonStartRequest) -> AdminResponse:
    return await _lesson_start(_main_tenant(), request)


@app.post("/admin/lesson/stop", response_model=AdminResponse)
async def admin_lesson_stop() -> AdminResponse:
    return await _lesson_stop(_main_tenant())


@app.get("/{tenant_id}/admin/phrases", response_model=PhrasesResponse)
async def tenant_admin_get_phrases(tenant_id: str) -> PhrasesResponse:
    return _phrases_get(_get_tenant_or_404(tenant_id))


@app.put("/{tenant_id}/admin/phrases", response_model=PhrasesResponse)
async def tenant_admin_save_phrases(tenant_id: str, request: PhrasesRequest) -> PhrasesResponse:
    return _phrases_save(_get_tenant_or_404(tenant_id), request)


@app.post("/{tenant_id}/admin/lesson/generate", response_model=LessonGenerateResponse)
async def tenant_admin_lesson_generate(tenant_id: str, request: LessonGenerateRequest) -> LessonGenerateResponse:
    _get_tenant_or_404(tenant_id)
    return await _lesson_generate(request)


@app.post("/{tenant_id}/admin/lesson/start", response_model=AdminResponse)
async def tenant_admin_lesson_start(tenant_id: str, request: LessonStartRequest) -> AdminResponse:
    return await _lesson_start(_get_tenant_or_404(tenant_id), request)


@app.post("/{tenant_id}/admin/lesson/stop", response_model=AdminResponse)
async def tenant_admin_lesson_stop(tenant_id: str) -> AdminResponse:
    return await _lesson_stop(_get_tenant_or_404(tenant_id))


# =============================================================================
# STATIC FILE ENDPOINTS
# =============================================================================

@app.get("/")
async def serve_index() -> FileResponse:
    """Serve the main application page."""
    return FileResponse("index.html")


@app.get("/admin")
async def serve_admin() -> FileResponse:
    """Serve the admin control panel."""
    return FileResponse("admin.html")


@app.get("/manifest.json")
async def serve_manifest() -> FileResponse:
    """Serve the PWA manifest file."""
    return FileResponse("manifest.json")


@app.get("/favicon.ico")
async def serve_favicon() -> FileResponse:
    """Serve the favicon using the logo image."""
    return FileResponse("static/sm.png", media_type="image/png")


# =============================================================================
# TENANT ADMIN PANEL (Dynamic - serves admin.html with tenant context)
# =============================================================================


@app.get("/{tenant_id}/admin")
async def serve_tenant_admin(tenant_id: str) -> HTMLResponse:
    """Serve the admin control panel for a specific tenant."""
    from fastapi import HTTPException
    
    # Validate tenant
    if not tenant_manager.is_valid_tenant(tenant_id):
        raise HTTPException(status_code=404, detail="Tenant not found")
    
    # Read the base admin.html and inject tenant context
    admin_path = Path("admin.html")
    if not admin_path.exists():
        raise HTTPException(status_code=404, detail="Admin template not found")
    
    html_content = admin_path.read_text(encoding="utf-8")
    
    # Inject tenant ID into the page via a script tag
    tenant_script = f'''<script>window.TENANT_ID = "{tenant_id}";</script>'''
    
    # Insert before the closing </head> tag
    html_content = html_content.replace("</head>", f"{tenant_script}\n</head>")
    
    # Update the title to include tenant name
    html_content = html_content.replace(
        "<title>Robot Admin</title>",
        f"<title>{tenant_id.upper()} Robot Admin</title>"
    )
    
    return HTMLResponse(content=html_content)


# =============================================================================
# DYNAMIC STATIC FILE SERVING (CATCH-ALL - MUST BE LAST)
# =============================================================================

@app.get("/{file_path:path}")
async def serve_dynamic_file(file_path: str):
    """
    Catch-all route to serve files securely from subdirectories in the project root.
    
    IMPORTANT: This route MUST be defined last to avoid intercepting specific routes.
    
    Security Rules:
    1. NO serving files from the project root (e.g. main.py, .env).
    2. NO serving files from hidden directories (e.g. .git/).
    3. NO serving files from system directories (e.g. venv/, __pycache__/).
    4. ONLY serve files if they are inside a valid subdirectory (e.g. ksu/index.html).
    """
    from fastapi import HTTPException
    
    # Resolve the path relative to the project root
    project_root = Path(".").resolve()
    requested_path = (project_root / file_path).resolve()
    
    # Security Check 1: Ensure path is within project root
    if not str(requested_path).startswith(str(project_root)):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Get the relative path parts
    try:
        relative_path = requested_path.relative_to(project_root)
    except ValueError:
        raise HTTPException(status_code=404, detail="File not found")
        
    parts = relative_path.parts
    
    # Security Check 2: Block root-level FILES
    # Allow root-level DIRECTORIES (to serve their index.html)
    # e.g. "ksu" -> parts=("ksu",) -> OK (if directory)
    # e.g. "main.py" -> parts=("main.py") -> BLOCKED (if file)
    if len(parts) < 2 and requested_path.is_file():
        raise HTTPException(status_code=404, detail="Not found")

    # Security Check 3: Block hidden and system directories
    top_level_dir = parts[0]
    if (
        top_level_dir.startswith(".") or 
        top_level_dir in ("venv", "__pycache__", "static", "__pycache__")
    ):
        raise HTTPException(status_code=404, detail="Not found")

    # If it's a directory, look for index.html and inject tenant context
    if requested_path.is_dir():
        index_path = requested_path / "index.html"
        if index_path.exists() and index_path.is_file():
            # Check if this is a tenant folder (first-level directory)
            if len(parts) == 1 and tenant_manager.is_valid_tenant(parts[0]):
                # Inject tenant ID into the HTML
                tenant_id = parts[0]
                html_content = index_path.read_text(encoding="utf-8")
                tenant_script = f'<script>window.TENANT_ID = "{tenant_id}";</script>'
                html_content = html_content.replace("</head>", f"{tenant_script}\n</head>")
                return HTMLResponse(content=html_content)
            return FileResponse(index_path)
        raise HTTPException(status_code=404, detail="File not found")

    # If it's a file, serve it
    if requested_path.exists() and requested_path.is_file():
        return FileResponse(requested_path)
    
    # Try adding .html extension (e.g., /ksu/admin -> /ksu/admin.html)
    html_path = requested_path.with_suffix(".html")
    if html_path.exists() and html_path.is_file():
        # Re-check security: ensure .html file is not at root level
        html_relative = html_path.relative_to(project_root)
        if len(html_relative.parts) >= 2:
            return FileResponse(html_path)
        
    raise HTTPException(status_code=404, detail="File not found")
