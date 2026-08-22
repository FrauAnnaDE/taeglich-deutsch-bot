import asyncio
import io
import json
import logging
import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import aiohttp
from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from openai import OpenAI


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("taeglich_deutsch")

DEBUG_TELEGRAM_ID = 328761045
PROJECT_ROOT = Path(__file__).resolve().parent
LESSONS_DIR = PROJECT_ROOT / "lessons"


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    yandex_api_key: str
    yandex_folder_id: str
    yandex_model: str = "gpt-oss-120b/latest"

    @classmethod
    def from_environment(cls) -> "Settings":
        required = {
            "TELEGRAM_BOT_TOKEN": os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
            "YANDEX_API_KEY": os.getenv("YANDEX_API_KEY", "").strip(),
            "YANDEX_FOLDER_ID": os.getenv("YANDEX_FOLDER_ID", "").strip(),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise RuntimeError(
                "Не заданы обязательные переменные окружения: " + ", ".join(missing)
            )
        return cls(
            telegram_bot_token=required["TELEGRAM_BOT_TOKEN"],
            yandex_api_key=required["YANDEX_API_KEY"],
            yandex_folder_id=required["YANDEX_FOLDER_ID"],
            yandex_model=os.getenv("YANDEX_MODEL", "gpt-oss-120b/latest").strip(),
        )


def load_lesson(day: int) -> dict[str, Any]:
    path = LESSONS_DIR / f"day_{day:02d}.json"
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


DAY_1 = load_lesson(1)


SYSTEM_INSTRUCTION = """Ты — доброжелательный преподаватель немецкого языка для русскоязычного ученика уровня A2.

Ты проверяешь расшифрованный голосовой ответ ученика на конкретное задание. Проверяй только критерии, перечисленные в запросе. Не исправляй пунктуацию, регистр букв и ошибки распознавания речи. Не придумывай ошибок и не требуй дословного повторения опоры: естественные правильные варианты допустимы.

Верни только корректный JSON без Markdown и без пояснений вокруг него:
{
  "accepted": true или false,
  "feedback": "короткая доброжелательная обратная связь на русском языке",
  "repeat_phrase": "одно короткое немецкое предложение для повторения"
}

Если в ответе есть все четыре требуемые мысли, seit употреблено с Präsens, а порядок слов понятный и естественный, поставь accepted=true. В feedback сначала кратко похвали ответ. В repeat_phrase выбери одну полезную фразу из естественного варианта ответа; предпочтительно предложение с seit. Сохрани названный учеником срок.

Если не хватает требуемой мысли или в целевой конструкции есть существенная ошибка, поставь accepted=false. Объясни только одну самую важную проблему, покажи правильную модель и попроси записать весь ответ ещё раз. repeat_phrase оставь пустой строкой.

SpeechKit часто не передаёт точки. Если четыре мысли выражены последовательно, не отклоняй ответ только из-за отсутствия пунктуации."""


class YandexServices:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.openai_client = OpenAI(
            api_key=settings.yandex_api_key,
            base_url="https://ai.api.cloud.yandex.net/v1",
            project=settings.yandex_folder_id,
            timeout=60.0,
            max_retries=2,
        )

    async def recognize_german(self, audio: bytes) -> str:
        url = "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize"
        headers = {"Authorization": f"Api-Key {self.settings.yandex_api_key}"}
        params = {
            "lang": "de-DE",
            "format": "oggopus",
            "topic": "general",
            "profanityFilter": "false",
        }
        timeout = aiohttp.ClientTimeout(total=45)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                url,
                headers=headers,
                params=params,
                data=audio,
            ) as response:
                payload = await response.json(content_type=None)
                if response.status != 200:
                    logger.error("SpeechKit error %s: %s", response.status, payload)
                    raise RuntimeError("SpeechKit не смог распознать запись")
        text = str(payload.get("result", "")).strip()
        if not text:
            raise RuntimeError("В записи не удалось распознать немецкую речь")
        return text

    async def create_feedback(self, student_answer: str) -> dict[str, Any]:
        prompt = DAY_1["feedback_prompt"].format(student_answer=student_answer)

        def request() -> str:
            response = self.openai_client.responses.create(
                model=(
                    f"gpt://{self.settings.yandex_folder_id}/"
                    f"{self.settings.yandex_model}"
                ),
                instructions=SYSTEM_INSTRUCTION,
                input=prompt,
                temperature=0.1,
                max_output_tokens=600,
            )
            return response.output_text.strip()

        raw_feedback = await asyncio.to_thread(request)
        if not raw_feedback:
            raise RuntimeError("Модель не вернула обратную связь")
        cleaned = raw_feedback.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I)
        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError as error:
            logger.error("Invalid feedback JSON: %s", raw_feedback)
            raise RuntimeError("Модель вернула ответ в неверном формате") from error
        if not isinstance(result.get("accepted"), bool):
            raise RuntimeError("В ответе модели нет статуса проверки")
        if not str(result.get("feedback", "")).strip():
            raise RuntimeError("В ответе модели нет обратной связи")
        if result["accepted"] and not str(result.get("repeat_phrase", "")).strip():
            result["repeat_phrase"] = DAY_1["default_repeat_phrase"]
        return result


@dataclass
class UserProgress:
    stage: str = "idle"
    repeat_phrase: str = ""


router = Router()
services: YandexServices | None = None
user_progress: dict[int, UserProgress] = {}


def get_services() -> YandexServices:
    if services is None:
        raise RuntimeError("Сервисы бота ещё не инициализированы")
    return services


def progress_for(user_id: int) -> UserProgress:
    return user_progress.setdefault(user_id, UserProgress())


def private_error_text(error: Exception) -> str:
    details = f"{type(error).__name__}: {error}"
    current_services = get_services()
    for secret in (
        current_services.settings.telegram_bot_token,
        current_services.settings.yandex_api_key,
    ):
        if secret:
            details = details.replace(secret, "***")
    return details[:1500]


def one_button(text: str, callback_data: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=text, callback_data=callback_data)]
        ]
    )


def main_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Начать День 1", callback_data="day1_start")],
            [InlineKeyboardButton(text="Как это работает", callback_data="how_it_works")],
        ]
    )


def quiz_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Unter Sayana", callback_data="day1_quiz_under")],
            [InlineKeyboardButton(text="Über Sayana", callback_data="day1_quiz_over")],
            [InlineKeyboardButton(text="Im Nachbarhaus", callback_data="day1_quiz_house")],
        ]
    )


WELCOME_TEXT = """Здравствуйте! Я — Täglich Deutsch с Frau Anna. 🇩🇪

Здесь немецкий занимает не больше 15 минут в день: короткая ситуация, полезные выражения и голосовой ответ с обратной связью от ИИ.

Сейчас доступен День 1 уровня A2."""


@router.message(CommandStart())
async def start(message: Message) -> None:
    if message.from_user:
        user_progress.pop(message.from_user.id, None)
    await message.answer(WELCOME_TEXT, reply_markup=main_keyboard())


@router.message(Command("help"))
async def help_command(message: Message) -> None:
    await message.answer(
        "Нажмите /start и выберите День 1. Проходите шаги по кнопкам, "
        "а в голосовом задании отвечайте по-немецки сообщением до 30 секунд."
    )


@router.message(Command("about"))
async def about_command(message: Message) -> None:
    await message.answer(
        "Täglich Deutsch — ежедневный A2-тренажёр Frau Anna. "
        "Речь распознаёт Yandex SpeechKit, а обратную связь формирует "
        "модель GPT OSS 120B в Yandex AI Studio."
    )


@router.callback_query(F.data == "day1_start")
async def day1_start(callback: CallbackQuery) -> None:
    await callback.answer()
    if not callback.message or not callback.from_user:
        return
    video_path = PROJECT_ROOT / DAY_1["media"]["video"]
    if not video_path.is_file():
        await callback.message.answer("Видео урока временно недоступно.")
        return
    progress_for(callback.from_user.id).stage = "video"
    await callback.message.answer(DAY_1["intro_text"])
    await callback.message.answer_video(
        video=FSInputFile(video_path),
        reply_markup=one_button("Wichtige Sätze", "day1_phrases"),
    )


@router.callback_query(F.data == "day1_phrases")
async def day1_phrases(callback: CallbackQuery) -> None:
    await callback.answer()
    if callback.message and callback.from_user:
        progress_for(callback.from_user.id).stage = "phrases"
        await callback.message.answer(
            DAY_1["phrases_text"],
            reply_markup=one_button("Weiter", "day1_grammar"),
        )


@router.callback_query(F.data == "day1_grammar")
async def day1_grammar(callback: CallbackQuery) -> None:
    await callback.answer()
    if callback.message and callback.from_user:
        progress_for(callback.from_user.id).stage = "grammar"
        await callback.message.answer(
            DAY_1["grammar_text"],
            reply_markup=one_button("Zur Frage", "day1_quiz"),
        )


@router.callback_query(F.data == "day1_quiz")
async def day1_quiz(callback: CallbackQuery) -> None:
    await callback.answer()
    if callback.message and callback.from_user:
        progress_for(callback.from_user.id).stage = "quiz"
        await callback.message.answer(
            f"<b>{DAY_1['quiz']['question']}</b>",
            reply_markup=quiz_keyboard(),
        )


@router.callback_query(F.data.in_({"day1_quiz_under", "day1_quiz_house"}))
async def day1_quiz_wrong(callback: CallbackQuery) -> None:
    await callback.answer()
    if callback.message:
        await callback.message.answer(DAY_1["quiz"]["wrong_text"])


async def send_voice_task(message: Message, user_id: int) -> None:
    audio_path = PROJECT_ROOT / DAY_1["media"]["voice_task"]
    if not audio_path.is_file():
        await message.answer("Аудиозадание временно недоступно.")
        return
    progress = progress_for(user_id)
    progress.stage = "answer"
    progress.repeat_phrase = ""
    await message.answer_audio(
        audio=FSInputFile(audio_path),
        caption=DAY_1["voice_task_text"],
        title="Tag 1 — голосовое задание",
    )


@router.callback_query(F.data == "day1_quiz_over")
async def day1_quiz_correct(callback: CallbackQuery) -> None:
    await callback.answer()
    if callback.message and callback.from_user:
        await callback.message.answer(DAY_1["quiz"]["correct_text"])
        await send_voice_task(callback.message, callback.from_user.id)


def normalize_german(text: str) -> str:
    number_words = {
        "0": "null",
        "1": "eins",
        "2": "zwei",
        "3": "drei",
        "4": "vier",
        "5": "fünf",
        "6": "sechs",
        "7": "sieben",
        "8": "acht",
        "9": "neun",
        "10": "zehn",
        "11": "elf",
        "12": "zwölf",
    }
    result = text.lower()
    for digit, word in number_words.items():
        result = re.sub(rf"\b{digit}\b", word, result)
    return " ".join(re.findall(r"[a-zäöüß]+", result))


def repeat_is_correct(recognized: str, expected: str) -> bool:
    recognized_normalized = normalize_german(recognized)
    expected_normalized = normalize_german(expected)
    if not recognized_normalized or not expected_normalized:
        return False
    ratio = SequenceMatcher(
        None, recognized_normalized, expected_normalized
    ).ratio()
    expected_words = expected_normalized.split()
    recognized_words = set(recognized_normalized.split())
    coverage = sum(word in recognized_words for word in expected_words) / len(
        expected_words
    )
    return ratio >= 0.74 or coverage >= 0.85


async def download_and_recognize(message: Message, bot: Bot) -> str:
    if message.voice is None:
        raise RuntimeError("Голосовое сообщение отсутствует")
    telegram_file = await bot.get_file(message.voice.file_id)
    if not telegram_file.file_path:
        raise RuntimeError("Telegram не вернул путь к голосовому файлу")
    buffer = io.BytesIO()
    await bot.download_file(telegram_file.file_path, destination=buffer)
    return await get_services().recognize_german(buffer.getvalue())


async def process_main_answer(
    message: Message, recognized_text: str, progress: UserProgress
) -> None:
    status = await message.answer("Проверяю ответ…")
    try:
        result = await get_services().create_feedback(recognized_text)
        if not result["accepted"]:
            await status.edit_text(str(result["feedback"]))
            progress.stage = "answer"
            return
        repeat_phrase = str(result["repeat_phrase"]).strip()
        progress.stage = "repeat"
        progress.repeat_phrase = repeat_phrase
        await status.edit_text(
            f"{result['feedback']}\n\n"
            f"<b>Wiederholen Sie bitte:</b>\n{repeat_phrase}"
        )
    except Exception as error:
        logger.exception("Feedback generation failed")
        await status.edit_text(
            "Не получилось сформировать обратную связь. Попробуйте ещё раз чуть позже."
        )
        if message.from_user and message.from_user.id == DEBUG_TELEGRAM_ID:
            await message.answer(
                "🔧 Диагностика (видна только владельцу):\n"
                f"{private_error_text(error)}"
            )


async def process_repeat(
    message: Message, recognized_text: str, progress: UserProgress
) -> None:
    if repeat_is_correct(recognized_text, progress.repeat_phrase):
        progress.stage = "done"
        await message.answer(
            "<b>Sehr gut! Tag 1 ist geschafft.</b>\n\nBis morgen!"
        )
        return
    await message.answer(
        "Попробуйте ещё раз. Повторите только эту фразу:\n\n"
        f"<b>{progress.repeat_phrase}</b>"
    )


@router.message(F.voice)
async def voice_answer(message: Message, bot: Bot) -> None:
    if message.voice is None or not message.from_user:
        return
    progress = progress_for(message.from_user.id)
    if progress.stage not in {"answer", "repeat"}:
        await message.answer(
            "Сначала откройте День 1 командой /start и дойдите до голосового задания."
        )
        return
    if message.voice.duration > 30:
        await message.answer(
            "Пока я принимаю голосовые сообщения длительностью до 30 секунд. "
            "Попробуйте ответить немного короче."
        )
        return
    status = await message.answer("Слушаю немецкий ответ…")
    try:
        recognized_text = await download_and_recognize(message, bot)
        await status.edit_text(f"Я услышала:\n{recognized_text}")
        if progress.stage == "answer":
            await process_main_answer(message, recognized_text, progress)
        else:
            await process_repeat(message, recognized_text, progress)
    except Exception as error:
        logger.exception("Voice processing failed")
        await status.edit_text(
            "Не удалось разобрать голосовое сообщение. Запишите ответ ещё раз "
            "в тихом месте и говорите чуть медленнее."
        )
        if message.from_user.id == DEBUG_TELEGRAM_ID:
            await message.answer(
                "🔧 Диагностика (видна только владельцу):\n"
                f"{private_error_text(error)}"
            )


@router.message(F.text)
async def text_answer(message: Message) -> None:
    if not message.text or message.text.startswith("/"):
        return
    if message.from_user and progress_for(message.from_user.id).stage in {
        "answer",
        "repeat",
    }:
        await message.answer("Пожалуйста, отправьте ответ голосовым сообщением.")
    else:
        await message.answer("Чтобы начать урок, нажмите /start.")


@router.callback_query(F.data == "how_it_works")
async def how_it_works(callback: CallbackQuery) -> None:
    await callback.answer()
    if callback.message:
        await callback.message.answer(
            "1. Вы смотрите короткую ситуацию.\n"
            "2. Разбираете выражения и одну конструкцию.\n"
            "3. Отвечаете на вопрос по видео.\n"
            "4. Записываете ответ по-немецки.\n"
            "5. Получаете обратную связь и повторяете одну фразу."
        )


async def run_bot() -> None:
    global services
    settings = Settings.from_environment()
    services = YandexServices(settings)
    bot = Bot(
        token=settings.telegram_bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dispatcher = Dispatcher()
    dispatcher.include_router(router)
    logger.info("Starting Täglich Deutsch bot")
    try:
        await dispatcher.start_polling(
            bot, allowed_updates=dispatcher.resolve_used_update_types()
        )
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(run_bot())
