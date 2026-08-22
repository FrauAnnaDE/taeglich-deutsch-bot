import asyncio
import io
import logging
import os
from dataclasses import dataclass

import aiohttp
from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from openai import OpenAI


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("taeglich_deutsch")


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


SYSTEM_INSTRUCTION = """Ты — преподаватель немецкого языка для русскоязычного ученика уровня A2.

Ты проверяешь расшифрованный голосовой ответ ученика. Вместе с ответом ты получаешь вопрос, правило текущего урока и допустимые варианты.

Твоя задача — дать короткую, доброжелательную и понятную обратную связь на русском языке.

Главное правило проверки:

Проверяй только то правило, которое указано в запросе в разделе «Правило проверки».

Не ищи и не исправляй никакие другие грамматические, лексические, стилистические или орфографические ошибки. Не исправляй пунктуацию и регистр букв. Не предлагай стилистический вариант как грамматическое исправление.

Если фраза соответствует указанному правилу и входит в допустимые варианты, считай её правильной. Не придумывай ошибку, если её нет.

Не меняй смысл ответа ученика, имена, числа, факты и период времени. Не добавляй информацию, которой не было в ответе.

Если в проверяемой конструкции несколько ошибок, выбери только одну самую важную.

Ответ всегда должен состоять ровно из пяти строк и идти строго в следующем порядке:

Я поняла: ...
Хорошо: ...
Исправим: ...
Естественный вариант: ...
А теперь повторите: ...

Правила для каждой строки:

1. «Я поняла»

Кратко перескажи на русском языке, о чём сообщил ученик. Не объясняй здесь грамматику и не называй правило урока.

2. «Хорошо»

Приведи только одну удачную немецкую фразу или часть фразы из исходного ответа ученика. Цитируй её дословно. Эта фраза должна уже быть правильной в исходном ответе.

Не выдавай исправленную тобой фразу за исходную.

Если в ответе нет ни одной подходящей правильной немецкой фразы, напиши: «Вы ответили на вопрос по теме».

3. «Исправим»

Если в проверяемой конструкции есть ошибка, покажи только её в формате:

Исправим: „ошибочный вариант“ → „правильный вариант“.

Не упоминай и не исправляй другие ошибки.

Если в проверяемой конструкции ошибки нет, напиши:

Исправим: В проверяемой конструкции всё верно.

4. «Естественный вариант»

Дай одно полное немецкое предложение с правильной конструкцией. Исправь в нём только правило текущего урока. Сохрани исходный смысл, факты, числа и период времени.

5. «А теперь повторите»

Дословно повтори немецкое предложение из строки «Естественный вариант». Не изменяй в нём ни одного слова. Никогда не проси повторять русское объяснение.

Не добавляй другие строки, заголовки, пояснения, вступление или заключение. Не пропускай ни одну из пяти строк."""


LESSON_PROMPT_TEMPLATE = """Уровень: A2

Вопрос:
Was ist gestern mit dem Paket passiert?

Правило проверки:
Проверяем только порядок слов в повествовательном главном предложении.

Спрягаемый глагол должен стоять на втором месте.
Если предложение начинается со слова „Gestern“, после него должен стоять глагол:

Gestern habe ich ein Paket angenommen.

Допустим также вариант:

Ich habe gestern ein Paket angenommen.

Не проверяй и не исправляй никакие другие ошибки.

Ответ ученика:
{student_answer}"""


WELCOME_TEXT = """Здравствуйте! Я — Täglich Deutsch с Frau Anna. 🇩🇪

Здесь немецкий занимает не больше 15 минут в день: короткая ситуация, полезная фраза и голосовой ответ с обратной связью от ИИ.

Сейчас доступен тестовый мини-урок уровня A2."""


LESSON_TEXT = """📦 Мини-урок: Ein Paket für die Nachbarin

Ситуация: вчера вы приняли посылку для соседки.

Полезные слова:
• das Paket — посылка
• annehmen — принимать
• die Nachbarin — соседка
• abholen — забирать

Сегодня тренируем порядок слов:
✅ Gestern habe ich ein Paket angenommen.

Ответьте голосовым сообщением на вопрос:
Was ist gestern mit dem Paket passiert?

Голосовое сообщение должно быть не длиннее 30 секунд."""


def main_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Начать мини-урок", callback_data="lesson_1")],
            [InlineKeyboardButton(text="Как это работает", callback_data="how_it_works")],
        ]
    )


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

    async def create_feedback(self, student_answer: str) -> str:
        prompt = LESSON_PROMPT_TEMPLATE.format(student_answer=student_answer)

        def request() -> str:
            response = self.openai_client.responses.create(
                model=(
                    f"gpt://{self.settings.yandex_folder_id}/"
                    f"{self.settings.yandex_model}"
                ),
                instructions=SYSTEM_INSTRUCTION,
                input=prompt,
                temperature=0.1,
                max_output_tokens=850,
            )
            return response.output_text.strip()

        feedback = await asyncio.to_thread(request)
        if not feedback:
            raise RuntimeError("Модель не вернула обратную связь")
        return feedback


router = Router()
services: YandexServices | None = None


def get_services() -> YandexServices:
    if services is None:
        raise RuntimeError("Сервисы бота ещё не инициализированы")
    return services


@router.message(CommandStart())
async def start(message: Message) -> None:
    await message.answer(WELCOME_TEXT, reply_markup=main_keyboard())


@router.message(Command("help"))
async def help_command(message: Message) -> None:
    await message.answer(
        "Нажмите /start, выберите мини-урок и отправьте короткий ответ "
        "голосовым сообщением. Для проверки можно также прислать ответ текстом."
    )


@router.message(Command("about"))
async def about_command(message: Message) -> None:
    await message.answer(
        "Täglich Deutsch — тестовый A2-тренажёр Frau Anna. "
        "Речь распознаёт Yandex SpeechKit, а обратную связь формирует "
        "модель GPT OSS 120B в Yandex AI Studio."
    )


@router.callback_query(F.data == "lesson_1")
async def lesson_1(callback: CallbackQuery) -> None:
    await callback.answer()
    if callback.message:
        await callback.message.answer(LESSON_TEXT)


@router.callback_query(F.data == "how_it_works")
async def how_it_works(callback: CallbackQuery) -> None:
    await callback.answer()
    if callback.message:
        await callback.message.answer(
            "1. Вы смотрите мини-урок.\n"
            "2. Получаете короткое задание.\n"
            "3. Отвечаете по-немецки голосом.\n"
            "4. ИИ проверяет только правило текущего урока и даёт одну фразу для повторения."
        )


async def send_feedback(message: Message, recognized_text: str) -> None:
    status = await message.answer("Проверяю ответ…")
    try:
        feedback = await get_services().create_feedback(recognized_text)
        await status.edit_text(feedback)
    except Exception:
        logger.exception("Feedback generation failed")
        await status.edit_text(
            "Не получилось сформировать обратную связь. Попробуйте ещё раз чуть позже."
        )


@router.message(F.voice)
async def voice_answer(message: Message, bot: Bot) -> None:
    if message.voice is None:
        return
    if message.voice.duration > 30:
        await message.answer(
            "Пока я принимаю голосовые сообщения длительностью до 30 секунд. "
            "Попробуйте ответить немного короче."
        )
        return

    status = await message.answer("Слушаю немецкий ответ…")
    try:
        telegram_file = await bot.get_file(message.voice.file_id)
        if not telegram_file.file_path:
            raise RuntimeError("Telegram не вернул путь к голосовому файлу")

        buffer = io.BytesIO()
        await bot.download_file(telegram_file.file_path, destination=buffer)
        audio = buffer.getvalue()
        recognized_text = await get_services().recognize_german(audio)

        await status.edit_text(f"Я услышала:\n{recognized_text}")
        await send_feedback(message, recognized_text)
    except Exception:
        logger.exception("Voice processing failed")
        await status.edit_text(
            "Не удалось разобрать голосовое сообщение. Запишите ответ ещё раз "
            "в тихом месте и говорите чуть медленнее."
        )


@router.message(F.text)
async def text_answer(message: Message) -> None:
    if not message.text or message.text.startswith("/"):
        return
    await send_feedback(message, message.text.strip())


async def run_bot() -> None:
    global services
    settings = Settings.from_environment()
    services = YandexServices(settings)

    bot = Bot(token=settings.telegram_bot_token)
    dispatcher = Dispatcher()
    dispatcher.include_router(router)

    logger.info("Starting Täglich Deutsch bot")
    try:
        await dispatcher.start_polling(bot, allowed_updates=dispatcher.resolve_used_update_types())
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(run_bot())
