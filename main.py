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

# Технические подробности ошибок видит только владелец бота.
DEBUG_TELEGRAM_ID = 328761045


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


SYSTEM_INSTRUCTION = """Ты — доброжелательный преподаватель немецкого языка для русскоязычного ученика уровня A2.

Ты проверяешь расшифрованный голосовой ответ ученика. Вместе с ответом ты получаешь вопрос, правило текущего урока и допустимые варианты.

Твоя задача — дать короткую, доброжелательную и понятную обратную связь на русском языке.

Главное правило проверки:

Проверяй только то правило, которое указано в запросе в разделе «Правило проверки».

Не ищи и не исправляй никакие другие грамматические, лексические, стилистические или орфографические ошибки. Не исправляй пунктуацию и регистр букв. Не предлагай стилистический вариант как грамматическое исправление.

Если фраза соответствует указанному правилу и входит в допустимые варианты, считай её правильной. Не придумывай ошибку, если её нет.

Не меняй смысл ответа ученика, имена, числа, факты и период времени. Не добавляй информацию, которой не было в ответе.

Сохраняй лицо и точку зрения говорящего при пересказе. Если ученик говорит „bei mir“, „ich“ или „mein“, пересказывай это как «у вас», «вы» или «ваш», а не как «у него», «он» или «его». Учитывай контекст разговора: если ученик обращается к соседке и говорит „Ihr Paket“, это её посылка, которую ученик принял для неё.

Если в проверяемой конструкции несколько ошибок, выбери только одну самую важную.

Распознанная речь может быть записана без заглавных букв и знаков препинания. Не считай это ошибкой. В примерах для ученика восстанавливай обычные заглавные буквы и пунктуацию.

Ответ всегда должен состоять ровно из пяти абзацев и идти строго в следующем порядке:

Я поняла: ...
Хорошо: ...
Попробуем цель урока: ...
Естественный вариант: ...
А теперь повторите: ...

Правила для каждой строки:

1. «Я поняла»

Кратко перескажи на русском языке, о чём сообщил ученик. Не объясняй здесь грамматику и не называй правило урока.

2. «Хорошо»

Приведи одну удачную немецкую фразу или часть фразы из ответа ученика и кратко объясни по-русски, что в ней получилось. Можно восстановить только заглавные буквы и пунктуацию, но нельзя незаметно менять слова или порядок слов.

Не выдавай исправленную тобой фразу за исходную.

Если в ответе нет ни одной подходящей правильной немецкой фразы, напиши: «Вы ответили на вопрос по теме».

3. «Попробуем цель урока»

Если ученик использовал грамматически правильный вариант, но не целевую модель урока, не называй это ошибкой. Мягко предложи перестроить фразу по модели урока и кратко напомни правило.

Если в целевой конструкции есть ошибка, объясни только одну самую важную ошибку и покажи правильную модель.

Если ученик уже использовал целевую модель правильно, коротко подтверди это и напомни правило.

4. «Естественный вариант»

Дай естественный и связный немецкий вариант всего ответа ученика, а не только тренировочное предложение. Сохрани каждую переданную им мысль: сообщения, пояснения и вопросы, а также все факты, числа и период времени. Если ученик выразил несколько мыслей, естественный вариант тоже должен содержать несколько соответствующих предложений. Используй целевую конструкцию урока там, где она подходит. Можно восстановить заглавные буквы и пунктуацию. Не добавляй новой информации.

5. «А теперь повторите»

Дай только одно короткое немецкое предложение с целевой конструкцией урока, которое ученик должен повторить. Это микротренировка правила, а не повтор всего естественного варианта. Если в естественном варианте несколько предложений, выбери из него только предложение с целевой конструкцией. Поэтому строки «Естественный вариант» и «А теперь повторите» не должны полностью совпадать, когда ученик выразил две или больше мыслей. Они могут совпасть только тогда, когда весь содержательный ответ ученика действительно состоит из одного предложения. Никогда не проси повторять русское объяснение.

Не используй слова «ученик сообщил». Обращайся к человеку на «вы». Пиши тепло, просто и конкретно.

Не добавляй другие строки, заголовки, пояснения, вступление или заключение. Не пропускай ни один из пяти абзацев."""


LESSON_PROMPT_TEMPLATE = """Уровень: A2

Вопрос:
Was ist gestern mit dem Paket passiert?

Правило проверки:
Проверяем и тренируем целевую модель: повествовательное предложение должно начинаться со слова „Gestern“.

После „Gestern“ спрягаемый глагол должен стоять на втором месте, а подлежащее — после глагола:

Gestern habe ich ein Paket angenommen.

Вариант „Ich habe gestern ein Paket angenommen“ грамматически правильный, но не является целевой моделью этого мини-урока. Не называй его ошибкой: похвали правильный порядок слов и предложи перестроить фразу, начав с „Gestern“.

Контекст: ученик вежливо обращается к соседке. Поэтому „Ihr Paket“ означает «вашу посылку», а „Sie“ — вежливое «вы». Не переводи „Ihr Paket“ как «их пакет».

Сохраняй точку зрения говорящего. „Das Paket ist bei mir“ означает «посылка у вас», потому что ответ даёт сам ученик. Не пересказывай эту фразу как «пакет у него».

Распознавание речи обычно не передаёт заглавные буквы и пунктуацию. Не исправляй и не комментируй их как ошибки, но восстанови их в немецких примерах.

Не проверяй и не исправляй никакие другие ошибки.

Пример желаемой обратной связи для ответа
„hallo ich habe gestern ihr paket angenommen wann können sie es abholen“:

Я поняла: вы сообщили соседке, что вчера приняли её посылку, и спросили, когда она сможет её забрать.
Хорошо: „Ich habe gestern Ihr Paket angenommen.“ — порядок слов правильный.
Попробуем цель урока: начните предложение со слова „Gestern“. После него сразу ставим глагол „habe“.
Естественный вариант: Gestern habe ich Ihr Paket angenommen. Wann können Sie es abholen?
А теперь повторите: Gestern habe ich Ihr Paket angenommen.

Используй этот пример только как образец структуры и тона. Для другого ответа ученика учитывай его собственный смысл и формулировки.

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

Сегодня тренируем порядок слов. Начните ответ со слова „Gestern“:
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


def private_error_text(error: Exception) -> str:
    """Return a short diagnostic message without bot/API secrets."""
    details = f"{type(error).__name__}: {error}"
    current_services = get_services()
    for secret in (
        current_services.settings.telegram_bot_token,
        current_services.settings.yandex_api_key,
    ):
        if secret:
            details = details.replace(secret, "***")
    return details[:1500]


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
            "1. Вы получаете короткое задание.\n"
            "2. Отвечаете по-немецки голосом.\n"
            "3. SpeechKit переводит речь в текст.\n"
            "4. ИИ проверяет только правило текущего урока и даёт одну фразу для повторения."
        )


async def send_feedback(message: Message, recognized_text: str) -> None:
    status = await message.answer("Проверяю ответ…")
    try:
        feedback = await get_services().create_feedback(recognized_text)
        await status.edit_text(feedback)
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
