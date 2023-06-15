# https://docs.aiogram.dev/en/latest/
from aiogram import Bot, Dispatcher, types
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.utils import executor
from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    CallbackQuery,
    InputMediaDocument,
    InputFile,
)
import webuiapi, io
import subprocess
import time
import json
import requests
import asyncio
import os
from datetime import datetime
import aiohttp
from typing import Union
from PIL import Image
from transformers import GPT2Tokenizer, GPT2LMHeadModel

API_TOKEN = "TOKEN_HERE"

bot = Bot(token=API_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)

# -------- GLOBAL ----------
# TODO брать из outdir_txt2img_samples
img_dir = "C:/html/stable-diffusion-webui/outputs/txt2img-images/"
formatted_date = datetime.today().strftime("%Y-%m-%d")
host = "127.0.0.1"
port = "7861"
# create API client with custom host, port
api = webuiapi.WebUIApi(host=host, port=port)
local = "http://" + host + ":" + port
process = None
sd = "❌"
# TODO брать динамически из http://127.0.0.1:7861/openapi.json #/components/schemas/StableDiffusionProcessingTxt2Img
data = {
    "prompt": "cute dog",
    "negative_prompt": "ugly, out of frame",
    # "seed":datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3],
    "styles": ["anime"],
    "cfg_scale": 8,
    "steps": 15,
    "seed": "-1",
    "width": 512,
    "height": 512,
    "batch_size": 1,
    "hr_upscaler": webuiapi.HiResUpscaler.ESRGAN_4x,  # https://github.com/mix1009/sdwebuiapi/blob/main/webuiapi/webuiapi.py#L24
    "hr_scale": 2,
    "hr_second_pass_steps": 15,
    "denoising_strength": "0.2",  # TODO сделать обработку запятых
    "enable_hr": "false",
    "firstphase_width": 0,
    "firstphase_height": 0,
    "save_images": "true",
    "use_async": "True",
}

dataParams = {"img_thumb": "true", "img_tg": "true", "img_real": "true"}

dataOld = data.copy()
dataOldParams = dataParams.copy()
dataOrig = data.copy()

# -------- CLASSES ----------

# https://aiogram-birdi7.readthedocs.io/en/latest/examples/finite_state_machine_example.html
# Dynamically create a new class with the desired attributes
state_classes = {}
for key in data:
    state_classes[key] = State()
for key in dataParams:
    state_classes[key] = State()

# Inherit from the dynamically created class
Form = type("Form", (StatesGroup,), state_classes)


# -------- FUNCTIONS ----------
def pilToImages(res, typeImages="tg"):
    media_group = []
    imagesAll = res.images
    if len(res.images) == 1:
        i = 0
    if len(res.images) > 1:
        i = -1
    for image in imagesAll:
        # костыль для отсечения первой картинки с гридами
        if i == -1:
            i = i + 1
            continue
        seed = str(res.info["all_seeds"][i])
        image_buffer = io.BytesIO()
        image.save(image_buffer, format="PNG")
        image_buffer.seek(0)
        # картинка в телеге
        if typeImages == "tg":
            media_group.append(types.InputMediaPhoto(media=image_buffer, caption=seed))
        # оригинал
        if typeImages == "real":
            media_group.append(
                types.InputMediaDocument(
                    media=InputFile(image_buffer, filename=seed + ".png"), caption=seed
                )
            )
        # превью
        if typeImages == "thumbs":
            img = Image.open(image_buffer)
            width, height = img.size
            # пропорции
            ratio = min(256 / width, 256 / height)
            new_size = (round(width * ratio), round(height * ratio))
            img = img.resize(new_size)
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)
            # media_group.attach_photo(img_byte_arr, '111')
            media_group.append(types.InputMediaPhoto(media=img_byte_arr, caption=seed))
        i = i + 1
    return media_group


def getJson(params=0):
    if params == 0:
        d = data
    else:
        d = dataParams
    json_list = [f"/{key} = {value}" for key, value in d.items()]
    json_str = "\n".join(json_list)
    return json_str


# генератор промптов https://huggingface.co/FredZhang7/distilgpt2-stable-diffusion-v2
def get_random_prompt():
    text = data["prompt"]  # from JSON
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model = GPT2LMHeadModel.from_pretrained("FredZhang7/distilgpt2-stable-diffusion-v2")
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    txt = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.8,
        top_k=8,
        max_length=120,
        num_return_sequences=1,
        repetition_penalty=1.2,
        penalty_alpha=0.6,
        no_repeat_ngram_size=0,
        early_stopping=True,
    )
    prompt = tokenizer.decode(txt[0], skip_special_tokens=True)
    return prompt


# Поиск картинки в папках и возврат полного пути если нашёл
# TODO брать из get_next_sequence_number
# TODO возвращать список seed`ов если одинаковый постфикс
async def sendImagesFromSeed(seed, message, sendMedia=1):
    print("sendImagesFromSeed")
    seed = str(seed)
    keyboard = InlineKeyboardMarkup(inline_keyboard=[getSet(0), getStart(0)])
    args = message.get_args()
    img_way = img_dir + formatted_date
    files = [f for f in os.listdir(img_way) if seed in f]
    way = ""
    if len(files) > 0:
        last_file = sorted(files)[-1]
        last_file_name = os.path.splitext(last_file)[0]
        last_file_num = int(last_file_name.split("-")[0])
        last_file_num_next = f"{last_file_num:05d}"
        way = f"{last_file_num_next}-{seed}.png"
    else:
        print("Папка пуста")
    if len(os.listdir(img_way)) == 0:
        way = "00000-" + seed + ".png"
    longWay = img_dir + formatted_date + "/" + way
    if sendMedia == 1:
        media = types.MediaGroup()
        media.attach_photo(types.InputFile(longWay), args)
        await bot.send_media_group(chat_id=message.chat.id, media=media)
        await bot.send_document(message.from_user.id, media=media)
        await bot.send_message(
            chat_id=message.from_user.id,
            text=args,
            reply_markup=keyboard,
            parse_mode="Markdown",
        )
    else:
        return longWay


# Запуск SD через subprocess и запись в глобальную переменную process
def start_sd():
    global process
    if not process:
        print("start_process sd")
        try:
            # ../../ launch.py = смотрим из stable-diffusion-webui/extensions/sd_telegram в корень папки stable-diffusion-webui
            process = subprocess.Popen(
                ["python", "../../launch.py", "--nowebui", "--xformers"]
            )
            # TODO stderr, stdout выводить в сообщение телеграм
        except subprocess.CalledProcessError as e:
            print("e:", e)


# Остановка SD
def stop_sd():
    global process, sd
    if process:
        print("stop_process sd")
        process.terminate()
        process = None
        sd = "❌"


# Проверка связи до запущенной локальной SD с nowebui
def ping(status: str):
    n = 0
    url = local + "/docs"
    if status == "stop":
        while n == 200:
            time.sleep(3)
            try:
                r = requests.get(url, timeout=3)
                r.raise_for_status()
                n = r.status_code
            except requests.exceptions.HTTPError as errh:
                print("Http Error:", errh)
            except requests.exceptions.ConnectionError as errc:
                print("Error Connecting:", errc)
            except requests.exceptions.Timeout as errt:
                print("Timeout Error:", errt)
            except requests.exceptions.RequestException as err:
                print("OOps: Something Else", err)
        return True
    else:
        while n != 200:
            time.sleep(3)
            try:
                r = requests.get(url, timeout=3)
                r.raise_for_status()
                n = r.status_code
                print(r.status_code)
            except requests.exceptions.HTTPError as errh:
                print("Http Error:", errh)
            except requests.exceptions.ConnectionError as errc:
                print("Error Connecting:", errc)
            except requests.exceptions.Timeout as errt:
                print("Timeout Error:", errt)
            except requests.exceptions.RequestException as err:
                print("OOps: Something Else", err)
        return True


# -------- MENU ----------
# Стартовое меню
def getStart(returnAll=1) -> InlineKeyboardMarkup:
    keys = [
        InlineKeyboardButton("sd" + sd, callback_data="sd"),
        InlineKeyboardButton("opt", callback_data="opt"),
        InlineKeyboardButton("gen", callback_data="gen"),
        InlineKeyboardButton("skip", callback_data="skip"),
        InlineKeyboardButton("status", callback_data="status"),
        InlineKeyboardButton("help", callback_data="help"),
    ]
    keyAll = InlineKeyboardMarkup(inline_keyboard=[keys])
    if returnAll == 1:
        return keyAll
    else:
        return keys


# Меню опций
def getOpt(returnAll=1) -> InlineKeyboardMarkup:
    keys = [
        InlineKeyboardButton("settings", callback_data="settings"),
        InlineKeyboardButton("scripts", callback_data="scripts"),
        InlineKeyboardButton("prompt", callback_data="prompt"),
    ]
    keyAll = InlineKeyboardMarkup(inline_keyboard=[keys])
    if returnAll == 1:
        return keyAll
    else:
        return keys


# Меню скриптов
def getScripts(returnAll=1) -> InlineKeyboardMarkup:
    keys = [
        InlineKeyboardButton("get_lora", callback_data="get_lora"),
        InlineKeyboardButton("seed2img", callback_data="seed2img"),
    ]
    keyAll = InlineKeyboardMarkup(inline_keyboard=[keys])
    if returnAll == 1:
        return keyAll
    else:
        return keys


# Меню настроек
def getSet(returnAll=1) -> InlineKeyboardMarkup:
    keys = [
        InlineKeyboardButton("change_param", callback_data="change_param"),
        InlineKeyboardButton("reset_param", callback_data="reset_param"),
    ]
    keyAll = InlineKeyboardMarkup(inline_keyboard=[keys])
    if returnAll == 1:
        return keyAll
    else:
        return keys


# Меню настроек
def getPrompt(returnAll=1) -> InlineKeyboardMarkup:
    keys = [InlineKeyboardButton("random_prompt", callback_data="random_prompt")]
    keyAll = InlineKeyboardMarkup(inline_keyboard=[keys])
    if returnAll == 1:
        return keyAll
    else:
        return keys


# Меню генераций
def getGen(returnAll=1) -> InlineKeyboardMarkup:
    keys = [
        InlineKeyboardButton("gen1", callback_data="gen1"),
        InlineKeyboardButton("gen4", callback_data="gen4"),
        InlineKeyboardButton("gen10", callback_data="gen10"),
        InlineKeyboardButton("gen_hr", callback_data="gen_hr"),
        InlineKeyboardButton("gen_hr4", callback_data="gen_hr4"),
    ]
    keyAll = InlineKeyboardMarkup(inline_keyboard=[keys])
    if returnAll == 1:
        return keyAll
    else:
        return keys


# Меню текста
def getTxt():
    return "/start /opt /gen /skip /status /seed2img /help"


# -------- COMMANDS ----------
# start или help
@dp.callback_query_handler(text="help")
@dp.message_handler(commands=["help"])
@dp.message_handler(commands=["start"])
async def cmd_start(message: Union[types.Message, types.CallbackQuery]) -> None:
    print("cmd_start")
    txt = "Это бот для локального запуска SD\n" + getTxt()
    if hasattr(message, "content_type"):
        await message.reply(txt, reply_markup=getStart())
    else:
        await message.message.edit_text(txt, reply_markup=getStart())


# webuiapi.HiResUpscaler
@dp.message_handler(commands=["test"])
async def cmd_test(message: Union[types.Message, types.CallbackQuery]) -> None:
    print("cmd_test")
    print(webuiapi)


# Получить опции
@dp.message_handler(commands=["opt"])
@dp.callback_query_handler(text="opt")
async def cmd_opt(message: Union[types.Message, types.CallbackQuery]) -> None:
    print("cmd_opt")
    keyboard = InlineKeyboardMarkup(inline_keyboard=[getOpt(0), getStart(0)])
    if hasattr(message, "content_type"):
        await message.reply("Опции", reply_markup=keyboard)
    else:
        print("inl_opt")
        await message.message.edit_text("Опции", reply_markup=keyboard)


# Запуск/Остановка SD. Завязываемся на глобальную иконку sd
@dp.callback_query_handler(text="sd")
async def inl_sd(callback: types.CallbackQuery) -> None:
    print("inl_sd")
    global sd
    if sd == "✅":
        stop_sd()
        sd = "⌛"
        await callback.message.edit_text(
            "Останавливаем SD\n" + getTxt(), reply_markup=getStart()
        )
        ping("stop")
        sd = "❌"
        await callback.message.edit_text(
            "SD остановлена\n" + getTxt(), reply_markup=getStart()
        )
    else:
        start_sd()
        sd = "⌛"
        await callback.message.edit_text(
            "Запускаем SD\n" + getTxt(), reply_markup=getStart()
        )
        ping("start")
        sd = "✅"
        await callback.message.edit_text(
            "SD запущена\n" + getTxt(), reply_markup=getStart()
        )


# Остановка SD по /stop
@dp.message_handler(commands=["stop"])
async def cmd_stop(message: types.Message) -> None:
    print("cmd_stop")
    global sd
    stop_sd()
    sd = "⌛"  # ?
    ping("stop")
    sd = "❌"
    await bot.send_message(
        chat_id=message.from_user.id, text="SD остановлена", reply_markup=getOpt()
    )


@dp.callback_query_handler(text="random_prompt")
async def random_prompt(callback: types.CallbackQuery) -> None:
    await bot.send_message(chat_id=callback.from_user.id, text=get_random_prompt())


# Вызов settings
@dp.message_handler(commands=["prompt"])
@dp.callback_query_handler(text="prompt")
async def inl_prompt(message: Union[types.Message, types.CallbackQuery]) -> None:
    print("inl_prompt")
    keyboard = InlineKeyboardMarkup(inline_keyboard=[getPrompt(0), getStart(0)])
    txt = "Промпты"
    if hasattr(message, "content_type"):
        await bot.send_message(
            chat_id=message.from_user.id, text=txt, reply_markup=keyboard
        )
    else:
        await bot.edit_message_text(
            chat_id=message.message.chat.id,
            message_id=message.message.message_id,
            text=txt,
            reply_markup=keyboard,
        )


# TODO delete
async def send_time(background_task: asyncio.Task):
    while True:
        current_time = datetime.now().strftime("%H:%M:%S")
        print(current_time)
        await asyncio.sleep(1)


# Проверка прогресса
@dp.message_handler(commands=["stat"])
async def cmd_stat(message: types.Message) -> None:
    keyboard = InlineKeyboardMarkup(inline_keyboard=[getGen(0), getStart(0)])
    asyncio.create_task(
        send_time(
            asyncio.create_task(bot.send_message(message.chat.id, "Текущее время:"))
        )
    )
    print("cmd_stat")
    send_message = await bot.send_message(message.chat.id, "Картинка генерируется")
    response = requests.get(
        local + "/sdapi/v1/progress?skip_current_image=false", data=json.dumps("")
    )
    e = response.json()["eta_relative"]
    while e > 0:
        response = requests.get(
            local + "/sdapi/v1/progress?skip_current_image=false", data=json.dumps("")
        )
        e = round(response.json()["eta_relative"], 1)
        time.sleep(2)
        await send_message.edit_text(e, reply_markup=keyboard)
    await send_message.edit_text("Готово", reply_markup=keyboard)


# Вызов settings
@dp.message_handler(commands=["settings"])
@dp.callback_query_handler(text="settings")
async def inl_settings(message: Union[types.Message, types.CallbackQuery]) -> None:
    print("inl_settings")
    keyboard = InlineKeyboardMarkup(inline_keyboard=[getSet(0), getStart(0)])
    txt = "Настройки"
    # Если команда /settings
    if hasattr(message, "content_type"):
        await bot.send_message(
            chat_id=message.from_user.id, text=txt, reply_markup=keyboard
        )
    else:
        await bot.edit_message_text(
            chat_id=message.message.chat.id,
            message_id=message.message.message_id,
            text=txt,
            reply_markup=keyboard,
        )


# Вызов script
@dp.message_handler(commands=["scripts"])
@dp.callback_query_handler(text="scripts")
async def inl_scripts(message: Union[types.Message, types.CallbackQuery]) -> None:
    print("inl_scripts")
    keyboard = InlineKeyboardMarkup(inline_keyboard=[getScripts(0), getStart(0)])
    txt = "Скрипты"
    # Если команда /settings
    if hasattr(message, "content_type"):
        await bot.send_message(
            chat_id=message.from_user.id, text=txt, reply_markup=keyboard
        )
    else:
        await bot.edit_message_text(
            chat_id=message.message.chat.id,
            message_id=message.message.message_id,
            text=txt,
            reply_markup=keyboard,
        )


# Вызов change_param
@dp.callback_query_handler(text="change_param")
async def inl_change_param(callback: types.CallbackQuery) -> None:
    print("inl_change_param")
    keyboard = InlineKeyboardMarkup(inline_keyboard=[getSet(0), getStart(0)])
    json_list = [f"/{key} = {value}" for key, value in data.items()]
    json_list_params = [f"/{key} = {value}" for key, value in dataParams.items()]
    json_str = "\n".join(json_list)
    json_str_params = "\n".join(json_list_params)
    await callback.message.edit_text(
        f"JSON параметры:\n{json_str}\n{json_str_params}", reply_markup=keyboard
    )


# Вызов reset_param, сброс JSON
@dp.message_handler(commands=["reset_param"])
@dp.callback_query_handler(text="reset_param")
async def inl_reset_param(message: Union[types.Message, types.CallbackQuery]) -> None:
    print("inl_reset_param")
    global data
    global dataParams
    data = dataOld
    dataParams = dataOldParams
    keyboard = InlineKeyboardMarkup(inline_keyboard=[getSet(0), getStart(0)])
    txt = f"JSON сброшен\n{getJson()}\n{getJson(1)}"
    # Если команда /reset_param
    if hasattr(message, "content_type"):
        await bot.send_message(
            chat_id=message.from_user.id, text=txt, reply_markup=keyboard
        )
    else:
        await bot.edit_message_text(
            chat_id=message.message.chat.id,
            message_id=message.message.message_id,
            text=txt,
            reply_markup=keyboard,
        )


# Вызов меню генераций getGen
@dp.message_handler(commands=["gen"])
@dp.callback_query_handler(text="gen")
async def inl_gen(message: Union[types.Message, types.CallbackQuery]) -> None:
    print("inl_gen")
    keyboard = InlineKeyboardMarkup(inline_keyboard=[getGen(0), getStart(0)])
    # Если команда /gen
    if hasattr(message, "content_type"):
        await bot.send_message(
            chat_id=message.from_user.id, text="Виды генераций", reply_markup=keyboard
        )
    else:
        await bot.edit_message_text(
            chat_id=message.message.chat.id,
            message_id=message.message.message_id,
            text="Виды генераций",
            reply_markup=keyboard,
        )


# Генерация одной картинки
# TODO gen4/gen10
@dp.callback_query_handler(text="gen1")
@dp.callback_query_handler(text="gen4")
@dp.callback_query_handler(text="gen10")
@dp.callback_query_handler(text="gen_hr")
@dp.callback_query_handler(text="gen_hr4")
async def inl_gen1(callback: types.CallbackQuery) -> None:
    print("inl_gen1")
    keyboard = InlineKeyboardMarkup(inline_keyboard=[getGen(0), getStart(0)])
    if callback.data == "gen1":
        dataOrig["batch_size"] = 1
    if callback.data == "gen4" or callback.data == "gen_hr4":
        dataOrig["batch_size"] = 4
    if callback.data == "gen10":
        dataOrig["batch_size"] = 10
    if callback.data == "gen_hr" or callback.data == "gen_hr4":
        dataOrig["enable_hr"] = "true"
        dataOrig["hr_resize_x"] = dataOrig["width"] * 2
        dataOrig["hr_resize_y"] = dataOrig["height"] * 2
    res = api.txt2img(**dataOrig)  # TODO заменить dataOrig на data, исправить костыль
    if dataParams["img_thumb"] == "true" or dataParams["img_thumb"] == "True":
        await bot.send_media_group(
            chat_id=callback.message.chat.id, media=pilToImages(res, "thumbs")
        )
    if dataParams["img_tg"] == "true" or dataParams["img_tg"] == "True":
        await bot.send_media_group(
            chat_id=callback.message.chat.id, media=pilToImages(res, "tg")
        )
    if dataParams["img_real"] == "true" or dataParams["img_real"] == "True":
        await bot.send_media_group(
            chat_id=callback.message.chat.id, media=pilToImages(res, "real")
        )
    await bot.send_message(
        chat_id=callback.message.chat.id,
        text=data["prompt"] + "\n" + str(res.info["all_seeds"]),
        reply_markup=keyboard,
        parse_mode="Markdown",
    )


# Обработчик команды /status
# TODO async
@dp.message_handler(commands=["status"])
@dp.callback_query_handler(text="status")
async def inl_status(message: Union[types.Message, types.CallbackQuery]) -> None:
    print(inl_status)
    if hasattr(message, "content_type"):
        async with aiohttp.ClientSession() as session:
            async with session.get(
                local + "/sdapi/v1/progress?skip_current_image=false"
            ) as response_progress:
                response_json = await response_progress.json()
                e = round(response_json["eta_relative"], 1)
                time.sleep(2)
                await message.answer(e)
    else:
        e = 1
        while e > 0:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    local + "/sdapi/v1/progress?skip_current_image=false"
                ) as response_progress:
                    response_json2 = await response_progress.json()
                    # print(response_json2)
                    e = round(response_json2["eta_relative"], 1)
                    time.sleep(2)
                    await bot.edit_message_text(
                        chat_id=message.message.chat.id,
                        message_id=message.message.message_id,
                        text=e,
                        reply_markup=getStart(),
                    )


# Обработчик команды /skip TODO
@dp.message_handler(commands=["skip"])
@dp.callback_query_handler(text="skip")
async def inl_skip(message: Union[types.Message, types.CallbackQuery]) -> None:
    print("skip")
    # Создаем сессию
    async with aiohttp.ClientSession() as session:
        # Отправляем POST-запрос ко второму сервису
        async with session.post(local + "/sdapi/v1/skip") as response:
            # Получаем ответ и выводим его
            response_json = await response.json()
            if hasattr(message, "content_type"):
                await message.answer("skip")
            else:
                await bot.edit_message_text(
                    chat_id=message.message.chat.id,
                    message_id=message.message.message_id,
                    text="Виды генераций",
                    reply_markup=getStart(),
                )


# Получить LORA
@dp.message_handler(commands=["get_lora"])
@dp.callback_query_handler(text="get_lora")
async def getLora(message: Union[types.Message, types.CallbackQuery]) -> None:
    print("getLora")
    keyboard = InlineKeyboardMarkup(inline_keyboard=[getScripts(0), getStart(0)])
    # Путь к папке "Lora"
    path = "../../models/Lora"

    # Получаем список файлов в папке
    file_list = os.listdir(path)

    # Фильтруем файлы, выбирая только те, которые заканчиваются на ".safetensors"
    lora_files = [
        file_name for file_name in file_list if file_name.endswith(".safetensors")
    ]

    # Выводим список файлов, отформатированный в нужном формате
    arr = ""
    for file_name in lora_files:
        name = file_name.replace(".safetensors", "")
        arr = arr + f"`<lora:{name}:1>`\n\n"
    if hasattr(message, "content_type"):
        await bot.send_message(
            chat_id=message.from_user.id,
            text=arr,
            reply_markup=keyboard,
            parse_mode="Markdown",
        )
    else:
        await bot.edit_message_text(
            chat_id=message.message.chat.id,
            message_id=message.message.message_id,
            text="Список LORA\n" + arr,
            reply_markup=keyboard,
            parse_mode="Markdown",
        )


# Ввели любой текст
@dp.message_handler(lambda message: True)
async def change_json(message: types.Message):
    print("change_json")
    keyboard = InlineKeyboardMarkup(inline_keyboard=[getSet(0), getStart(0)])
    text = message.text
    nam = text.split()[0][1:]  # txt из /txt 321
    state_names = [attr for attr in dir(Form) if isinstance(getattr(Form, attr), State)]
    args = message.get_args()  # это 321, когда ввели /txt 321
    # Поиск команд из data
    if nam in state_names:
        if args == "":
            await message.answer("Напиши любое " + nam)
            if nam in state_names:
                await getattr(Form, nam).set()
            else:
                print("Ошибка какая-то")
        else:
            # /txt 321 пишем 321 в data['txt']
            data[nam] = args
            # TODO answer поменять на edit_text
            await message.answer(
                f"JSON параметры:\n{getJson()}\n{getJson(1)}", reply_markup=keyboard
            )
    else:
        if args != None:
            # /seed2img + только буквы
            if nam == "seed2img" and args.isalpha():
                await message.answer("Введи seed")
            # /seed2img
            if nam == "seed2img" and args == "":
                await message.answer("Введи seed")
            # /seed2img 000-321
            if nam == "seed2img" and (
                args.isdigit()
                or ("-" in args and all(s.isdigit() for s in args.split("-")))
            ):
                await sendImagesFromSeed(args, message)
        if text != "" and args == None:
            # цифры или цифры с минусом
            if text.isdigit() or (
                "-" in text and all(s.isdigit() for s in text.split("-"))
            ):
                print("")
            # только буквы TODO optimize
            if (
                not text.isdigit() or (not "-" in text and not text.isdigit())
            ) and not (
                text.isdigit()
                or ("-" in text and all(s.isdigit() for s in text.split("-")))
            ):
                data["prompt"] = message.text
                await message.answer(
                    f"Записали промпт. JSON параметры:\n{getJson()}\n{getJson(1)}",
                    reply_markup=keyboard,
                )


# Ввели ответ на change_json
@dp.message_handler(state=Form)
async def answer_handler(message: types.Message, state: FSMContext):
    keyboard = InlineKeyboardMarkup(inline_keyboard=[getSet(0), getStart(0)])
    current_state = await state.get_state()  # Form:команда
    txt = message.text
    for key, val in dataParams.items():
        if current_state == "Form:" + key:
            dataParams[key] = txt
            break
    for key, val in data.items():
        if current_state == "Form:" + key:
            data[key] = txt
            break
    await state.reset_state()
    await message.answer(
        f"JSON параметры:\n{getJson()}\n{getJson(1)}", reply_markup=keyboard
    )


# -------- BOT POLLING ----------
if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)

# -------- COPYRIGHT ----------
# Мишген
# join https://t.me/mishgenai
