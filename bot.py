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
import random
from datetime import datetime
import aiohttp
from typing import Union
from PIL import Image
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import inspect
from translate import Translator

API_TOKEN = "900510503:AAG5Xug_JEERhKlf7dpOpzxXcJIzlTbWX1M"

bot = Bot(token=API_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)

# Получаем список аргументов функции api.txt2img и возвращаем JSON {"/prompt": "","/seed": "-1",...}
def getAttrtxt2img():
    spec = inspect.getfullargspec(api.txt2img)
    arguments = spec.args
    values = [spec.defaults[i] if i >= (len(spec.defaults) or 0)*-1 else None for i in range(-1, (-1)*(len(arguments)+1), -1)][::-1]
    params = {arg: value for arg, value in zip(arguments, values) if value is not None}
    params = {arg: json.loads(value) if isinstance(value, str) and value.startswith(('{', '[')) else json.loads(json.dumps(value)) if value is not None else None for arg, value in params.items()}
    return params

# -------- GLOBAL ----------
formatted_date = datetime.today().strftime("%Y-%m-%d")
host = "127.0.0.1"
port = "7861"
# https://github.com/mix1009/sdwebuiapi
api = webuiapi.WebUIApi(host=host, port=port)
# TODO --share used shared link. https://123456.gradio.live/docs does not work
local = "http://" + host + ":" + port
process = None
sd = "❌"

data = getAttrtxt2img()
data['prompt'] = 'cat in space' # Ý
data['steps'] = 15
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
# Запуск SD через subprocess и запись в глобальную переменную process
def start_sd():
    global process, sd
    if not process:
        print('start_process start_sd')
        process = subprocess.Popen(["python", "../../launch.py", "--nowebui", "--xformers"])
        sd = "✅"

def stop_sd():
    global process, sd
    if process:
        print('stop_process stop_sd')
        process.terminate()
        process = None
        sd = "❌"

def submit_get(url: str, data: dict):
    return requests.get(url, data=json.dumps(data))

def pilToImages(res, typeImages="tg"):
    media_group = []
    imagesAll = res.images
    if len(res.images) == 1:
        i = 0
    if len(res.images) > 1:
        i = -1
    for image in imagesAll:
        # костыль для отсечения первой картинки с гридами
        #if i == -1:
        #    i = i + 1
        #    continue
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

def rnd_prmt_lxc():
    txt = random.choice(submit_get('https://lexica.art/api/v1/search?q='+data['prompt'], '').json()['images'])['prompt']
    return txt

# Translate
def translateRuToEng(text):
    translator = Translator(from_lang="ru", to_lang="en")
    return translator.translate(text)

# Вывод прогресса в заменяемое сообщение
async def getProgress(msgTime):
    points = '.'
    while True:
        # TODO aiogram.utils.exceptions.MessageToEditNotFound: Message to edit not found
        await asyncio.sleep(1)
        proc = round(api.get_progress()['progress']*100)
        points = '.' * (proc % 9)
        await bot.edit_message_text(
            chat_id=msgTime.chat.id,
            message_id=msgTime.message_id,
            text=str(proc)+'% ' + points
        )

# -------- MENU ----------
# Стартовое меню
def getKeyboard(keysArr, returnAll):
    keys = keysArr
    keyAll = InlineKeyboardMarkup(inline_keyboard=[keys])
    if returnAll == 1:
        return keyAll
    else:
        return keys

# Стандартное меню
async def getKeyboardUnion(txt, message, keyboard):
    # Если команда /settings
    if hasattr(message, "content_type"):
        await bot.send_message(
            chat_id=message.from_user.id,
            text=txt,
            reply_markup=keyboard,
            parse_mode="Markdown"
        )
    else:
        await bot.edit_message_text(
            chat_id=message.message.chat.id,
            message_id=message.message.message_id,
            text=txt,
            reply_markup=keyboard,
            parse_mode="Markdown"
        )

def getStart(returnAll=1) -> InlineKeyboardMarkup:
    keysArr = [
        InlineKeyboardButton(sd + "sd",  callback_data="sd"),
        InlineKeyboardButton("opt",      callback_data="opt"),
        InlineKeyboardButton("gen",      callback_data="gen"),
        InlineKeyboardButton("skip",     callback_data="skip"),
        InlineKeyboardButton("status",   callback_data="status"),
        InlineKeyboardButton("help",     callback_data="help"),
    ]
    return (getKeyboard(keysArr, returnAll))

# Меню опций
def getOpt(returnAll=1) -> InlineKeyboardMarkup:
    keysArr = [
        InlineKeyboardButton("settings", callback_data="settings"),
        InlineKeyboardButton("scripts", callback_data="scripts"),
        InlineKeyboardButton("prompt", callback_data="prompt"),
    ]
    return (getKeyboard(keysArr, returnAll))


# Меню скриптов
def getScripts(returnAll=1) -> InlineKeyboardMarkup:
    keysArr = [
        InlineKeyboardButton("get_lora", callback_data="get_lora"),
    ]
    return (getKeyboard(keysArr, returnAll))


# Меню настроек
def getSet(returnAll=1) -> InlineKeyboardMarkup:
    keysArr = [
        InlineKeyboardButton("change_param", callback_data="change_param"),
        InlineKeyboardButton("reset_param", callback_data="reset_param"),
    ]
    return (getKeyboard(keysArr, returnAll))

# Меню промпта
def getPrompt(returnAll=1) -> InlineKeyboardMarkup:
    keysArr = [InlineKeyboardButton("random_prompt", callback_data="random_prompt"),
               InlineKeyboardButton("lxc_prompt", callback_data="lxc_prompt"),]
    return (getKeyboard(keysArr, returnAll))

# Меню текста
def getTxt():
    return "/start /opt /gen /skip /status /help"

# -------- COMMANDS ----------

# start или help
@dp.callback_query_handler(text="help")
@dp.message_handler(commands=["help"])
@dp.message_handler(commands=["start"])
async def cmd_start(message: Union[types.Message, types.CallbackQuery]) -> None:
    print("cmd_start")
    txt = "Это бот для локального запуска SD\n" + getTxt()
    await getKeyboardUnion(txt, message, getStart())

# TODO optimize
# Запуск/Остановка SD. Завязываемся на глобальную иконку sd
@dp.callback_query_handler(text="sd")
async def inl_sd(callback: types.CallbackQuery) -> None:
    print("inl_sd")
    global sd
    if sd == '✅':
        stop_sd()
        sd = "⌛"
        await callback.message.edit_text(
            "Останавливаем SD\n" + getTxt(), reply_markup=getStart()
        )
        sd = '❌'
        await callback.message.edit_text(
            "SD остановлена\n" + getTxt(), reply_markup=getStart()
        )
    else:
        start_sd()
        sd = "⌛"
        await callback.message.edit_text(
            "Запускаем SD\n" + getTxt(), reply_markup=getStart()
        )
        url = 'http://127.0.0.1:7861/docs'
        n = 0
        while n != 200:
            time.sleep(2)
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
        sd = "✅"
        await callback.message.edit_text(
            "SD запущена\n" + getTxt(), reply_markup=getStart()
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
    keyboard = InlineKeyboardMarkup(inline_keyboard=[getSet(0), getOpt(0), getStart(0)])
    txt = f"JSON сброшен\n{getJson()}\n{getJson(1)}"
    await getKeyboardUnion(txt, message, keyboard)

# Обработчик команды /status
@dp.message_handler(commands=["status"])
@dp.callback_query_handler(text="status")
async def inl_status(message: Union[types.Message, types.CallbackQuery]) -> None:
    print(inl_status)
    print(api.get_progress()["eta_relative"])

@dp.message_handler(commands=["gen"])
@dp.callback_query_handler(text="gen")
async def inl_gen(message: Union[types.Message, types.CallbackQuery]) -> None:
    if hasattr(message, "content_type"):
        chatId = message.chat.id
    else:
        chatId = message.message.chat.id
    keyboard = InlineKeyboardMarkup(inline_keyboard=[getOpt(0), getSet(0), getStart(0)])
    msgTime = await bot.send_message(
        chat_id=chatId,
        text='Начали'
    )
    # Включаем асинхрон, чтоб заработал await api.txt2img
    data["use_async"] = "True"
    asyncio.create_task(getProgress(msgTime))
    res = await api.txt2img(**data)
    if dataParams["img_thumb"] == "true" or dataParams["img_thumb"] == "True":
        await bot.send_media_group(
            chat_id=chatId, media=pilToImages(res, "thumbs")
        )
    if dataParams["img_tg"] == "true" or dataParams["img_tg"] == "True":
        await bot.send_media_group(
            chat_id=chatId, media=pilToImages(res, "tg")
        )
    if dataParams["img_real"] == "true" or dataParams["img_real"] == "True":
        await bot.send_media_group(
            chat_id=chatId, media=pilToImages(res, "real")
        )
    await bot.send_message(
        chat_id=chatId,
        text=data["prompt"] + "\n" + str(res.info["all_seeds"]),
        reply_markup=keyboard,
        parse_mode="Markdown",
    )
    # Удаляем сообщение с прогрессом
    await bot.delete_message(chat_id=msgTime.chat.id, message_id=msgTime.message_id)

# Получить меню действий с промптами
@dp.callback_query_handler(text="prompt")
async def cmd_prompt(message: Union[types.Message, types.CallbackQuery]) -> None:
    print("cmd_prompt")
    keyboard = InlineKeyboardMarkup(inline_keyboard=[getPrompt(0), getOpt(0), getStart(0)])
    await getKeyboardUnion("Опции", message, keyboard)

# Получить опции
@dp.message_handler(commands=["opt"])
@dp.callback_query_handler(text="opt")
async def cmd_opt(message: Union[types.Message, types.CallbackQuery]) -> None:
    print("cmd_opt")
    keyboard = InlineKeyboardMarkup(inline_keyboard=[getOpt(0), getStart(0)])
    await getKeyboardUnion("Опции", message, keyboard)

# Вызов settings
@dp.message_handler(commands=["settings"])
@dp.callback_query_handler(text="settings")
async def inl_settings(message: Union[types.Message, types.CallbackQuery]) -> None:
    print("inl_settings")
    keyboard = InlineKeyboardMarkup(inline_keyboard=[getSet(0), getOpt(0), getStart(0)])
    await getKeyboardUnion("Настройки", message, keyboard)

# Вызов script
@dp.message_handler(commands=["scripts"])
@dp.callback_query_handler(text="scripts")
async def inl_scripts(message: Union[types.Message, types.CallbackQuery]) -> None:
    print("inl_scripts")
    keyboard = InlineKeyboardMarkup(inline_keyboard=[getScripts(0), getOpt(0), getStart(0)])
    await getKeyboardUnion("Скрипты", message, keyboard)

# Вызов change_param
@dp.callback_query_handler(text="change_param")
async def inl_change_param(callback: types.CallbackQuery) -> None:
    print("inl_change_param")
    keyboard = InlineKeyboardMarkup(inline_keyboard=[getSet(0), getOpt(0), getStart(0)])
    json_list = [f"/{key} = {value}" for key, value in data.items()]
    json_list_params = [f"/{key} = {value}" for key, value in dataParams.items()]
    json_str = "\n".join(json_list)
    json_str_params = "\n".join(json_list_params)
    await callback.message.edit_text(
        f"JSON параметры:\n{json_str}\n{json_str_params}", reply_markup=keyboard
    )

# Получить LORA
@dp.message_handler(commands=["get_lora"])
@dp.callback_query_handler(text="get_lora")
async def getLora(message: Union[types.Message, types.CallbackQuery]) -> None:
    print("getLora")
    keyboard = InlineKeyboardMarkup(inline_keyboard=[getScripts(0), getOpt(0), getStart(0)])
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
    await getKeyboardUnion(arr, message, keyboard)

# Рандомный промпт с lexica.art на основе data['prompt']
@dp.message_handler(commands=["lxc_prompt"])
@dp.callback_query_handler(text="lxc_prompt")
async def get_lxc_prompt(message: Union[types.Message, types.CallbackQuery]) -> None:
    print("get_lxc_prompt")
    keyboard = InlineKeyboardMarkup(inline_keyboard=[getPrompt(0), getOpt(0), getStart(0)])
    txt = rnd_prmt_lxc()
    await getKeyboardUnion(txt, message, keyboard)

@dp.callback_query_handler(text="random_prompt")
async def random_prompt(callback: types.CallbackQuery) -> None:
    keyboard = InlineKeyboardMarkup(inline_keyboard=[getPrompt(0), getOpt(0), getStart(0)])
    await getKeyboardUnion(get_random_prompt(), callback, keyboard)

# Ввели любой текст
@dp.message_handler(lambda message: True)
async def change_json(message: types.Message):
    print("change_json")
    keyboard = InlineKeyboardMarkup(inline_keyboard=[getSet(0), getStart(0)])
    text = message.text
    print(514)
    print(text)
    nam = text.split()[0][1:]  # txt из /txt 321
    state_names = [attr for attr in dir(Form) if isinstance(getattr(Form, attr), State)]
    print(516)
    print(nam)
    print(state_names)
    args = message.get_args()  # это 321, когда ввели /txt 321
    # Поиск команд из data
    if nam in state_names:
        print(524)
        if args == "":
            print(526)
            await message.answer("Напиши любое " + nam)
            print(528)
            if nam in state_names:
                await getattr(Form, nam).set()
            else:
                print("Ошибка какая-то")
        else:
            # /txt 321 пишем 321 в data['txt']
            print(533)
            data[nam] = args
            # TODO answer поменять на edit_text
            await message.answer(
                f"JSON параметры:\n{getJson()}\n{getJson(1)}", reply_markup=keyboard
            )
    else:
        data["prompt"] = message.text
        await message.answer(
            f"Записали промпт. JSON параметры:\n{getJson()}\n{getJson(1)}",
            reply_markup=keyboard,
        )

# Ввели ответ на change_json
@dp.message_handler(state=Form)
async def answer_handler(message: types.Message, state: FSMContext):
    print('answer_handler')
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
