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
import inspect
from translate import Translator

API_TOKEN = "TOKEN_HERE"

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

# Получаем список аргументов функции api.txt2img и возвращаем JSON {"/prompt": "","/seed": "-1",...}
def getAttrtxt2img():
    #argspec = inspect.getfullargspec(api.txt2img)
    #defaults = argspec.defaults or []
    #args = argspec.args[1:]
    #values = list(defaults) + [None] * (len(args) - len(defaults))
    #params = {arg: str(value) if value is not None else "" for arg, value in zip(args, values)}
    spec = inspect.getfullargspec(api.txt2img)
    arguments = spec.args
    values = [spec.defaults[i] if i >= (len(spec.defaults) or 0)*-1 else None for i in range(-1, (-1)*(len(arguments)+1), -1)][::-1]
    params = {arg: value for arg, value in zip(arguments, values) if value is not None}
    params = {arg: json.loads(value) if isinstance(value, str) and value.startswith(('{', '[')) else json.loads(json.dumps(value)) if value is not None else None for arg, value in params.items()}
    return params

def get_argspec_json2(function):
    spec = inspect.getfullargspec(function)
    arguments = spec.args
    # Формирование списка значений аргументов функции
    # значения берутся из списка defaults (если они определены)
    # если значение не определено, используется тип по умолчанию
    values = [spec.defaults[i] if i >= (len(spec.defaults) or 0)*-1 else None for i in range(-1, (-1)*(len(arguments)+1), -1)][::-1]
    # Формирование словаря аргументов и их значений
    params = {arg: value for arg, value in zip(arguments, values) if value is not None}

    # Конвертация значений в соответствующие типы
    params = {arg: json.loads(value) if isinstance(value, str) and value.startswith(('{', '[')) else json.loads(json.dumps(value)) if value is not None else None for arg, value in params.items()}
    # Формирование словаря типов данных аргументов
    types = {arg: str(type(value).__name__) for arg, value in params.items()}

    # Конвертация словарей в JSON
    return json.dumps({'params': params, 'arg_types': types})

#print(get_argspec_json(api.txt2img))

# -------- GLOBAL ----------
host = "127.0.0.1"
port = "7861"
# https://github.com/mix1009/sdwebuiapi
api = webuiapi.WebUIApi(host=host, port=port)
# TODO --share used shared link. https://123456.gradio.live/docs does not work
local = "http://" + host + ":" + port
process = None
sd = "❌"

data = getAttrtxt2img()
dataParams = {"img_thumb": "true", "img_tg": "true", "img_real": "true"}
dataOld = data.copy()
dataOldParams = dataParams.copy()
dataOrig = data.copy()

# -------- FUNCTIONS ----------
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
            media_group.append(types.InputMediaPhoto(media=img_byte_arr, caption=seed))
        i = i + 1
    return media_group

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

# -------- MENU ----------
# Стартовое меню
def getKeyboard(keysArr, returnAll):
    keys = keysArr
    keyAll = InlineKeyboardMarkup(inline_keyboard=[keys])
    if returnAll == 1:
        return keyAll
    else:
        return keys

def getStart(returnAll=1) -> InlineKeyboardMarkup:
    keysArr = [
        InlineKeyboardButton("sd" + sd,  callback_data="sd"),
        InlineKeyboardButton("opt",      callback_data="opt"),
        InlineKeyboardButton("gen",      callback_data="gen"),
        InlineKeyboardButton("skip",     callback_data="skip"),
        InlineKeyboardButton("progress", callback_data="progress"),
        InlineKeyboardButton("help",     callback_data="help"),
    ]
    return (getKeyboard(keysArr, returnAll))

# Меню генераций
def getGen(returnAll=1) -> InlineKeyboardMarkup:
    keysArr = [
        InlineKeyboardButton("gen1",    callback_data="gen1"),
        InlineKeyboardButton("gen4",    callback_data="gen4"),
        InlineKeyboardButton("gen10",   callback_data="gen10"),
        InlineKeyboardButton("gen_hr",  callback_data="gen_hr"),
        InlineKeyboardButton("gen_hr4", callback_data="gen_hr4"),
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
        InlineKeyboardButton("seed2img", callback_data="seed2img"),
    ]
    return (getKeyboard(keysArr, returnAll))


# Меню настроек
def getSet(returnAll=1) -> InlineKeyboardMarkup:
    keysArr = [
        InlineKeyboardButton("change_param", callback_data="change_param"),
        InlineKeyboardButton("reset_param", callback_data="reset_param"),
    ]
    return (getKeyboard(keysArr, returnAll))

# Меню настроек
def getPrompt(returnAll=1) -> InlineKeyboardMarkup:
    keysArr = [InlineKeyboardButton("random_prompt", callback_data="random_prompt")]
    return (getKeyboard(keysArr, returnAll))

# Меню генераций
def getGen(returnAll=1) -> InlineKeyboardMarkup:
    keysArr = [
        InlineKeyboardButton("gen1", callback_data="gen1"),
        InlineKeyboardButton("gen4", callback_data="gen4"),
        InlineKeyboardButton("gen10", callback_data="gen10"),
        InlineKeyboardButton("gen_hr", callback_data="gen_hr"),
        InlineKeyboardButton("gen_hr4", callback_data="gen_hr4"),
    ]
    return (getKeyboard(keysArr, returnAll))

# Меню текста
def getTxt():
    return "/start /opt /gen /skip /status /seed2img /help"

# Проверка связи до запущенной локальной SD с nowebui
def ping(status: str):
    url = local + "/docs"
    while True:
        try:
            r = requests.get(url, timeout=3)
            r.raise_for_status()
            code = r.status_code
            print(code)
            if (status == "stop" and code != 200) or (status == "start" and code == 200):
                return True
        except requests.exceptions.RequestException as err:
            print("Error:", err)
        time.sleep(3)


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

# Генерация изображений
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
    print(dataOrig)
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
    print(data["prompt"])
    await bot.send_message(
        chat_id=callback.message.chat.id,
        text=data["prompt"] + "\n" + str(res.info["all_seeds"]),
        reply_markup=keyboard,
        parse_mode="Markdown",
    )

@dp.callback_query_handler(text="random_prompt")
async def random_prompt(callback: types.CallbackQuery) -> None:
    await bot.send_message(chat_id=callback.from_user.id, text=get_random_prompt())

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

@dp.message_handler(commands=["test"])
async def cmd_test(message: Union[types.Message, types.CallbackQuery]) -> None:
    print("cmd_test")
    #print(api.get_options())
    #options = {}
    #options['outdir_txt2img_samples'] = '../../outputs/txt2img-images'
    #api.set_options(options)
    #print(api.get_options())

    #getAttrtxt2img() # options default
    #options = {}
    #options['ttt'] = '11'
    #api.set_options(options)
    #print(api.get_options())
    translator = Translator(to_lang="en")

    text = input("Введите текст на русском языке: ")

    translation = translator.translate(text)

    print("Перевод на английский язык: ", translation)

@dp.message_handler(commands=["start2"])
async def cmd_start(message: Union[types.Message, types.CallbackQuery]) -> None:
    print("cmd_start")
    subprocess.Popen(
        ["python", "../../launch.py", "--nowebui", "--xformers"]
    )

@dp.message_handler(commands=["test2"])
async def cmd_test2(message: Union[types.Message, types.CallbackQuery]) -> None:
    print("cmd_test2")
    print(getAttrtxt2img()['enable_hr'])
    print(data['enable_hr'])



# -------- BOT POLLING ----------
if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)

# -------- COPYRIGHT ----------
# Мишген
# join https://t.me/mishgenai
