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
    InputMediaPhoto,
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
import base64

# from https://t.me/BotFather
API_TOKEN = "TOKEN_HERE"

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
data['sampler_name'] = 'Euler a'
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

async def stop_sd():
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

# get settings. TODO - cut 4000 symbols
def get_prompt_settings():
    prompt = data['prompt']
    cfg_scale = data['cfg_scale']
    width = data['width']
    height = data['height']
    steps = data['steps']
    negative_prompt = data['negative_prompt']
    sampler_name = data['sampler_name']
    sd_model_checkpoint = api.get_options()['sd_model_checkpoint']
    txt = f"prompt = <code>{prompt}</code>\nsteps = {steps} \ncfg_scale = {cfg_scale} \nwidth = {width} \nheight = {height} \nsampler_name = {sampler_name} \nsd_model_checkpoint = {sd_model_checkpoint} \nnegative_prompt = <code>{negative_prompt}</code> "
    return txt

# Translate
def translateRuToEng(text):
    translator = Translator(from_lang="ru", to_lang="en")
    return translator.translate(text)

# Вывод прогресса в заменяемое сообщение
async def getProgress(msgTime):
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
#TODO
async def getProgress2(msgTime):
    points = '.'
    while True:
        # TODO aiogram.utils.exceptions.MessageToEditNotFound: Message to edit not found
        await asyncio.sleep(2)
        print(187)
        print(api.get_progress())
        proc = round(api.get_progress()['progress']*100)
        points = '.' * (proc % 9)
        #await bot.edit_message_text(
        #    chat_id=msgTime.chat.id,
        #    message_id=msgTime.message_id,
        #    #text=str(proc)+'% ' + points# + str(int(time.time() * 1000))
        #    text=str(proc)+'% ' + points + '\n'+str(api.get_progress()['eta_relative']) + '\n'+str(api.get_progress()['state'])
        #)
        #await bot.send_message(
        #    chat_id=msgTime.chat.id,
        #    text='проверка'
        #)
        #image = base64.b64decode(api.get_progress()['current_image'])
        #image_buffer = io.BytesIO()
        #image.save(image_buffer, format="PNG")
        #image_buffer.seek(0)
        #img = Image.open(image_buffer)
        #width, height = img.size
        #ratio = min(256 / width, 256 / height)
        #new_size = (round(width * ratio), round(height * ratio))
        #img = img.resize(new_size)
        #img_byte_arr = io.BytesIO()
        #img.save(img_byte_arr, format="PNG")
        #img_byte_arr.seek(0)
        #image_data = base64.b64decode(api.get_progress()['current_image'])
        #input_file = InputFile(image_data, filename="image.png")
        #input_media = InputMediaPhoto(input_file)
        #await bot.edit_message_media(chat_id=msgTime.chat.id, message_id=msgTime.message_id, media=input_media,
        #                             text=str(proc)+'% ' + points + '\n'+str(api.get_progress()['eta_relative']) + '\n'+str(api.get_progress()['state'])
        #                             )
        #await bot.send_media_group(
        #    chat_id=msgTime.chat.id, media=pilToImages(api.get_progress()['current_image'], "thumbs")
        #)

        #img_base64 = "iVBORw0KGgoAAAANSUhEUgAAAlgAAAAmAhAAAADZI+25AAAClklEQVR4nO3UwQ2AMAADwMu33Hr/gWCFnltl9Ydu0v+PiH47qajzqajvOmrx7qssxDT48mi1lKCTpSEwRWhIKgbDBA9ABViSKgMBUxSIoAAD+CvP/Z6f14klgADwRv/vKd+zXZBIg2Azgv+5fjvTtT/ran2MJS5dSVTzxg/gkAF/nXUOLj2T539nCZcLDWsAYAAAAASUVORK5CYII="
        media_group = []
        image_base64 = api.get_progress()['current_image']
        image = base64.b64decode(image_base64)
        image_buffer = io.BytesIO(image)
        img = Image.open(image_buffer)
        img.save(image_buffer, format="PNG")
        image_buffer.seek(0)
        #media_group.append(types.InputMediaPhoto(media=image_buffer.getvalue(), caption='1121'))
        #await bot.send_media_group(chat_id=msgTime.chat.id, media=media_group)

        # Сохраняем изображение на диск
        img_path = 'image.png'
        with open(img_path, 'wb') as f:
            f.write(image)

        # Открываем изображение в виде InputFile
        media_group.append(types.InputMediaPhoto(img_path, caption='1121'))
        os.remove(img_path)

        # Отправляем медиа-группу
        await bot.send_media_group(chat_id=msgTime.chat.id, media=media_group)
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
async def getKeyboardUnion(txt, message, keyboard, parse_mode = 'Markdown'):
    # Если команда с слешем
    if hasattr(message, "content_type"):
        await bot.send_message(
            chat_id=message.from_user.id,
            text=txt,
            reply_markup=keyboard,
            parse_mode=parse_mode
        )
    else:
        await bot.edit_message_text(
            chat_id=message.message.chat.id,
            message_id=message.message.message_id,
            text=txt,
            reply_markup=keyboard,
            parse_mode=parse_mode
        )

def getStart(returnAll = 1) -> InlineKeyboardMarkup:
    keysArr = [
        InlineKeyboardButton(sd + "sd",  callback_data="sd"),
        InlineKeyboardButton("opt",      callback_data="opt"),
        InlineKeyboardButton("gen",      callback_data="gen"),
        InlineKeyboardButton("skip",     callback_data="skip"),
        InlineKeyboardButton("help",     callback_data="help"),
    ]
    return (getKeyboard(keysArr, returnAll))

# Меню опций
def getOpt(returnAll = 1) -> InlineKeyboardMarkup:
    keysArr = [
        InlineKeyboardButton("sttngs",  callback_data="sttngs"),
        InlineKeyboardButton("scrpts",  callback_data="scrpts"),
        InlineKeyboardButton("mdl",     callback_data="mdl"),
        InlineKeyboardButton("smplr",   callback_data="smplr"),
        InlineKeyboardButton("hr",      callback_data="hr"),
        InlineKeyboardButton("prompt",  callback_data="prompt"),
    ]
    return (getKeyboard(keysArr, returnAll))


# Меню скриптов
def getScripts(returnAll = 1) -> InlineKeyboardMarkup:
    keysArr = [
        InlineKeyboardButton("get_lora", callback_data="get_lora"),
        InlineKeyboardButton("rnd_mdl",  callback_data="rnd_mdl"),
        InlineKeyboardButton("rnd_smp",  callback_data="rnd_smp"),
    ]
    return (getKeyboard(keysArr, returnAll))


# Меню настроек
def getSet(returnAll = 1) -> InlineKeyboardMarkup:
    keysArr = [
        InlineKeyboardButton("change_param", callback_data="change_param"),
        InlineKeyboardButton("reset_param",  callback_data="reset_param"),
    ]
    return (getKeyboard(keysArr, returnAll))

# Меню галочек Да/Нет
def getYesNo(returnAll = 1, nam = '') -> InlineKeyboardMarkup:
    keysArr = [
        InlineKeyboardButton("✅", callback_data="✅"+nam),
        InlineKeyboardButton("❌", callback_data="❌"+nam)
    ]
    return (getKeyboard(keysArr, returnAll))

# Меню промпта
def getPrompt(returnAll = 1) -> InlineKeyboardMarkup:
    keysArr = [InlineKeyboardButton("get",           callback_data="get"),
               InlineKeyboardButton("random_prompt", callback_data="random_prompt"),
               InlineKeyboardButton("lxc_prompt",    callback_data="lxc_prompt"),]
    return (getKeyboard(keysArr, returnAll))

# Меню текста
def getTxt():
    return "/start /opt /gen /skip /stop /help"

def set_array(arrAll, itemArr, callback_data, useIn = 1):
    print('set_array')
    print(arrAll)
    arr = []
    arr2 = []
    i = 1
    for item in arrAll:
        if useIn == 1:
            arrayIn = item[itemArr]
        else:
            arrayIn = item
        arr.append(InlineKeyboardButton(arrayIn, callback_data=callback_data+'|'+arrayIn))
        if i % 3 == 0:
             arr2.append(arr)
             arr = []
        i += 1
    if arr != []:
        arr2.append(arr)
    return arr2

# get all models from stable-diffusion-webui\models\Stable-diffusion
def get_models():
    models = api.get_sd_models()
    return set_array(models, 'model_name', 'models')

# get samplers
def get_samplers_list():
    samplers = api.get_samplers()
    return set_array(samplers, 'name', 'samplers')

# get hr
def get_hr_list():
    hrs = [str(choice.value) for choice in webuiapi.HiResUpscaler]
    return set_array(hrs, 'hr', 'hrs', 0)

# random
async def rnd_script(message, typeScript):
    keyboard = InlineKeyboardMarkup(inline_keyboard=[getOpt(0), getSet(0), getStart(0)])
    if hasattr(message, "content_type"):
        chatId = message.chat.id
    else:
        chatId = message.message.chat.id
    if typeScript == 'models':
        elements = api.util_get_model_names()
    else:
        elements = api.get_samplers()
    numbers = list(range(len(elements)))
    random.shuffle(numbers)
    await bot.send_message(
        chat_id=chatId,
        text='Цикл по '+str(len(elements)) + (' моделям' if typeScript == 'models' else ' семплерам')
    )

    for i, number in enumerate(numbers):
        if typeScript == 'models':
            api.util_wait_for_ready()
            api.util_set_model(elements[number])
        else:
            options = {}
            options['sampler_name'] = elements[number]['name']
            api.set_options(options)
            data['sampler_name'] = elements[number]['name']  # Ý
        data["use_async"] = False
        res = api.txt2img(**data)
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
            text=elements[number] if typeScript == 'models' else elements[number]['name']
        )
    await bot.send_message(
        chat_id=chatId,
        text="Готово \n"+str(data['prompt']) +
                    "\n cfg_scale = " + str(data['cfg_scale']) +
                    "\n width = " + str(data['width']) +
                    "\n height = " + str(data['height']) +
                    "\n steps = " + str(data['steps']) +
                    "\n negative = " + str(data['negative_prompt'])
        ,
        reply_markup=keyboard
    )

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
@dp.message_handler(commands=["stop"])
@dp.callback_query_handler(text="sd")
async def inl_sd(message: Union[types.Message, types.CallbackQuery]) -> None:
    print("inl_sd")
    global sd
    if hasattr(message, "content_type"):
        if message.text == '/stop':
            await inl_skip(message)
            await stop_sd()
            await bot.send_message(
                chat_id=message.chat.id,
                text = "Останавливаем SD\n" + getTxt(),
                reply_markup=getStart()
            )
    else:
        if sd == '✅':
            await stop_sd()
            sd = "⌛"
            await message.message.edit_text(
                "Останавливаем SD\n" + getTxt(), reply_markup=getStart()
            )
            sd = '❌'
            await message.message.edit_text(
                "SD остановлена\n" + getTxt(), reply_markup=getStart()
            )
        else:
            start_sd()
            sd = "⌛"
            await message.message.edit_text(
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
            await message.message.edit_text(
                "SD запущена\n" + getTxt(), reply_markup=getStart()
            )

# Вызов reset_param, сброс JSON
@dp.message_handler(commands=["reset_param"])
@dp.callback_query_handler(text="reset_param")
async def inl_reset_param(message: Union[types.Message, types.CallbackQuery]) -> None:
    print("inl_reset_param")
    global data
    global dataParams
    global dataOld
    global dataOldParams
    data = dataOld
    dataParams = dataOldParams
    keyboard = InlineKeyboardMarkup(inline_keyboard=[getSet(0), getOpt(0), getStart(0)])
    txt = f"JSON сброшен\n{getJson()}\n{getJson(1)}"
    await getKeyboardUnion(txt, message, keyboard, '')

# Обработчик команды /skip
@dp.message_handler(commands=["skip"])
@dp.callback_query_handler(text="skip")
async def inl_skip(message: Union[types.Message, types.CallbackQuery]) -> None:
    print('inl_skip')
    # Создаем сессию
    async with aiohttp.ClientSession() as session:
        # Отправляем POST-запрос ко второму сервису
        async with session.post(local + "/sdapi/v1/skip"):
            # Получаем ответ и выводим его
            #await response.json()
            if hasattr(message, "content_type"):
                await message.answer("skip")
            else:
                await bot.edit_message_text(
                    chat_id=message.message.chat.id,
                    message_id=message.message.message_id,
                    text="Пропущено",
                    reply_markup=getStart(),
                )

@dp.message_handler(commands=["gen"])
@dp.callback_query_handler(text="gen")
async def inl_gen(message: Union[types.Message, types.CallbackQuery]) -> None:
    if hasattr(message, "content_type"):
        chatId = message.chat.id
    else:
        chatId = message.message.chat.id
    global sd
    if sd == '✅':
        keyboard = InlineKeyboardMarkup(inline_keyboard=[getSet(0), getOpt(0), getStart(0)])
        msgTime = await bot.send_message(
            chat_id=chatId,
            text='Начали'
        )
        # Включаем асинхрон, чтоб заработал await api.txt2img
        data["use_async"] = "True"
        asyncio.create_task(getProgress(msgTime))
        # TODO try catch if wrong data
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
    else:
        keyboard = InlineKeyboardMarkup(inline_keyboard=[getSet(0), getOpt(0), getStart(0)])
        await getKeyboardUnion("Turn on SD"+sd, message, keyboard)

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

# Вызов sttngs
@dp.message_handler(commands=["sttngs"])
@dp.callback_query_handler(text="sttngs")
async def inl_sttngs(message: Union[types.Message, types.CallbackQuery]) -> None:
    print("inl_sttngs")
    keyboard = InlineKeyboardMarkup(inline_keyboard=[getSet(0), getOpt(0), getStart(0)])
    await getKeyboardUnion("Настройки", message, keyboard)

# Вызов script
@dp.message_handler(commands=["scrpts"])
@dp.callback_query_handler(text="scrpts")
async def inl_scrpts(message: Union[types.Message, types.CallbackQuery]) -> None:
    print("inl_scrpts")
    keyboard = InlineKeyboardMarkup(inline_keyboard=[getScripts(0), getOpt(0), getStart(0)])
    await getKeyboardUnion("Скрипты", message, keyboard)

# Вызов get_models
@dp.message_handler(commands=["mdl"])
@dp.callback_query_handler(text="mdl")
async def inl_mdl(message: Union[types.Message, types.CallbackQuery]) -> None:
    print("inl_mdl")
    global sd
    if sd == '✅':
        menu = get_models()
        menu.append(getOpt(0))
        menu.append(getStart(0))
        await getKeyboardUnion("Скрипты", message, InlineKeyboardMarkup(inline_keyboard=menu))
    else:
        keyboard = InlineKeyboardMarkup(inline_keyboard=[getOpt(0), getStart(0)])
        await getKeyboardUnion("Turn on SD"+sd, message, keyboard)

# Вызов get_samplers
@dp.message_handler(commands=["smplr"])
@dp.message_handler(commands=["sampler_name"])
@dp.callback_query_handler(text="smplr")
async def inl_smplr(message: Union[types.Message, types.CallbackQuery]) -> None:
    print("inl_smplr")
    global sd
    if sd == '✅':
        menu = get_samplers_list()
        menu.append(getOpt(0))
        menu.append(getStart(0))
        await getKeyboardUnion("Скрипты", message, InlineKeyboardMarkup(inline_keyboard=menu))
    else:
        keyboard = InlineKeyboardMarkup(inline_keyboard=[getOpt(0), getStart(0)])
        await getKeyboardUnion("Turn on SD"+sd, message, keyboard)

# Вызов get_hr_list
@dp.message_handler(commands=["hr"])
@dp.message_handler(commands=["hr_upscaler"])
@dp.callback_query_handler(text="hr")
async def inl_hr(message: Union[types.Message, types.CallbackQuery]) -> None:
    print("inl_hr")
    global sd
    if sd == '✅':
        menu = get_hr_list()
        menu.append(getOpt(0))
        menu.append(getStart(0))
        await getKeyboardUnion("HiResUpscaler", message, InlineKeyboardMarkup(inline_keyboard=menu))
    else:
        keyboard = InlineKeyboardMarkup(inline_keyboard=[getOpt(0), getStart(0)])
        await getKeyboardUnion("Turn on SD"+sd, message, keyboard)

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

# script random gen from models
@dp.message_handler(commands=["rnd_mdl"])
@dp.callback_query_handler(text='rnd_mdl')
async def inl_rnd_mdl(message: Union[types.Message, types.CallbackQuery]) -> None:
    print('inl_rnd_mdl')
    await rnd_script(message, 'models')


# script random gen from models
@dp.message_handler(commands=["rnd_smp"])
@dp.callback_query_handler(text='rnd_smp')
async def inl_rnd_smp(message: Union[types.Message, types.CallbackQuery]) -> None:
    print('inl_rnd_smp')
    await rnd_script(message, 'samplers')


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

@dp.callback_query_handler(text="get")
async def get_prompt(message: Union[types.Message, types.CallbackQuery]) -> None:
    keyboard = InlineKeyboardMarkup(inline_keyboard=[getPrompt(0), getOpt(0), getStart(0)])
    if sd == '✅':
        await getKeyboardUnion(get_prompt_settings(), message, keyboard, types.ParseMode.HTML)
    else:
        await getKeyboardUnion("Turn on SD"+sd, message, keyboard)

# тыкнули на модельку
@dp.callback_query_handler(text_startswith="models")
async def inl_models(callback: types.CallbackQuery) -> None:
    print('inl_models')
    mdl = callback.data.split("|")[1]
    api.util_set_model(mdl)
    api.util_wait_for_ready()
    menu = get_models()
    menu.append(getOpt(0))
    menu.append(getStart(0))
    await getKeyboardUnion('Теперь модель = ' + str(mdl), callback, InlineKeyboardMarkup(inline_keyboard=menu), '')

# тыкнули на сэмплер
@dp.callback_query_handler(text_startswith="samplers")
async def inl_samplers(callback: types.CallbackQuery) -> None:
    print('inl_samplers')
    smplr = callback.data.split("|")[1]
    options = {}
    options['sampler_name'] = smplr
    api.set_options(options)
    data['sampler_name'] = smplr # Ý
    menu = get_samplers_list()
    menu.append(getOpt(0))
    menu.append(getStart(0))
    await getKeyboardUnion('Теперь сэмплер = ' + str(smplr), callback, InlineKeyboardMarkup(inline_keyboard=menu), '')

# тыкнули на hr_upscaler
@dp.callback_query_handler(text_startswith="hrs")
async def inl_hrs(callback: types.CallbackQuery) -> None:
    print('inl_hrs')
    hrs = callback.data.split("|")[1]
    options = {}
    options['hr_upscaler'] = hrs
    api.set_options(options)
    data['hr_upscaler'] = hrs # Ý
    menu = get_hr_list()
    menu.append(getOpt(0))
    menu.append(getStart(0))
    await getKeyboardUnion('Теперь hr_upscaler = ' + str(hrs), callback, InlineKeyboardMarkup(inline_keyboard=menu), '')

# тыкнули на ✅ или ❌
@dp.callback_query_handler(text_startswith="✅")
@dp.callback_query_handler(text_startswith="❌")
async def inl_yes_no(callback: types.CallbackQuery) -> None:
    print('inl_yes_no')
    keyboard = InlineKeyboardMarkup(inline_keyboard=[getSet(0), getStart(0)])
    if callback.data[:1] == "✅":
        if callback.data[1:] in data.keys():
            data[callback.data[1:]] = 'True'
        if callback.data[1:] in dataParams.keys():
            dataParams[callback.data[1:]] = 'True'
    if callback.data[:1] == "❌":
        if callback.data[1:] in data.keys():
            data[callback.data[1:]] = 'False'
        if callback.data[1:] in dataParams.keys():
            dataParams[callback.data[1:]] = 'False'
    await bot.edit_message_text(
        chat_id=callback.message.chat.id,
        message_id=callback.message.message_id,
        text=f"JSON параметры:\n{getJson()}\n{getJson(1)}",
        reply_markup=keyboard,
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
            if nam in data.keys():
                if str(data[nam]).lower() in ['true', 'false']:
                    await message.answer(
                        f"Выбирай значение для "+nam, reply_markup=getYesNo(1, nam)
                    )
                else:
                    await message.answer("Напиши любое " + nam)
                    if nam in state_names:
                        await getattr(Form, nam).set()
                    else:
                        print("Ошибка какая-то")
            if nam in dataParams.keys():
                if str(dataParams[nam]).lower() in ['true', 'false']:
                    await message.answer(
                        f"Выбирай значение для "+nam, reply_markup=getYesNo(1, nam)
                    )
                else:
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
