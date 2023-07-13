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
from pathlib import Path
import logging
import vk_api
from vk_api import VkUpload #https://github.com/python273/vk_api
from ok_api import OkApi, Upload # https://github.com/needkirem/ok_api

# Настройка логгера
logging.basicConfig(format="[%(asctime)s] %(levelname)s : %(name)s : %(message)s",
                    level=logging.DEBUG, datefmt="%d-%m-%y %H:%M:%S")

logging.getLogger('aiogram').setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

# from https://t.me/BotFather
API_BOT_TOKEN = "TOKEN_HERE"
#заходим в https://oauth.vk.com/authorize?client_id=123&scope=photos&redirect_uri=http%3A%2F%2Foauth.vk.com%2Fblank.html&display=page&response_type=token,
# где 123 - номер вашего включенного приложения, созданного в https://vk.com/apps?act=manage,
# photos - зона доступа.
# После перехода и подтверждения выцепляем access_token из адресной строки
# TODO auto requests
# OK https://ok.ru/vitrine/myuploaded
# Добавить приложение - https://ok.ru/app/setup
# дбавить платформу - OAUTH
# VALUABLE_ACCESS = Обязательно
# PHOTO_CONTENT = Обязательно
# Ссылка на страницу = https://apiok.ru/oauth_callback
# Список разрешённых redirect_uri = https://apiok.ru/oauth_callback
# сохранить, перезайти
# Ищем ID приложения справа от "Основные настройки приложения" - ID 123
# Открываем в браузере https://connect.ok.ru/oauth/authorize?client_id=123&scope=PHOTO_CONTENT;VALUABLE_ACCESS&response_type=token&redirect_uri=https://apiok.ru/oauth_callback
# С адресной строки копируем token в access_token ниже
# application_key = Публичный ключ справа от "Основные настройки приложения"
# Вечный access_token - Получить новый
# application_secret_key = Session_secret_key
VK_TOKEN = 'VK_TOKEN_HERE'
API_BOT_TOKEN = 'API_BOT_TOKEN_HERE'
VK_ALBUM_ID = 'VK_ALBUM_ID' # брать с адресной строки, когда открываешь ВК. Пример https://vk.com/album123_789
OK_ACCESS_TOKEN = 'OK_ACCESS_TOKEN_HERE'
OK_APPLICATION_KEY = 'OK_APPLICATION_KEY_HERE'
OK_APPLICATION_SECRET_KEY = 'OK_APPLICATION_SECRET_KEY_HERE'
OK_GROUP_ID = 'OK_GROUP_ID_HERE'

bot = Bot(token=API_BOT_TOKEN)
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
doc = ''
chatHistory = ''
chatHistoryPrompt = ''

data = getAttrtxt2img()
data['prompt'] = 'cat in space' # Ý
data['steps'] = 15
data['sampler_name'] = 'Euler a'
dataParams = {"img_thumb": "true",
              "img_tg": "true",
              "img_real": "true",
              "stop_sd": "true",
              "sd_model_checkpoint": "",
              "use_prompt": "true"}
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
        logging.info('start_process start_sd')
        process = subprocess.Popen(["python", "../../launch.py", "--nowebui", "--xformers", "--disable-nan-check"])
        sd = "✅"

async def stop_sd():
    global process, sd
    if process:
        logging.info('stop_process stop_sd')
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
def get_random_prompt(text = data['prompt'], max_length = 120):
    if str(dataParams['use_prompt']).lower() == 'true':
        text = data['prompt']
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model = GPT2LMHeadModel.from_pretrained("FredZhang7/distilgpt2-stable-diffusion-v2")
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    txt = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.8,
        top_k=8,
        max_length=max_length,
        num_return_sequences=1,
        repetition_penalty=1.2,
        penalty_alpha=0.6,
        no_repeat_ngram_size=0,
        early_stopping=True,
    )
    prompt = tokenizer.decode(txt[0], skip_special_tokens=True)
    return prompt

def rnd_prmt_lxc():
    txt = data['prompt']
    if str(dataParams['use_prompt']).lower() == 'false':
        txt = dataOrig['prompt']
    txt = random.choice(submit_get('https://lexica.art/api/v1/search?q='+txt, '').json()['images'])['prompt']
    return txt

# get settings. TODO - cut 4000 symbols
def get_prompt_settings(typeCode = 'HTML'):
    global sd
    prompt = data['prompt'].replace('<', '&lt;').replace('>', '&gt;')
    cfg_scale = data['cfg_scale']
    width = data['width']
    height = data['height']
    steps = data['steps']
    negative_prompt = data['negative_prompt'].replace('<', '&lt;').replace('>', '&gt;')
    sampler_name = data['sampler_name']
    if sd == '❌':
        sd_model_checkpoint = dataParams['sd_model_checkpoint']
    else:
        sd_model_checkpoint = api.get_options()['sd_model_checkpoint']
    if typeCode == 'HTML':
        txt = f"prompt = <code>{prompt}</code>\nsteps = {steps} \ncfg_scale = {cfg_scale} \nwidth = {width} \nheight = {height} \nsampler_name = {sampler_name} \nsd_model_checkpoint = {sd_model_checkpoint} \nnegative_prompt = <code>{negative_prompt}</code> "
    else:
        txt = f"prompt = {prompt}\n\nsteps = {steps} cfg_scale = {cfg_scale} width = {width} height = {height} sampler_name = {sampler_name} sd_model_checkpoint = {sd_model_checkpoint} \n\nnegative_prompt = {negative_prompt} "
    return txt

# Translate
def translateRuToEng(text):
    translator = Translator(from_lang="ru", to_lang="en")
    return translator.translate(text)

# Вывод прогресса в заменяемое сообщение
async def getProgress(msgTime):
    while True:
        # TODO aiogram.utils.exceptions.MessageToEditNotFound: Message to edit not found
        proc = round(api.get_progress()['progress']*100)
        points = '.' * (proc % 9)
        await bot.edit_message_text(
            chat_id=msgTime.chat.id,
            message_id=msgTime.message_id,
            text=str(proc)+'% ' + points
        )
        await asyncio.sleep(1)
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
        InlineKeyboardButton("inf",      callback_data="inf"),
    ]
    return (getKeyboard(keysArr, returnAll))


# Меню настроек
def getSet(returnAll = 1) -> InlineKeyboardMarkup:
    keysArr = [
        InlineKeyboardButton("change_param", callback_data="change_param"),
        InlineKeyboardButton("reset_param",  callback_data="reset_param"),
        InlineKeyboardButton("fast_param",   callback_data="fast_param"),
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
    global chatHistory

    if chatHistory != '':
        keysArr = [InlineKeyboardButton("get",    callback_data="get"),
                   InlineKeyboardButton("random", callback_data="random_prompt"),
                   InlineKeyboardButton("lxc",    callback_data="lxc_prompt"),
                   InlineKeyboardButton("json",   callback_data="next")]
    else:
        keysArr = [InlineKeyboardButton("get",    callback_data="get"),
                   InlineKeyboardButton("random", callback_data="random_prompt"),
                   InlineKeyboardButton("lxc",    callback_data="lxc_prompt")]
    return (getKeyboard(keysArr, returnAll))

# Меню промпта из JSON
def getPromptFromJson(returnAll = 1) -> InlineKeyboardMarkup:
    keysArr = [InlineKeyboardButton("Next prompt", callback_data="next"),
               InlineKeyboardButton("Save",        callback_data="save_prompt")]
    return (getKeyboard(keysArr, returnAll))

# Меню текста
def getTxt():
    return "/start /opt /gen /skip /stop /help"

def set_array(arrAll, itemArr, callback_data, useIn = 1):
    logging.info('set_array')
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
    dataPromptOld = data['prompt']
    await bot.send_message(
        chat_id=chatId,
        text='Цикл по '+str(len(elements)) + (' моделям' if typeScript == 'models' else ' семплерам') + ', ' + dataPromptOld
    )
    for i, number in enumerate(numbers):
        time.sleep(5)
        for itemTxt in data['prompt'].split(';'):
            if typeScript == 'models':
                api.util_wait_for_ready()
                dataParams['sd_model_checkpoint'] = elements[number]
                api.util_set_model(elements[number])
            else:
                options = {}
                options['sampler_name'] = elements[number]['name']
                api.set_options(options)
                data['sampler_name'] = elements[number]['name']  # Ý
            data["use_async"] = "False"
            data['prompt'] = itemTxt
            try:
                res = await api.txt2img(**data)
                await show_thumbs(chatId, res)
                await bot.send_message(
                    chat_id=chatId,
                    text=elements[number] if typeScript == 'models' else elements[number]['name']
                )
            except Exception as e:
                await bot.send_message(
                    chat_id=chatId,
                    text=e
                )
    data['prompt'] = dataPromptOld
    await bot.send_message(
        chat_id=chatId,
        text="Готово \n"+str(dataPromptOld) +
                    "\n cfg_scale = " + str(data['cfg_scale']) +
                    "\n width = " + str(data['width']) +
                    "\n height = " + str(data['height']) +
                    "\n steps = " + str(data['steps']) +
                    "\n negative = " + str(data['negative_prompt'])
        ,
        reply_markup=keyboard
    )
    if str(dataParams['stop_sd']).lower() == 'true':
        await stop_sd()

# show thumb/tg/real
async def show_thumbs(chat_id, res):
    if dataParams["img_thumb"] == "true" or dataParams["img_thumb"] == "True":
        await bot.send_media_group(
            chat_id=chat_id, media=pilToImages(res, "thumbs")
        )
    if dataParams["img_tg"] == "true" or dataParams["img_tg"] == "True":
        await bot.send_media_group(
            chat_id=chat_id, media=pilToImages(res, "tg")
        )
    if dataParams["img_real"] == "true" or dataParams["img_real"] == "True":
        mes_file = await bot.send_media_group(
            chat_id=chat_id, media=pilToImages(res, "real")
        )
        await bot.send_message(
                chat_id=chat_id,
                text="⬇ send to VK and OK ⬇",
                reply_markup=InlineKeyboardMarkup(
                inline_keyboard=[[InlineKeyboardButton(mes_file[0].document.file_id, callback_data='send_vk')]])
        )

# -------- COMMANDS ----------

# start или help
@dp.callback_query_handler(text="help")
@dp.message_handler(commands=["help"])
@dp.message_handler(commands=["start"])
async def cmd_start(message: Union[types.Message, types.CallbackQuery]) -> None:
    logging.info("cmd_start")
    txt = "Это бот для локального запуска SD\n" + getTxt()
    await getKeyboardUnion(txt, message, getStart())

# TODO optimize
# Запуск/Остановка SD. Завязываемся на глобальную иконку sd
@dp.message_handler(commands=["stop"])
@dp.callback_query_handler(text="sd")
async def inl_sd(message: Union[types.Message, types.CallbackQuery]) -> None:
    logging.info("inl_sd")
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
                    logging.info(r.status_code)
                except requests.exceptions.HTTPError as errh:
                    logging.info("Http Error:", errh)
                except requests.exceptions.ConnectionError as errc:
                    logging.info("Error Connecting:", errc)
                except requests.exceptions.Timeout as errt:
                    logging.info("Timeout Error:", errt)
                except requests.exceptions.RequestException as err:
                    logging.info("OOps: Something Else", err)
            sd = "✅"
            await message.message.edit_text(
                "SD запущена\n" + getTxt(), reply_markup=getStart()
            )

# save prompt
@dp.callback_query_handler(text="save_prompt")
async def inl_save_prompt(callback: types.CallbackQuery) -> None:
    logging.info("inl_save_prompt")
    global data, chatHistoryPrompt
    data['prompt'] = chatHistoryPrompt
    keyboard = InlineKeyboardMarkup(inline_keyboard=[getPromptFromJson(0), getStart(0)])
    await bot.edit_message_text(
        chat_id=callback.message.chat.id,
        message_id=callback.message.message_id,
        text='Промпт сохранён: ' + chatHistoryPrompt,
        reply_markup=keyboard
    )

# upload result.json from chat history
@dp.callback_query_handler(text="uplchat")
@dp.callback_query_handler(text="next")
async def inl_uplchat(callback: types.CallbackQuery) -> None:
    logging.info("inl_uplchat")
    # TODO cache chatHistory
    global chatHistory, chatHistoryPrompt
    file = await chatHistory.download()
    with open(file.name, 'r', encoding='utf-8') as f:
        data = json.load(f)
    t = random.choice(data['messages'])['text']
    if t == '':
        while True:
            t2 = random.choice(data['messages'])['text']
            if t2 != '':
                t = t2
                break

    chatHistoryPrompt = t#translateRuToEng(t)
    keyboard = InlineKeyboardMarkup(inline_keyboard=[getPromptFromJson(0), getStart(0)])
    await bot.edit_message_text(
        chat_id=callback.message.chat.id,
        message_id=callback.message.message_id,
        text=t.replace('<', '&lt;').replace('>', '&gt;'),#translateRuToEng(t).replace('<', '&lt;').replace('>', '&gt;'),
        reply_markup=keyboard,
        parse_mode = types.ParseMode.HTML
    )


# upload Lora/Model
@dp.callback_query_handler(text="uplora")
@dp.callback_query_handler(text="uplmodel")
async def inl_uplora(callback: types.CallbackQuery) -> None:
    logging.info("inl_uplora")
    global doc
    if callback.data == 'uplora':
        folder_path = Path('../../models/Lora')
    else:
        folder_path = Path('../../models/Stable-diffusion')
    file_id = doc.file_id
    file_name = doc.file_name
    destination_path = os.path.join(folder_path, file_name)
    file_path = folder_path / file_name
    if file_path.exists():
        await callback.message.reply(f"Файл '{file_name}' уже существует в {folder_path}")
    else:
        file_path = await bot.get_file(file_id)
        await file_path.download(destination_path)
        await callback.message.reply(f"Файл '{file_name}' загружен в {folder_path}")

# Вызов reset_param, сброс JSON
@dp.message_handler(commands=["reset_param"])
@dp.callback_query_handler(text="reset_param")
async def inl_reset_param(message: Union[types.Message, types.CallbackQuery]) -> None:
    logging.info("inl_reset_param")
    global data
    global dataParams
    global dataOld
    global dataOldParams
    data = dataOld
    dataParams = dataOldParams
    keyboard = InlineKeyboardMarkup(inline_keyboard=[getSet(0), getOpt(0), getStart(0)])
    txt = f"JSON сброшен\n{getJson()}\n{getJson(1)}"
    await getKeyboardUnion(txt, message, keyboard, '')

# Вызов fast_param, быстрые настройки
@dp.message_handler(commands=["fast_param"])
@dp.callback_query_handler(text="fast_param")
async def inl_fast_param(message: Union[types.Message, types.CallbackQuery]) -> None:
    logging.info("inl_fast_param")
    global data
    global dataParams
    data['steps'] = 35
    data['sampler_name'] = 'Euler a'
    data['enable_hr'] = 'True'
    data['denoising_strength'] = '0.5'
    data['hr_upscaler'] = 'ESRGAN_4x'
    data['hr_second_pass_steps'] = '10'
    data['cfg_scale'] = '6'
    data['width'] = '512'
    data['height'] = '768'
    data['restore_faces'] = 'true'
    data['do_not_save_grid'] = 'true'
    data['negative_prompt'] = 'easynegative, bad-hands-5, bad-picture-chill-75v, bad-artist, bad_prompt_version2, rmadanegative4_sd15-neg, bad-image-v2-39000, illustration, painting, cartoons, sketch, (worst quality:2), (low quality:2), (normal quality:2), lowres, bad anatomy, bad hands, ((monochrome)), ((grayscale)), collapsed eyeshadow, multiple eyeblows, vaginas in breasts, (cropped), oversaturated, extra limb, missing limbs, deformed hands, long neck, long body, imperfect, (bad hands), signature, watermark, username, artist name, conjoined fingers, deformed fingers, ugly eyes, imperfect eyes, skewed eyes, unnatural face, unnatural body, error, asian, obese, tatoo, stacked torsos, totem pole, watermark, black and white, close up, cartoon, 3d, denim, (disfigured), (deformed), (poorly drawn), (extra limbs), blurry, boring, sketch, lackluster, signature, letters'
    data['save_images'] = 'true'
    dataParams = {"img_thumb": "false",
                  "img_tg": "true",
                  "img_real": "true",
                  "stop_sd": "true",
                  "use_prompt": "true"}
    keyboard = InlineKeyboardMarkup(inline_keyboard=[getSet(0), getOpt(0), getStart(0)])
    txt = f"JSON сброшен\n{getJson()}\n{getJson(1)}"
    await getKeyboardUnion(txt, message, keyboard, '')

# Обработчик команды /skip
@dp.message_handler(commands=["skip"])
@dp.callback_query_handler(text="skip")
async def inl_skip(message: Union[types.Message, types.CallbackQuery]) -> None:
    logging.info('inl_skip')
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
    keyboard = InlineKeyboardMarkup(inline_keyboard=[getSet(0), getOpt(0), getStart(0)])
    global sd
    dataPromptOld = data['prompt']
    if sd == '✅':
        for itemTxt in data['prompt'].split(';'):
            try:
                msgTime = await bot.send_message(
                    chat_id=chatId,
                    text='Начали'
                )
                # Включаем асинхрон, чтоб заработал await api.txt2img
                data["use_async"] = "True"
                data["prompt"] = itemTxt # только для **data
                asyncio.create_task(getProgress(msgTime))
                # TODO try catch if wrong data
                res = await api.txt2img(**data)
                # show_thumbs dont work because use_async
                if dataParams["img_thumb"] == "true" or dataParams["img_thumb"] == "True":
                    await bot.send_media_group(
                        chat_id=chatId, media=pilToImages(res, "thumbs")
                    )
                if dataParams["img_tg"] == "true" or dataParams["img_tg"] == "True":
                    await bot.send_media_group(
                        chat_id=chatId, media=pilToImages(res, "tg")
                    )
                if dataParams["img_real"] == "true" or dataParams["img_real"] == "True":
                    mes_file = await bot.send_media_group(
                        chat_id=chatId,
                        media=pilToImages(res, "real")
                    )
                    # send button load in VK
                    # TODO long message
                    await bot.send_message(
                        chat_id=chatId,
                        text="⬇ send to VK and OK ⬇",
                        reply_markup=InlineKeyboardMarkup(
                            inline_keyboard=[[InlineKeyboardButton(mes_file[0].document.file_id, callback_data='send_vk')]])
                    )
                await bot.send_message(
                    chat_id=chatId,
                    text=data["prompt"] + "\n" + str(res.info["all_seeds"])
                )
                # Удаляем сообщение с прогрессом
                await bot.delete_message(chat_id=msgTime.chat.id, message_id=msgTime.message_id)
            except Exception as e:
                logging.error(f"gen error: {e}")
                await bot.send_message(
                    chat_id=chatId,
                    text=e,
                    reply_markup=keyboard,
                    parse_mode="Markdown",
                )
        await bot.send_message(
            chat_id=chatId,
            text=f"`{dataPromptOld}`",
            reply_markup=keyboard,
            parse_mode="Markdown",
        )
    else:
        keyboard = InlineKeyboardMarkup(inline_keyboard=[getSet(0), getOpt(0), getStart(0)])
        await getKeyboardUnion("Turn on SD"+sd, message, keyboard)

# upload in VK
# TODO actual prompt
@dp.callback_query_handler(text="send_vk")
async def send_vk(callback: types.CallbackQuery) -> None:
    try:
        # Export VK
        global VK_TOKEN, VK_ALBUM_ID, OK_ACCESS_TOKEN, OK_APPLICATION_KEY, OK_APPLICATION_SECRET_KEY, OK_GROUP_ID
        file_id = callback.message.reply_markup.inline_keyboard[0][0].text #TODO
        file_obj = await bot.get_file(file_id)
        vk_session = vk_api.VkApi(token=VK_TOKEN)
        vk_upload = VkUpload(vk_session)
        file_url = f'https://api.telegram.org/file/bot{API_BOT_TOKEN}/{file_obj.file_path}'

        #TODO optimize
        with open('temp.png', 'wb') as file:
            file.write(requests.get(file_url).content)
        vk_upload.photo(
            photos='temp.png',
            album_id=VK_ALBUM_ID,
            caption=data['prompt'] #TODO actual from ID message
        )
        # Export OK
        ok = OkApi(
            access_token=OK_ACCESS_TOKEN,
            application_key=OK_APPLICATION_KEY,
            application_secret_key=OK_APPLICATION_SECRET_KEY)
        group_id = OK_GROUP_ID
        upload = Upload(ok)
        upload_response = upload.photo(photos=['temp.png'], album=group_id)
        for photo_id in upload_response['photos']:
            token = upload_response['photos'][photo_id]['token']
            response = ok.photosV2.commit(photo_id=photo_id, token=token, comment=data['prompt'])
            print(response.text)

        # clear garbage
        os.remove('temp.png')
        await callback.message.edit_text(
            'Фотка в VK и OK загружена'
        )
    except Exception as e:
        await bot.send_message(
            chat_id=callback.message.chat.id,
            text=e,
            parse_mode=types.ParseMode.HTML
        )

# Получить меню действий с промптами
@dp.callback_query_handler(text="prompt")
async def cmd_prompt(message: Union[types.Message, types.CallbackQuery]) -> None:
    logging.info("cmd_prompt")
    keyboard = InlineKeyboardMarkup(inline_keyboard=[getPrompt(0), getOpt(0), getStart(0)])
    await getKeyboardUnion("Опции", message, keyboard)

# Получить опции
@dp.message_handler(commands=["opt"])
@dp.callback_query_handler(text="opt")
async def cmd_opt(message: Union[types.Message, types.CallbackQuery]) -> None:
    logging.info("cmd_opt")
    keyboard = InlineKeyboardMarkup(inline_keyboard=[getOpt(0), getStart(0)])
    await getKeyboardUnion("Опции", message, keyboard)

# Вызов sttngs
@dp.message_handler(commands=["sttngs"])
@dp.callback_query_handler(text="sttngs")
async def inl_sttngs(message: Union[types.Message, types.CallbackQuery]) -> None:
    logging.info("inl_sttngs")
    keyboard = InlineKeyboardMarkup(inline_keyboard=[getSet(0), getOpt(0), getStart(0)])
    await getKeyboardUnion("Настройки", message, keyboard)

# Вызов script
@dp.message_handler(commands=["scrpts"])
@dp.callback_query_handler(text="scrpts")
async def inl_scrpts(message: Union[types.Message, types.CallbackQuery]) -> None:
    logging.info("inl_scrpts")
    keyboard = InlineKeyboardMarkup(inline_keyboard=[getScripts(0), getOpt(0), getStart(0)])
    await getKeyboardUnion("Скрипты", message, keyboard)

# Вызов get_models
@dp.message_handler(commands=["mdl"])
@dp.callback_query_handler(text="mdl")
async def inl_mdl(message: Union[types.Message, types.CallbackQuery]) -> None:
    logging.info("inl_mdl")
    global sd
    if sd == '✅':
        menu = get_models()
        menu.append(getOpt(0))
        menu.append(getStart(0))
        await getKeyboardUnion("Модельки", message, InlineKeyboardMarkup(inline_keyboard=menu))
    else:
        keyboard = InlineKeyboardMarkup(inline_keyboard=[getOpt(0), getStart(0)])
        await getKeyboardUnion("Turn on SD"+sd, message, keyboard)

# Вызов get_samplers
@dp.message_handler(commands=["smplr"])
@dp.message_handler(commands=["sampler_name"])
@dp.callback_query_handler(text="smplr")
async def inl_smplr(message: Union[types.Message, types.CallbackQuery]) -> None:
    logging.info("inl_smplr")
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
    logging.info("inl_hr")
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
    logging.info("inl_change_param")
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
    logging.info('inl_rnd_mdl')
    await rnd_script(message, 'models')

# script random gen from models
@dp.message_handler(commands=["rnd_smp"])
@dp.callback_query_handler(text='rnd_smp')
async def inl_rnd_smp(message: Union[types.Message, types.CallbackQuery]) -> None:
    logging.info('inl_rnd_smp')
    await rnd_script(message, 'samplers')

# inf function
async def inf_func(chatId):
    logging.info('inf_func')
    # SCALE
    data['cfg_scale'] = round(random.uniform(4.7, 15), 1)
    # STEPS
    data['steps'] = round(random.uniform(10, 80))
    # WIDTH and HEIGHT
    width = random.randrange(512, 1601, 64)
    height = random.randrange(512, 1601, 64)

    while width * height > 1000000:
        width = random.randrange(512, 1601, 64)
        height = random.randrange(512, 1601, 64)
    data['width'] = width
    data['height'] = height
    # MODEL
    models = api.util_get_model_names()
    num_mdl = random.randint(0, len(models) - 1)
    api.util_wait_for_ready()
    dataParams['sd_model_checkpoint'] = models[num_mdl]
    api.util_set_model(models[num_mdl])
    # SAMPLER
    samplers = api.get_samplers()
    num_smp = random.randint(0, len(samplers) - 1)
    options = {}
    options['sampler_name'] = samplers[num_smp]['name']
    api.set_options(options)
    data['sampler_name'] = samplers[num_smp]['name']  # Ý

    data["use_async"] = False
    # GEN
    try:
        res = api.txt2img(**data)
        await show_thumbs(chatId, res)
        await bot.send_message(
            chat_id=chatId,
            text=get_prompt_settings(),
            parse_mode=types.ParseMode.HTML
        )
    except Exception as e:
        await bot.send_message(
            chat_id=chatId,
            text=e,
            parse_mode=types.ParseMode.HTML
        )

# script random infinity gen from https://random-word-api.herokuapp.com/word?lang=en
@dp.message_handler(commands=["inf"])
@dp.callback_query_handler(text='inf')
async def inl_rnd_inf(message: Union[types.Message, types.CallbackQuery]) -> None:
    logging.info('inl_rnd_inf')
    global dataParams # ?
    if hasattr(message, "content_type"):
        chatId = message.chat.id
    else:
        chatId = message.message.chat.id
    if str(dataParams['use_prompt']).lower() == 'true':
        await bot.send_message(
            chat_id=chatId,
            text='use_prompt включен, будет использоваться промпт ' + data['prompt']
        )
    while True:
        # PROMPT
        if str(dataParams['use_prompt']).lower() == 'false':
            t = requests.get('https://random-word-api.herokuapp.com/word?lang=en').text
            text = json.loads(t)[0] # from JSON
            prompt = get_random_prompt(text, 20)
            prompt_lxc = random.choice(submit_get('https://lexica.art/api/v1/search?q=' + prompt, '').json()['images'])['prompt']
            data['prompt'] = prompt + ', ' + prompt_lxc
            await inf_func(chatId)
        else:
            dataPromptOld = data['prompt']
            for itemTxt in data['prompt'].split(';'):
                data['prompt'] = itemTxt
                await inf_func(chatId)
            data['prompt'] = dataPromptOld

# Получить LORA
@dp.message_handler(commands=["get_lora"])
@dp.callback_query_handler(text="get_lora")
async def getLora(message: Union[types.Message, types.CallbackQuery]) -> None:
    logging.info("getLora")
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
    logging.info("get_lxc_prompt")
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
    await getKeyboardUnion(get_prompt_settings(), message, keyboard, types.ParseMode.HTML)

# тыкнули на модельку
@dp.callback_query_handler(text_startswith="models")
async def inl_models(callback: types.CallbackQuery) -> None:
    logging.info('inl_models')
    menu = get_models()
    menu.append(getOpt(0))
    menu.append(getStart(0))
    mdl = callback.data.split("|")[1]
    dataParams['sd_model_checkpoint'] = mdl
    api.util_set_model(mdl)
    api.util_wait_for_ready()
    await getKeyboardUnion('Теперь модель = ' + str(mdl), callback, InlineKeyboardMarkup(inline_keyboard=menu), '')

# тыкнули на сэмплер
@dp.callback_query_handler(text_startswith="samplers")
async def inl_samplers(callback: types.CallbackQuery) -> None:
    logging.info('inl_samplers')
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
    logging.info('inl_hrs')
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
    logging.info('inl_yes_no')
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
    #await bot.delete_message(chat_id=callback.message.chat.id, message_id=callback.message.message_id)
    await bot.edit_message_text(
        chat_id=callback.message.chat.id,
        message_id=callback.message.message_id,
        text=f"JSON параметры:\n{getJson()}\n{getJson(1)}",
        reply_markup=keyboard,
    )

# Ввели любой текст
@dp.message_handler(lambda message: True)
async def change_json(message: types.Message):
    logging.info("change_json")
    keyboard = InlineKeyboardMarkup(inline_keyboard=[getSet(0), getOpt(0), getStart(0)])
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
                    await bot.delete_message(chat_id=message.chat.id, message_id=message.message_id)
                else:
                    await message.answer("Напиши любое " + nam + ', сейчас оно = ' + str(data[nam]))
                    if nam in state_names:
                        await getattr(Form, nam).set()
                    else:
                        logging.info("Ошибка какая-то")
            if nam in dataParams.keys():
                if str(dataParams[nam]).lower() in ['true', 'false']:
                    await message.answer(
                        f"Выбирай значение для "+nam, reply_markup=getYesNo(1, nam)
                    )
                    await bot.delete_message(chat_id=message.chat.id, message_id=message.message_id)
                else:
                    await bot.delete_message(chat_id=message.chat.id, message_id=message.message_id)
                    await message.answer("Напиши любое " + nam + ', сейчас оно = ' + str(data[nam]))
                    if nam in state_names:
                        await getattr(Form, nam).set()
                    else:
                        logging.info("Ошибка какая-то")
        else:
            # /txt 321 пишем 321 в data['txt']
            # Ý 0,1 into 0.1
            newArgs = args
            if str(args[1:2]) == ",":
                newArgs = args.replace(',','.')
            data[nam] = newArgs
            # TODO answer поменять на edit_text
            await bot.delete_message(chat_id=message.chat.id, message_id=message.message_id)
            await message.answer(
                f"JSON параметры:\n{getJson()}\n{getJson(1)}", reply_markup=keyboard
            )
    else:
        data["prompt"] = message.text#translateRuToEng(message.text)
        await bot.delete_message(chat_id=message.chat.id, message_id=message.message_id)
        await message.answer(
            f"Записали промпт. JSON параметры:\n{getJson()}\n{getJson(1)}",
            reply_markup=keyboard,
        )

# Ввели ответ на change_json
@dp.message_handler(state=Form)
async def answer_handler(message: types.Message, state: FSMContext):
    logging.info('answer_handler')
    keyboard = InlineKeyboardMarkup(inline_keyboard=[getSet(0), getOpt(0), getStart(0)])
    current_state = await state.get_state()  # Form:команда
    txt = message.text
    for key, val in dataParams.items():
        if current_state == "Form:" + key:
            dataParams[key] = txt
            break
    for key, val in data.items():
        if current_state == "Form:" + key:
            newTxt = txt
            if str(txt[1:2]) == ",":
                newTxt = txt.replace(',','.')
            data[key] = newTxt
            break
    await state.reset_state()
    await bot.delete_message(chat_id=message.chat.id, message_id=message.message_id)
    await message.answer(
        f"JSON параметры:\n{getJson()}\n{getJson(1)}", reply_markup=keyboard
    )

@dp.message_handler(content_types=['document'])
async def handle_file(message: types.Message):
    logging.info('handle_file')

    #
    if message.document.file_name == 'result.json':
        global chatHistory
        if chatHistory == '':
            chatHistory = message.document
    else:
        global doc
        doc = message.document
    keysArr = [InlineKeyboardButton("Lora",         callback_data="uplora"),
               InlineKeyboardButton("Model",        callback_data="uplmodel"),
               InlineKeyboardButton("Chat History", callback_data="uplchat"),]
    await bot.send_message(
        chat_id=message.chat.id,
        text="Что грузим?",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[keysArr])
    )

# -------- BOT POLLING ----------
if __name__ == "__main__":
    logging.info("starting bot")
    executor.start_polling(dp, skip_updates=True)

# -------- COPYRIGHT ----------
# Мишген
# join https://t.me/mishgenai
