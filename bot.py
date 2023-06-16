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

API_TOKEN = "TOKEN_HERE"

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

# -------- GLOBAL ----------
host = "127.0.0.1"
port = "7861"
api = webuiapi.WebUIApi(host=host, port=port)
# TODO --share used shared link. https://123456.gradio.live/docs does not work
local = "http://" + host + ":" + port
process = None
sd = "❌"

dataParams = {"img_thumb": "true", "img_tg": "true", "img_real": "true"}

@dp.message_handler(commands=["test"])
async def cmd_test(message: Union[types.Message, types.CallbackQuery]) -> None:
    print("cmd_test")
    options = {}
    options['outdir_txt2img_samples'] = '../../outputs/txt2img-images'
    api.set_options(options)

    res = await api.txt2img(prompt="cat in car",
                      negative_prompt="ugly, out of frame",
                      seed=-1,
                      styles=["anime"],
                      cfg_scale=7,
                      sampler_index='DDIM',
                      steps=15,
                      use_async=True,
                      save_images=True,
                      #                      enable_hr=True,
                      #                      hr_scale=2,
                      #                      hr_upscaler=webuiapi.HiResUpscaler.Latent,
                      #                      hr_second_pass_steps=20,
                      #                      hr_resize_x=1536,
                      #                      hr_resize_y=1024,
                      #                      denoising_strength=0.4,
                      )
    # images contains the returned images (PIL images)
    # result1.images

    # image is shorthand for images[0]
    # result1.image

    # info contains text info about the api call
    print(res.info)

    # info contains paramteres of the api call
    print(res.parameters)

    print('api.get_options()')
    print(api.get_options())
    print('api.util_get_current_model()')
    print(api.util_get_current_model())

@dp.message_handler(commands=["start"])
async def cmd_start(message: Union[types.Message, types.CallbackQuery]) -> None:
    print("cmd_start")
    subprocess.Popen(
        ["python", "../../launch.py", "--nowebui", "--xformers"]
    )

@dp.message_handler(commands=["test2"])
async def cmd_test2(message: Union[types.Message, types.CallbackQuery]) -> None:
    print("cmd_test2")
    print(getAttrtxt2img())

def getAttrtxt2img():
    # Получаем список аргументов функции api.txt2img и возвращаем JSON
    argspec = inspect.getfullargspec(api.txt2img)
    defaults = argspec.defaults or []
    args = argspec.args[1:]
    values = list(defaults) + [None] * (len(args) - len(defaults))
    params = {"/" + arg: str(value) if value is not None else "" for arg, value in zip(args, values)}
    result = json.dumps(params, indent=4)
    return result

# -------- BOT POLLING ----------
if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)

# -------- COPYRIGHT ----------
# Мишген
# join https://t.me/mishgenai
