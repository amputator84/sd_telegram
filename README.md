# sd_telegram
telegram bot on aiogram to generate images with automatic1111 fast api

```bash
cp sample.env .env
```

create bot from [BotFather](https://t.me/BotFather) and use token in API_TOKEN variable.
add vk and ok tokens to env file.

> заходим в https://oauth.vk.com/authorize?client_id=51626357&scope=photos&redirect_uri=http%3A%2F%2Foauth.vk.com%2Fblank.html&display=page&response_type=token \
> где 51626357 - номер вашего включенного приложения, созданного в https://vk.com/apps?act=manage, \
> photos - зона доступа. \
> После перехода и подтверждения выцепляем access_token из адресной строки \
> TODO auto requests \
> OK https://ok.ru/vitrine/myuploaded \
> Добавить приложение - https://ok.ru/app/setup \
> дбавить платформу - OAUTH \
> VALUABLE_ACCESS = Обязательно \
> PHOTO_CONTENT = Обязательно \ 
> Ссылка на страницу = https://apiok.ru/oauth_callback \
> Список разрешённых redirect_uri = https://apiok.ru/oauth_callback \
> сохранить, перезайти \
> Ищем ID приложения справа от "Основные настройки приложения" - ID 512002358821 \
> Открываем в браузере https://connect.ok.ru/oauth/authorize?client_id=512002358821&scope=PHOTO_CONTENT;VALUABLE_ACCESS&response_type=token&redirect_uri=https://apiok.ru/oauth_callback \
> С адресной строки копируем token в access_token ниже \
> application_key = Публичный ключ справа от "Основные настройки приложения" \
> Вечный access_token - Получить новый \
> application_secret_key = Session_secret_key \

run bot with python
```bash
python -m venv venv
source venv/bin/activate
pip install -m requirements.txt 
python bot.py
```
run bot in docker
```bash
docker build -t soaska/sd_bot .
docker run soaska/sd_bot
```

use ; in prompt as delimiter to be divided into several separate parts, like ```cat;dog;car```

Commands  
**start**  
_**SD**  
__❌ = off, ✅ = on Stable Diffusion  
_**opt**  
__**sttngs**  
___**change_param**  
change JSON parameters  
img_thumb/img_tg/img_real - little/original from tg/real size from doc  
___**reset_param**  
reset to default  
___**fast_param**  
reset to my default params  
comp, mobile, no hr, big, inc, sdxl, w↔h  
__**scrpts**  
___**get_lora**  
get list LORA`s from stable-diffusion-webui/models/Lora  
___**rnd_mdl**  
script for generating images for **all** models in **random** order, taking into account JSON settings  
___**rnd_smp**  
script for generating images for one models with all samplers  
___**inf**  
Endless script with random width, height, scale, model, sampler, steps, prompt(1. random-word-api.herokuapp.com/word?lang=en, 2.GPT2Tokenizer, 3. lexica.art/api/v1/search?q=2, 4 = 2 + 3)  
If _json_prompt_ = true then run random prompt from json  
__**mdl**  
change model from list  
__**smplr**  
change sampler from list  
__**hr**  
change hr_upscale from list  
__**prompt**  
___**get**  
get settings in string  
___**random_prompt**  
get random prompt from GPT2Tokenizer FredZhang7 distilgpt2    
___**lxc_prompt**  
get random prompt from lexica.art (+ your prompt begin)  
_**gen**  
generate images  
_**skip**  
skip one or all generations  
_**help**  
help  

If you send file, view 2 command:  
_**Lora**  
_**Model**  

Uploading files to folders:  
\models\Stable-diffusion
\models\Lora

You upload a file to Telegram, choose what it is and the file automatically goes to the folder.
A forwarded message with a file also works.

Please pay attention to the file size limit in Telegram API:  
[sending-files](https://core.telegram.org/bots/api#sending-files)  
[senddocument](https://core.telegram.org/bots/api#senddocument)

_**Chat History**  
We go into any Telegram chat with prompts (individual messages), click three dots in the upper right corner, upload only text messages in json format. We get the result.json file, which we throw into the bot and select "Chat History". We get a random prompt, which we can save in data  

If you chose _/img_real = true_, the document will be unloaded for you, and after it the social network **VK** and **OK** upload button.
Before that, you need to set up a token and enter the ID of the album where the photo will be sent.  

<img src="https://raw.githubusercontent.com/partyfind/sd_bot/master/trash/photo_2023-06-22_15-29-24.jpg" alt="drawing" width="400"/>
<img src="https://raw.githubusercontent.com/partyfind/sd_bot/master/trash/photo_2023-06-22_15-29-27.jpg" width="350"/>

**TODO**  
1. use share link (not yet possible, because the API is running in the background)  
2. show error in tg  
3. Get all files/pictures from the computer for today (preview and seeds)
4. Ability to send everything with one command with settings
5. Preloading photos when waiting for a long time so that you can skip
6. Progress in script (done with no HR)
7. Translator capable of translating up to 4000 characters and detecting the language  

**TNX**  
[AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui)  
[API](https://github.com/mix1009/sdwebuiapi)  
[aiogram](https://docs.aiogram.dev/en/latest/)  
And respect for Santa 🎅

Donations are **not needed**. Who wants to subscribe to [my generations](https://t.me/mishgenai)

Lifehack`s:  
If Lora dont work, see [this](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/7984?ref=blog.hinablue.me#issuecomment-1514312942) 
  