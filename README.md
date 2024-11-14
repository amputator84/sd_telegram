# sd_telegram
telegram bot on aiogram python 3.10.6 to generate images in automatic1111 locally

create bot from [BotFather](https://t.me/BotFather) and use token in [API_TOKEN](https://github.com/amputator84/sd_telegram/blob/master/bot.py#L32)

the bot is installed in automatic1111 via extensions or use _git clone_ into _C:\stable-diffusion-webui\extensions\sd_telegram_

```
pip install aiogram  
pip install webuiapi  
pip install translate  
pip install transformers  
pip install vk_api  
pip install ok_api  
cd C:\stable-diffusion-webui\extensions\sd_telegram\  
python bot.py
``````

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
__**sh**
change shedulers from list
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
  
© _Mishgen_