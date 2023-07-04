# sd_telegram
telegram bot on aiogram python 3.10.6 to generate images in automatic1111 locally

create bot from [BotFather](https://t.me/BotFather) and use token in [API_TOKEN](https://github.com/amputator84/sd_telegram/blob/master/bot.py#L32)

the bot is installed in automatic1111 via extensions or use _git clone_ into _C:\stable-diffusion-webui\extensions\sd_telegram_

```
pip install aiogram  
pip install webuiapi  
pip install transformers  
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
__**scrpts**  
___**get_lora**  
get list LORA`s from stable-diffusion-webui/models/Lora  
___**rnd_mdl**  
script for generating images for **all** models in **random** order, taking into account JSON settings  
___**rnd_smp**  
script for generating images for one models with all samplers  
___**inf**  
Endless script with random width, height, scale, model, sampler, steps, prompt(1. random-word-api.herokuapp.com/word?lang=en, 2.GPT2Tokenizer, 3. lexica.art/api/v1/search?q=2, 4 = 2 + 3)  
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

<img src="https://raw.githubusercontent.com/partyfind/sd_bot/master/trash/photo_2023-06-22_15-29-24.jpg" alt="drawing" width="400"/>
<img src="https://raw.githubusercontent.com/partyfind/sd_bot/master/trash/photo_2023-06-22_15-29-27.jpg" width="350"/>

**TODO**  
1. use share link (not yet possible, because the API is running in the background)  
2. uploading Lora from TG
3. uploading models from TG (~4gb)
4. Logging
5. show error in tg
6. translate
7. Get all files/pictures from the computer for today (preview and seeds)
8. Several prompts in one via ;
9. Ability to send everything with one command with settings
10. Preloading photos when waiting for a long time so that you can skip
11. Uploading random.json from export TG channel
12. Progress in script

**TNX**  
[AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui)  
[API](https://github.com/mix1009/sdwebuiapi)  
[aiogram](https://docs.aiogram.dev/en/latest/)  
And respect for Santa 🎅

Donations are **not needed**. Who wants to subscribe to [my generations](https://t.me/mishgenai)
  
© _Mishgen_