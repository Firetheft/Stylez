import os
import html
import datetime
import urllib.parse
import gradio as gr
from PIL import Image
import shutil
import json
import csv
import re
from modules import scripts, shared,script_callbacks
from scripts import promptgen as PG
from scripts import superprompt as SP
from scripts import florence as FL
from modules import (
    generation_parameters_copypaste as parameters_copypaste,
)
try:
    from modules.call_queue import wrap_gradio_gpu_call
except ImportError:
    from webui import wrap_gradio_gpu_call

extension_path = scripts.basedir()
refresh_symbol = '\U0001f504'  # 🔄
close_symbol = '\U0000274C'  # ❌
save_symbol = '\U0001F4BE' #💾
delete_style = '\U0001F5D1' #🗑️
clear_symbol = '\U0001F9F9' #🧹

card_size_value = 0
card_size_min = 0
card_size_max = 0
favourites = []
hideoldstyles = False
config_json = os.path.join(extension_path,"scripts" ,"config.json")

def save_card_def(value):
    global card_size_value
    save_settings("card_size",value)
    card_size_value = value
    
if not os.path.exists(config_json):
    default_config = {
        "card_size": 108,
        "card_size_min": 50,
        "card_size_max": 200,
        "autoconvert": True,
        "hide_old_styles": False,
        "favourites": []
    }
    
    with open(config_json, 'w') as config_file:
        json.dump(default_config, config_file, indent=4)

# Load values from the JSON file
with open(config_json, "r") as json_file:
    data = json.load(json_file)
    card_size_value = data["card_size"]
    card_size_min = data["card_size_min"]
    card_size_max = data["card_size_max"]
    autoconvert = data["autoconvert"]
    favourites = data["favourites"]
    hide_old_styles = data["hide_old_styles"]

def reload_favourites():
    with open(config_json, "r") as json_file:
        data = json.load(json_file)
        global favourites
        favourites = data["favourites"]

def save_settings(setting,value):
    with open(config_json, "r") as json_file:
        data = json.load(json_file)
    data[setting] = value
    with open(config_json, "w") as json_file:
        json.dump(data, json_file, indent=4)

def img_to_thumbnail(img):
    return gr.update(value=img)

character_translation_table = str.maketrans('"*/:<>?\\|\t\n\v\f\r', '＂＊／：＜＞？＼￨     ')
leading_space_or_dot_pattern = re.compile(r'^[\s.]')


def replace_illegal_filename_characters(input_filename: str):
    r"""
    Replace illegal characters with full-width variant
    if leading space or dot then add underscore prefix
    if input is blank then return underscore
    Table
    "           ->  uff02 full-width quotation mark         ＂
    *           ->  uff0a full-width asterisk               ＊
    /           ->  uff0f full-width solidus                ／
    :           ->  uff1a full-width colon                  ：
    <           ->  uff1c full-width less-than sign         ＜
    >           ->  uff1e full-width greater-than sign      ＞
    ?           ->  uff1f full-width question mark          ？
    \           ->  uff3c full-width reverse solidus        ＼
    |           ->  uffe8 half-width forms light vertical   ￨
    \t\n\v\f\r  ->  u0020 space
    """
    if input_filename:
        output_filename = input_filename.translate(character_translation_table)
        # if  leading character is a space or a dot, add _ in front
        return '_' + output_filename if re.match(leading_space_or_dot_pattern, output_filename) else output_filename
    return '_'  # if input is None or blank


def create_json_objects_from_csv(csv_file):
    json_objects = []
    with open(csv_file, 'r', newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Retrieve values from CSV with special character handling
            name = row.get('name', None)
            prompt = row.get('prompt', None)
            negative_prompt = row.get('negative_prompt', None)
            if name is None or prompt is None or negative_prompt is None:
                print("Warning: Skipping row with missing values.")
                continue
            safe_name = replace_illegal_filename_characters(name)
            json_data = {
                "name": safe_name,
                "description": "converted from csv",
                "preview": f"{safe_name}.jpg",
                "prompt": prompt,
                "negative": negative_prompt,
            }
            json_objects.append(json_data)
    return json_objects

def save_json_objects(json_objects):
    if not json_objects:
        print("Warning: No JSON objects to save.")
        return

    styles_dir = os.path.join(extension_path, "styles")
    csv_conversion_dir = os.path.join(styles_dir, "CSVConversion")
    os.makedirs(csv_conversion_dir, exist_ok=True)

    nopreview_image_path = os.path.join(extension_path, "nopreview.jpg")
    for json_obj in json_objects:
        try:
            json_file_path = os.path.join(csv_conversion_dir, f"{json_obj['name']}.json")
            with open(json_file_path, 'w') as jsonfile:
                json.dump(json_obj, jsonfile, indent=4)
            image_path = os.path.join(csv_conversion_dir, f"{json_obj['name']}.jpg")
            shutil.copy(nopreview_image_path, image_path)
        except Exception as e:
            print(f'{e}\nStylez Failed to convert {json_obj.get("name", str(json_obj))}')

        
if autoconvert:
    styles_files = shared.cmd_opts.styles_file if isinstance(shared.cmd_opts.styles_file, list) else [shared.cmd_opts.styles_file]
    for styles_file_path in styles_files:
        if os.path.exists(styles_file_path):
            json_objects = create_json_objects_from_csv(styles_file_path)
            save_json_objects(json_objects)
        else:
            print(f"File does not exist: {styles_file_path}")  # Optional: log or handle the case where a file doesn't exist

    save_settings("autoconvert", False)


def generate_html_code():
    reload_favourites()
    style = None
    style_html = ""
    categories_list = ["All","Favourites"]
    save_categories_list =[]
    styles_dir = os.path.join(extension_path, "styles")
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime('%H:%M:%S.%f')
    formatted_time = formatted_time.replace(":", "")
    formatted_time = formatted_time.replace(".", "")
    try:
        for root, dirs, _ in os.walk(styles_dir):
            for directory in dirs:
                subfolder_name = os.path.basename(os.path.join(root, directory))
                if subfolder_name.lower() not in categories_list:
                    categories_list.append(subfolder_name)
                if subfolder_name.lower() not in save_categories_list:
                    save_categories_list.append(subfolder_name)    
        for root, _, files in os.walk(styles_dir):
            for filename in files:
                if filename.endswith(".json"):
                    json_file_path = os.path.join(root, filename)
                    subfolder_name = os.path.basename(root)
                    with open(json_file_path, "r", encoding="utf-8") as f:
                        try:
                            style = json.load(f)
                            title = style.get("name", "")
                            preview_image = style.get("preview", "")
                            description = style.get("description", "")
                            img = os.path.join(os.path.dirname(json_file_path), preview_image)
                            img = os.path.abspath(img)
                            prompt = style.get("prompt", "")
                            prompt = html.escape(json.dumps(prompt))
                            prompt_negative = style.get("negative", "")
                            prompt_negative =html.escape(json.dumps(prompt_negative))
                            imghack = img.replace("\\", "/")
                            json_file_path = json_file_path.replace("\\", "/")
                            encoded_filename = urllib.parse.quote(filename, safe="")
                            titlelower = str(title).lower()
                            color = ""
                            stylefavname =subfolder_name + "/" + filename
                            if (stylefavname in favourites):
                                color = "#EBD617"
                            else:
                                color = "#ffffff"
                            style_html += f"""
                            <div class="style_card" data-category='{subfolder_name}' data-title='{titlelower}' style="min-height:{card_size_value}px;max-height:{card_size_value}px;min-width:{card_size_value}px;max-width:{card_size_value}px;">
                                <div class="style_card_checkbox" onclick="toggleCardSelection(event, '{subfolder_name}','{encoded_filename}')">◉</div>  <!-- 这里添加勾选框 -->
                                <img class="styles_thumbnail" src="{"file=" + img +"?timestamp"+ formatted_time}" alt="{title} Preview">
                                <div class="EditStyleJson">
                                    <button onclick="editStyle(`{title}`,`{imghack}`,`{description}`,`{prompt}`,`{prompt_negative}`,`{subfolder_name}`,`{encoded_filename}`,`Stylez`)">🖉</button>
                                </div>
                                <div class="favouriteStyleJson">
                                    <button class="favouriteStyleBtn" style="color:{color};" onclick="addFavourite('{subfolder_name}','{encoded_filename}', this)">★</button>
                                </div>
                                    <div onclick="applyStyle(`{prompt}`,`{prompt_negative}`,`Stylez`)" onmouseenter="event.stopPropagation(); hoverPreviewStyle(`{prompt}`,`{prompt_negative}`,`Stylez`)" onmouseleave="hoverPreviewStyleOut()" class="styles_overlay"></div>
                                    <div class="styles_title">{title}</div>
                                    <p class="styles_description">{description}</p>
                                </img>
                            </div>
                            """
                        except json.JSONDecodeError:
                            print(f"Error parsing JSON in file: {filename}")
                        except KeyError as e:
                            print(f"KeyError: {e} in file: {filename}")
    except FileNotFoundError:
        print("Directory '/models/styles' not found.")
    return style_html, categories_list, save_categories_list

def refresh_styles(cat):
    if cat is None or len(cat) == 0 or cat  == "[]" :
        cat = None
    newhtml = generate_html_code()
    newhtml_sendback = newhtml[0]
    newcat_sendback = newhtml[1]
    newfilecat_sendback = newhtml[2]
    return newhtml_sendback,gr.update(choices=newcat_sendback),gr.update(value="All"),gr.update(choices=newfilecat_sendback)

def save_style(title, img, description, prompt, prompt_negative, filename, save_folder):
    print(f"""Saved: '{save_folder}/{filename}'""")
    if save_folder and filename:
        if img is None or img == "":
            img = Image.open(os.path.join(extension_path, "nopreview.jpg")) 
        img = img.resize((200, 200))
        save_folder_path = os.path.join(extension_path, "styles", save_folder)
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
        json_data = {
            "name": title,
            "description": description,
            "preview": filename + ".jpg",
            "prompt": prompt,
            "negative": prompt_negative,
        }
        json_file_path = os.path.join(save_folder_path, filename + ".json")
        with open(json_file_path, "w") as json_file:
            json.dump(json_data, json_file, indent=4)
        img_path = os.path.join(save_folder_path, filename + ".jpg")
        img.save(img_path)
        msg = f"""File Saved to '{save_folder}'"""
        info(msg)
    else:
        msg = """Please provide a valid save folder and Filename"""
        warning(msg)
    return filename_check(save_folder,filename)

def info(message):
    gr.Info(message)

def warning(message):
    gr.Warning(message)
    
def tempfolderbox(dropdown):
    return gr.update(value=dropdown)

def filename_check(folder,filename):
    if filename is None or len(filename) == 0 :
        warning = """<p id="style_filename_check" style="color:orange;">请输入文件名！！！</p>"""
    else:
        save_folder_path = os.path.join(extension_path, "styles", folder)
        json_file_path = os.path.join(save_folder_path, filename + ".json")
        if os.path.exists(json_file_path):
            warning = f"""<p id="style_filename_check" style="color:green;">文件已添加到 '{folder}'</p>"""
        else:
            warning = """<p id="style_filename_check" style="color:green;">文件名有效！！！</p>"""
    return gr.update(value=warning)

def clear_style():
    previewimage = os.path.join(extension_path, "nopreview.jpg")
    return gr.update(value=None),gr.update(value=previewimage),gr.update(value=None),gr.update(value=None),gr.update(value=None),gr.update(value=None),gr.update(value=None)

def deletestyle(folder, filename):
    base_path = os.path.join(extension_path, "styles", folder)
    json_file_path = os.path.join(base_path, filename + ".json")
    jpg_file_path = os.path.join(base_path, filename + ".jpg")

    if os.path.exists(json_file_path):
        os.remove(json_file_path)
        warning(f"""Stlye "{filename}" deleted!! """)
        if os.path.exists(jpg_file_path):
            os.remove(jpg_file_path)
        else:
            warning(f"Error: {jpg_file_path} not found.")
    else:
        warning(f"Error: {json_file_path} not found.")

def addToFavourite(style):
 global favourites
 if (style not in favourites):
     favourites.append(style)
     save_settings("favourites",favourites)
     info("style added to favourites")

def removeFavourite(style):
 global favourites
 if (style in favourites):
     favourites.remove(style)
     save_settings("favourites",favourites)
     info("style removed from favourites")

def oldstyles(value):
    with open(config_json, "r") as json_file:
        data = json.load(json_file)
        if (data["hide_old_styles"] == True):
            save_settings("hide_old_styles",False)
        else:
            save_settings("hide_old_styles",True)

def generate_style(prompt,temperature,top_k,max_length,repetition_penalty,usecomma):
    result = PG.generate(prompt,temperature,top_k,max_length,repetition_penalty,usecomma)
    return gr.update(value=result)

def call_generate_super_prompt(prompt,superprompt_max_length,superprompt_seed):
    return SP.generate_super_prompt(prompt, max_new_tokens=superprompt_max_length, seed=superprompt_seed)

def create_ar_button(label, width, height, button_class="ar-button"):
    return gr.Button(label, elem_classes=button_class).click(fn=None, _js=f'sendToARbox({width}, {height})')

def update_prompt_types(model_name):
    """根据选择的模型更新提示类型下拉菜单"""
    if model_name in ["MiaoshouAI/Florence-2-base-PromptGen-v1.5", "MiaoshouAI/Florence-2-large-PromptGen-v1.5"]:
        choices = [
            "<GENERATE_TAGS>",
            "<CAPTION>",
            "<DETAILED_CAPTION>",
            "<MORE_DETAILED_CAPTION>",
            "<MIXED_CAPTION>",
        ]
    else:
        choices = [
            "<CAPTION>",
            "<DETAILED_CAPTION>",
            "<MORE_DETAILED_CAPTION>",
        ]
    return gr.update(choices=choices)

def add_tab():
    generate_styles_and_tags = generate_html_code()
    nopreview = os.path.join(extension_path, "nopreview.jpg")
    global hideoldstyles
    with gr.Blocks(analytics_enabled=False,) as ui:
        with gr.Tabs(elem_id = "Stylez"): 
            gr.HTML("""<div id="stylezPreviewBoxid" class="stylezPreviewBox"><p id="stylezPreviewPositive">test</p><p id="stylezPreviewNegative">test</p></div>""")
            with gr.TabItem(label="风格库"):
                with gr.Row():                      
                    with gr.Column(elem_id="style_quicklist_column"):
                        with gr.Row():
                            gr.Text("快速保存提示词",show_label=False)
                            with gr.Row():
                                stylezquicksave_add = gr.Button("添加" ,elem_classes="stylezquicksave_add")
                                stylezquicksave_clear = gr.Button("清除" ,elem_classes="stylezquicksave_add")
                        with gr.Row(elem_id="style_cards_row"):                        
                                gr.HTML("""<ul id="styles_quicksave_list"></ul>""")
                    with gr.Column():
                        with gr.Row(elem_id="style_search_search"):
                            Style_Search = gr.Textbox('', label="搜索框", elem_id="style_search", placeholder="搜索...", elem_classes="textbox", lines=1,scale=3)
                            category_dropdown = gr.Dropdown(label="风格大类", choices=generate_styles_and_tags[1], value="All", elem_id="style_Catagory", elem_classes="dropdown styles_dropdown",scale=1)
                            refresh_button = gr.Button(refresh_symbol, elem_id="style_refresh", elem_classes="tool")
                        with gr.Row():
                            with gr.Column(elem_id="style_cards_column"):
                                Styles_html=gr.HTML(generate_styles_and_tags[0])
                with gr.Row(elem_id="stylesPreviewRow"):
                    gr.Checkbox(value=True,label="应用/移除正向词", elem_id="styles_apply_prompt", elem_classes="styles_checkbox checkbox")
                    gr.Checkbox(value=True,label="应用/移除负向词", elem_id="styles_apply_neg", elem_classes="styles_checkbox checkbox")
                    gr.Checkbox(value=True,label="悬停预览", elem_id="HoverOverStyle_preview", elem_classes="styles_checkbox checkbox")
                    oldstylesCB = gr.Checkbox(value=hideoldstyles,label="隐藏原始样式栏", elem_id="hide_default_styles", elem_classes="styles_checkbox checkbox", interactive=True)
                    setattr(oldstylesCB,"do_not_save_to_config",True)
                    card_size_slider = gr.Slider(value=card_size_value,minimum=card_size_min,maximum=card_size_max,label="预览尺寸:", elem_id="card_thumb_size")
                    setattr(card_size_slider,"do_not_save_to_config",True)
                with gr.Row(elem_id="stylesPreviewRow"):
                    favourite_temp = gr.Text(elem_id="favouriteTempTxt",interactive=False,label="Positive:",lines=2,visible=False)
                    add_favourite_btn = gr.Button(elem_id="stylezAddFavourite",visible=False)
                    remove_favourite_btn = gr.Button(elem_id="stylezRemoveFavourite",visible=False)

            #with gr.TabItem(label="C站热词"):
            #    with gr.Row():
            #        with gr.Column(elem_id="civit_tags_column"):
            #            nsfwlvl = gr.Dropdown(label="NSFW:", choices=["None", "Soft", "Mature", "X"], value="None", lines=1, elem_id="civit_nsfwfilter", elem_classes="dropdown styles_dropdown",scale=1)
            #            sortcivit  = gr.Dropdown(label="分类:", choices=["Most Reactions", "Most Comments", "Newest"], value="Most Reactions", lines=1, elem_id="civit_sortfilter", elem_classes="dropdown styles_dropdown",scale=1)
            #            periodcivit  = gr.Dropdown(label="时间段:", choices=["AllTime", "Year", "Month", "Week", "Day"], value="AllTime", lines=1, elem_id="civit_periodfilter", elem_classes="dropdown styles_dropdown",scale=1)
            #        with gr.Column():
            #            with gr.Row(elem_id="style_search_search"):
            #                fdg = gr.Textbox('', label="搜索框", elem_id="style_search", placeholder="不起作用！API不支持！", elem_classes="textbox", lines=1,scale=3)
            #                civitAI_refresh = gr.Button(refresh_symbol, label="Refresh", elem_id="style_refresh", elem_classes="tool", lines=1)
            #                pagenumber = gr.Number(label="Page:",value=1,minimum=1,visible=False)
            #            with gr.Row():
            #                with gr.Column(elem_id="civit_cards_column"):
            #                    gr.HTML(f"""<div><div id="civitaiimages_loading"><p>Loading...</p></div><div onscroll="civitaiaCursorLoad(this)" id="civitai_cardholder" data-nopreview='{nopreview}'></div></div>""")

            # with gr.TabItem(label="提示词生成",elem_id="styles_libary"):
            #     with gr.Column():
            #         with gr.Column():
            #             with gr.Tabs(elem_id = "libs"):

            #                 with gr.TabItem(label="一般提示词",elem_id="styles_generator"):
            #                     with gr.Row():
            #                         with gr.Column():
            #                             style_geninput_txt = gr.Textbox(label="输入:", lines=7,placeholder="在这里输入原始正向提示词。首次使用会自动下载安装模型文件，保持良好的的网络状况，需要等待几分钟时间。如下载失败请手动安装模型~", elem_classes="stylez_promptgenbox")
            #                             with gr.Row():
            #                                 style_gengrab_btn = gr.Button("获取正向提示词",elem_id="style_promptgengrab_btn")
            #                         with gr.Column():
            #                             style_genoutput_txt = gr.Textbox(label="输出:", lines=7,placeholder="生成润色后的正向提示词",elem_classes="stylez_promptgenbox")
            #                             with gr.Row():
            #                                 style_gen_btn = gr.Button("生成",elem_id="style_promptgen_btn")
            #                                 style_gensend_btn = gr.Button("应用正向提示词",elem_id="style_promptgen_send_btn")
            #                     with gr.Row():
            #                         style_genusecomma_btn = gr.Checkbox(label="使用逗号", value=True)
            #                     with gr.Row():
            #                         with gr.Column():
            #                             style_gen_temp = gr.Slider(label="温度（越高 = 多样性更高但一致性较低）: ", minimum=0.1, maximum=1.0 ,value=0.9)
            #                             style_gen_top_k = gr.Slider(label="top_k（每步采样的字符数量）:", minimum=1, maximum=50 ,value=8,step=1)
            #                         with gr.Column():
            #                             style_max_length = gr.Slider(label="最大字符数量:", minimum=1, maximum=160 ,value=80,step=1)
            #                             style_gen_repetition_penalty = gr.Slider(label="重复惩罚:", minimum=0.1, maximum=2 ,value=1.2,step=0.1)

            with gr.TabItem(label="超级提示词", elem_id="superprompt_generator"):
                with gr.Row():
                    with gr.Column():
                        superprompt_input_txt = gr.Textbox(label="输入:", lines=7, placeholder="在这里输入原始正向提示词。首次使用会自动下载安装模型文件，保持良好的的网络状况，需要等待几分钟时间。如下载失败请手动安装模型~", elem_classes="superprompt_box")
                        with gr.Row():
                            superprompt_gen_btn = gr.Button("获取正向提示词", elem_id="style_superprompt_btn")
                    with gr.Column():
                        superprompt_output_txt = gr.Textbox(label="输出:", lines=7, placeholder="生成润色后的自然语言提示词", elem_classes="superprompt_box")
                        with gr.Row():
                            style_super_btn = gr.Button("生成",elem_id="style_superprompt_btn")
                            superprompt_apply_btn = gr.Button("应用正向提示词", elem_id="style_superprompt_send_btn")
                with gr.Row():
                    superprompt_max_length = gr.Slider(
                        label="最大字符量:", 
                        minimum=25, 
                        maximum=512, 
                        value=128, 
                        step=1
                    )
                    superprompt_seed = gr.Slider(
                        label="种子值:", 
                        minimum=0, 
                        maximum=2**32-1, 
                        value=123456, 
                        step=1
                    )

            with gr.TabItem(label="提示词反推", elem_id="florence_prompt_generator"): # 新增 "提示词反推" Tab
                with gr.Row():
                    with gr.Column():
                        florence_image = gr.Image(
                            sources=["upload"],
                            interactive=True,
                            type="pil",
                            elem_classes="stylez_promptgenbox", # 应用样式类
                        )
                        florence_model_name = gr.Dropdown(
                            label="选择模型",
                            choices=FL.available_models,
                            value=FL.available_models[0],
                        )
                        florence_prompt_type = gr.Dropdown(
                            label="提示类型",
                            choices=FL.available_prompt_type,
                            value=FL.available_prompt_type[0],
                        )
                        florence_max_new_token = gr.Slider(
                            label="最大字符量", value=1024, minimum=1, maximum=4096, step=1
                        )
                    with gr.Column():
                        florence_tags = gr.State(value="")
                        florence_html_tags = gr.HTML(
                            value="输出<br><br><br><br>",
                            label="标签",
                            elem_id="tags",
                            elem_classes="stylez_promptgenbox", # 应用样式类
                        )
                        with gr.Row():
                            parameters_copypaste.bind_buttons(
                                parameters_copypaste.create_buttons(
                                    ["txt2img", "img2img"],
                                ),
                                None,
                                florence_tags,
                            )
                        florence_generate_btn = gr.Button(
                            value="生成", variant="primary", elem_id="style_promptgen_btn"
                        )
                        gr.HTML("""<p style="color: #F36812; font-size: 14px; height: 14px; margin: -5px 0px;">模型选择建议:下载"CogFlorence-2.2-Large"能满足基本反推要求，如果追求更准确的反推可以下载"Florence-2-large-PromptGen-v1.5"</p>""")

            with gr.TabItem(label="风格编辑器",elem_id="styles_editor"):
                with gr.Row():
                    with gr.Column():
                        style_title_txt = gr.Textbox(label="标题:", lines=1,placeholder="标题放在这里！",elem_id="style_title_txt")
                        style_description_txt = gr.Textbox(label="描述:", lines=1,placeholder="描述放在这里！", elem_id="style_description_txt")
                        style_prompt_txt = gr.Textbox(label="正向提示词:", lines=2,placeholder="正向提示词放在这里！", elem_id="style_prompt_txt")
                        style_negative_txt = gr.Textbox(label="负向提示词:", lines=2,placeholder="负向提示词放在这里！", elem_id="style_negative_txt")
                    with gr.Column():
                        with gr.Row():
                            style_save_btn = gr.Button(save_symbol, elem_classes="tool", elem_id="style_save_btn")
                            style_clear_btn = gr.Button(clear_symbol, elem_classes="tool" ,elem_id="style_clear_btn")
                            style_delete_btn = gr.Button(delete_style, elem_classes="tool", elem_id="style_delete_btn")
                        thumbnailbox = gr.Image(value=None,label="缩略图（请使用1:1图片）:",elem_id="style_thumbnailbox",elem_classes="image",interactive=True,type='pil')
                        style_img_url_txt = gr.Text(label=None,lines=1,placeholder="Invisible textbox", elem_id="style_img_url_txt",visible=False)
                with gr.Row():
                    style_grab_current_btn = gr.Button("获取提示词", elem_id="style_grab_current_btn")
                    style_lastgen_btn =gr.Button("获取最新生成图片", elem_id="style_lastgen_btn")
                with gr.Row():
                    with gr.Column():
                            style_filename_txt = gr.Textbox(label="文件名命名:", lines=1,placeholder="文件名", elem_id="style_filename_txt")
                            style_filname_check = gr.HTML("""<p id="style_filename_check" style="color:orange;">请输入文件名！！！</p>""",elem_id="style_filename_check_container")
                    with gr.Column():
                        with gr.Row():
                            style_savefolder_txt = gr.Dropdown(label="保存至文件夹（非中文命名）:", value="Styles", choices=generate_styles_and_tags[2], elem_id="style_savefolder_txt", elem_classes="dropdown",allow_custom_value=True)
                            style_savefolder_temp = gr.Textbox(label="Save Folder:", lines=1, elem_id="style_savefolder_temp",visible=False)
                        style_savefolder_refrsh_btn = gr.Button(refresh_symbol, elem_classes="tool")

            with gr.TabItem(label="尺寸设置", elem_id="Size settings"):
                with gr.Row():
                    with gr.Column():
                        gr.HTML("""<p style="color: #F36812; font-size: 14px; height: 14px; margin: -5px 0px;">宽度×高度（SDXL）:</p>""")
                        with gr.Row():
                            create_ar_button("1024×1024 | 1:1", 1024, 1024, button_class="ar2-button")
                        with gr.Row():
                            create_ar_button("576×1728 | 1:3", 576, 1728)
                            create_ar_button("1728×576 | 3:1", 1728, 576)
                            create_ar_button("576×1664 | 9:26", 576, 1664)
                            create_ar_button("1664×576 | 26:9", 1664, 576)
                        with gr.Row():
                            create_ar_button("640×1600 | 2:5", 640, 1600)
                            create_ar_button("1600×640 | 5:2", 1600, 640)
                            create_ar_button("640×1536 | 5:12", 640, 1536)
                            create_ar_button("1536×640 | 12:5", 1536, 640)
                        with gr.Row():
                            create_ar_button("704×1472 | 11:23", 704, 1472)
                            create_ar_button("1472×704 | 23:11", 1472, 704)
                            create_ar_button("704×1408 | 1:2", 704, 1408)
                            create_ar_button("1408×704 | 2:1", 1408, 704)
                        with gr.Row():
                            create_ar_button("704×1344 | 11:21", 704, 1344)
                            create_ar_button("1344×704 | 21:11", 1344, 704)
                            create_ar_button("768×1344 | 4:7", 768, 1344)
                            create_ar_button("1344×768 | 7:4", 1344, 768, button_class="ar2-button")
                        with gr.Row():
                            create_ar_button("768×1280 | 3:5", 768, 1280)
                            create_ar_button("1280×768 | 5:3", 1280, 768)
                            create_ar_button("832×1216 | 13:19", 832, 1216, button_class="ar2-button")
                            create_ar_button("1216×832 | 19:13", 1216, 832)
                        with gr.Row():
                            create_ar_button("832×1152 | 13:18", 832, 1152)
                            create_ar_button("1152×832 | 18:13", 1152, 832)
                            create_ar_button("896×1152 | 7:9", 896, 1152)
                            create_ar_button("1152×896 | 9:7", 1152, 896)
                        with gr.Row():
                            create_ar_button("896×1088 | 14:17", 896, 1088)
                            create_ar_button("1088×896 | 17:14", 1088, 896)
                            create_ar_button("960×1088 | 15:17", 960, 1088)
                            create_ar_button("1088×960 | 17:15", 1088, 960)
                        with gr.Row():
                            create_ar_button("960×1024 | 15:16", 960, 1024)
                            create_ar_button("1024×960 | 16:15", 1024, 960)
                        gr.HTML("""<p style="color: #F36812; font-size: 14px; height: 14px; margin: -5px 0px;">宽度×高度（SD1.5）:</p>""")
                        with gr.Row():
                            create_ar_button("512×512 | 1:1", 512, 512, button_class="ar2-button")
                            create_ar_button("768×768 | 1:1", 768, 768)
                            create_ar_button("576×1024 | 9:16", 576, 1024)
                            create_ar_button("1024×576 | 16:9", 1024, 576)
                        with gr.Row():
                            create_ar_button("512×768 | 2:3", 512, 768)
                            create_ar_button("768×512 | 3:2", 768, 512)
                            create_ar_button("576×768 | 3:4", 576, 768)
                            create_ar_button("768×576 | 4:3", 768, 576)
                        gr.HTML("""<p style="color: #F36812; font-size: 14px; height: 14px; margin: -5px 0px;">宽度×高度（Custom）近似:</p>""")
                        with gr.Row():
                            create_ar_button("880×1176 | 3:4", 880, 1176)
                            create_ar_button("1176×880 | 4:3", 1176, 880)
                            create_ar_button("768×1360 | 9:16", 768, 1360)
                            create_ar_button("1360×768 | 16:9", 1360, 768)
                        with gr.Row():
                            create_ar_button("1576×656 | 2.39:1", 1576, 656)
                            create_ar_button("1392×752 | 1.85:1", 1392, 752)
                            create_ar_button("1176×888 | 1.33:1", 1176, 888)
                            create_ar_button("1568×664 | 2.35:1", 1568, 664)
                        with gr.Row():
                            create_ar_button("1312×792 | 1.66:1", 1312, 792)
                            create_ar_button("1224×856 | 1.43:1", 1224, 856)
                            create_ar_button("912×1144 | 4:5", 912, 1144)
                            create_ar_button("1296×800 | 1.618:1", 1296, 800)
                        gr.HTML("""<p style="color: #F36812; font-size: 14px; height: 14px; margin: -5px 0px;">宽度×高度（Custom）强制:</p>""")
                        with gr.Row():
                            create_ar_button("720×1280 | 9:16", 720, 1280)
                            create_ar_button("1280×720 | 16:9", 1280, 720)
                            create_ar_button("800×1280 | 10:16", 800, 1280)
                            create_ar_button("1280×800 | 16:10", 1280, 800)

            with gr.TabItem(label="注意"):  # 新增的Tab标题           
                gr.Markdown("""
                <p style="color: #F36812; font-size: 18px; margin-bottom: 8px; height: 12px;">注意事项：</p>
                <p style="margin-bottom: 8px;"><span style="color: #F36812;">1. </span>需要说明的是如果你不小心使用了<span style="color: #F36812;">WebUI</span>生成按钮下面的清空提示词，此插件风格库中你已经选择的风格卡片标记并不会被同步取消，你需要刷新一下风格大类清空标记。</p>
                <p style="margin-bottom: 8px;"><span style="color: #F36812;">2. </span>此插件提示词全部采用标准格式，如果你安装了<span style="color: #F36812;">All in one</span>这个插件，请打开设置菜单点击第二个图标进行Prompt格式调整（勾选第二项去除Prompt最后的一个逗号，其他项全部取消勾选。）</p>
                <p style="margin-bottom: 8px;"><span style="color: #F36812;">3. </span>如果你因为网络问题无法自动下载<span style="color: #F36812;">风格生成器</span>和<span style="color: #F36812;">超级提示词</span>的相关依赖模型，可以访问我的相关文章<a href="https://www.disambo.com/2024/03/18/sd-webui-next-style/" style="color: green;">sd-webui-next-style</a>进行模型下载。</p>
                <p style="margin-bottom: 8px;"><span style="color: #F36812;">4. </span>风格编辑小技巧：任何包含关键字<span style="color: #F36812;">{prompt}</span>的提示都将自动获取你当前的提示，并将其插入到<span style="color: #F36812;">{prompt}</span>的位置。一个简单的示例，你有一个风格的提示词是这样写的<span style="color: Gray;">A dynamic, black-and-white graphic novel scene with intense action, a paiting of {prompt}</span>，现在你在正向提示词中输入<span style="color: Gray;">Several stray cats</span>,当你应用这个风格模板后，正向提示词会变成<span style="color: Gray;">A dynamic, black-and-white graphic novel scene with intense action, a paiting of Several stray cats</span>。总之，如果你想自己编辑风格模板，可以先看看现有模板的格式。</p>
                <p style="margin-bottom: 8px;"><span style="color: #F36812;">5. </span>如果用的愉快请点击下面图标收藏哦！顺便也可以逛逛我的个人网站<a href="https://www.disambo.com" style="color: green;">disambo.com</a></p>
                <a href="https://github.com/Firetheft/sd-webui-next-style" target="_blank">
                    <img src="https://bu.dusays.com/2024/03/10/65edbb64b1ece.png" alt="GitHub" style="height: 24px; width: 24px; margin-right: 8px;"/>
                </a>
                """)
        #civitAI_refresh.click(fn=None,_js="refreshfetchCivitai",inputs=[nsfwlvl,sortcivit,periodcivit])
        #periodcivit.change(fn=None,_js="refreshfetchCivitai",inputs=[nsfwlvl,sortcivit,periodcivit])
        #sortcivit.change(fn=None,_js="refreshfetchCivitai",inputs=[nsfwlvl,sortcivit,periodcivit])
        #nsfwlvl.change(fn=None,_js="refreshfetchCivitai",inputs=[nsfwlvl,sortcivit,periodcivit])
        # style_gengrab_btn.click(fn=None,_js="stylesgrabprompt" ,outputs=[style_geninput_txt])
        # style_gensend_btn.click(fn=None,_js='sendToPromtbox',inputs=[style_genoutput_txt])
        # style_gen_btn.click(fn=generate_style,inputs=[style_geninput_txt,style_gen_temp,style_gen_top_k,style_max_length,style_gen_repetition_penalty,style_genusecomma_btn],outputs=[style_genoutput_txt])
        superprompt_gen_btn.click(fn=None,_js="stylesgrabprompt" ,outputs=[superprompt_input_txt])
        superprompt_apply_btn.click(fn=None,_js='sendToPromtbox',inputs=[superprompt_output_txt])
        style_super_btn.click(fn=call_generate_super_prompt,inputs=[superprompt_input_txt,superprompt_max_length,superprompt_seed],outputs=[superprompt_output_txt])
        oldstylesCB.change(fn=oldstyles,inputs=[oldstylesCB],_js="hideOldStyles")
        refresh_button.click(fn=refresh_styles,inputs=[category_dropdown], outputs=[Styles_html,category_dropdown,category_dropdown,style_savefolder_txt])
        card_size_slider.release(fn=save_card_def,inputs=[card_size_slider])
        card_size_slider.change(fn=None,inputs=[card_size_slider],_js="cardSizeChange")
        category_dropdown.change(fn=None,_js="filterSearch",inputs=[category_dropdown,Style_Search])
        Style_Search.change(fn=None,_js="filterSearch",inputs=[category_dropdown,Style_Search])
        style_img_url_txt.change(fn=img_to_thumbnail, inputs=[style_img_url_txt],outputs=[thumbnailbox])
        style_grab_current_btn.click(fn=None,_js='grabCurrentSettings')
        style_lastgen_btn.click(fn=None,_js='grabLastGeneratedimage')
        style_savefolder_refrsh_btn.click(fn=refresh_styles,inputs=[category_dropdown], outputs=[Styles_html,category_dropdown,category_dropdown,style_savefolder_txt])
        style_save_btn.click(fn=save_style, inputs=[style_title_txt, thumbnailbox, style_description_txt,style_prompt_txt, style_negative_txt, style_filename_txt, style_savefolder_temp], outputs=[style_filname_check])
        style_filename_txt.change(fn=filename_check, inputs=[style_savefolder_temp,style_filename_txt], outputs=[style_filname_check])
        style_savefolder_txt.change(fn=tempfolderbox, inputs=[style_savefolder_txt], outputs=[style_savefolder_temp])
        style_savefolder_temp.change(fn=filename_check, inputs=[style_savefolder_temp,style_filename_txt], outputs=[style_filname_check])
        style_clear_btn.click(fn=clear_style, outputs=[style_title_txt,style_img_url_txt,thumbnailbox,style_description_txt,style_prompt_txt,style_negative_txt,style_filename_txt])
        style_delete_btn.click(fn=deletestyle, inputs=[style_savefolder_temp,style_filename_txt])
        add_favourite_btn.click(fn=addToFavourite, inputs=[favourite_temp])
        remove_favourite_btn.click(fn=removeFavourite, inputs=[favourite_temp])
        stylezquicksave_add.click(fn=None,_js="addQuicksave")
        stylezquicksave_clear.click(fn=None,_js="clearquicklist")
        florence_generate_btn.click(
            fn=wrap_gradio_gpu_call(FL.generate_prompt_fn),
            inputs=[florence_image, florence_model_name, florence_max_new_token, florence_prompt_type],
            outputs=[florence_tags, florence_html_tags],
        )
        florence_model_name.change(
            fn=update_prompt_types,
            inputs=florence_model_name,
            outputs=florence_prompt_type,
        )
    return [(ui, "stylez_menutab", "stylez_menutab")]

script_callbacks.on_ui_tabs(add_tab)
