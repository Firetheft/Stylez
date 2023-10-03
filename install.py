import launch
import os
import pkg_resources
from typing import Tuple, Optional
import shutil

js_file = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"), "Stylez.js")
os_js_folder = os.path.join(os.getcwd(), "javascript")

if not os.path.exists(os.path.join(os_js_folder, "Stylez.js")):
    shutil.copy(js_file, os_js_folder)

