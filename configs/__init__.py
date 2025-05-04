# configs/__init__.py
import json
from pathlib import Path
import sys
import os
def resource_path(relative_path):
    """ 动态获取资源的绝对路径，兼容开发环境与PyInstaller打包后的环境 """
    if hasattr(sys, '_MEIPASS'):
        # 打包后，资源位于临时目录 sys._MEIPASS 下
        base_path = sys._MEIPASS
    else:
        # 开发时，使用当前目录的相对路径
        base_path = os.path.abspath(".")
    
    # 拼接路径并标准化（处理路径分隔符）
    return os.path.normpath(os.path.join(base_path, relative_path))
class ConfigManager:
    def __init__(self):
        self.config_path = Path(resource_path("configs/config.json"))
        self.config = self._load_config()

    def _load_config(self):
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                "api_key": "",
                "agent_prompt": "",
                "audio_path": "",
                "prompt_text": "",
                "wake_words": ["neuro", "你好"],
                "base_url": "",
                "rss_map":{
                "少数派": "https://sspai.com/feed",
                "爱范儿": "https://ifanr.com/feed",
                "阮一峰的博客": "http://www.ruanyifeng.com/blog/atom.xml"
                },
                "short_break_gap": [0, 30, 0],
                "long_break_gap": [1, 0, 0],
                "short_break_duration": [0, 1, 0],
                "long_break_duration": [0, 5, 0]
            }

    def save_config(self):
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)

    def update_config(self, key, value):
        """ 更新指定键的值并保存 """
        if key in self.config:
            if isinstance(self.config[key], dict) and isinstance(value, dict):
                self.config[key].update(value)  # 合并字典
            else:
                self.config[key] = value
        else:
            self.config[key] = value
        self.save_config()
    def delete_config(self, key, value):
        if key in self.config and isinstance(self.config[key], list):
            # 从列表中移除指定值
            if value in self.config[key]:
                self.config[key].remove(value)
                self.save_config()
            else:
                print(f"Value '{value}' not found in '{key}'.")
        else:
            print(f"Key '{key}' not found or is not a list.")
    def get_config(self, key, default=None, ensure_list=False):
        value = self.config.get(key, default)
        
        if ensure_list:
            if value is None:
                return []
            if isinstance(value, str):
                return [value.strip()] if value.strip() else []
            if isinstance(value, (list, tuple)):
                return list(value)
            return [str(value)]
        
        return value