import logging
import soundfile as sf
import pygame
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Signal
# from dashscope.api_entities.dashscope_response import Role
from configs import ConfigManager
import torch
import os 
import sys
import platform
from cli.SparkTTS import SparkTTS
from datetime import datetime

logging.basicConfig(level=logging.DEBUG)

def resource_path(relative_path):
    base_path = sys._MEIPASS if hasattr(sys, '_MEIPASS') else os.path.abspath(".")
    return os.path.normpath(os.path.join(base_path, relative_path))

class Response_To_TTS(QWidget):
    received_audio_path = Signal(str)

    def __init__(self):
        super().__init__()
        self.config = ConfigManager()
        
        # 初始化设备配置
        if platform.system() == "Darwin" and torch.backends.mps.is_available():
            self.device = torch.device("mps:0")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        
        # 初始化SparkTTS模型
        self.spark_tts = SparkTTS(
            model_dir=resource_path('pretrained_models/SparkAudio/Spark-TTS-0.5B'),
            device=self.device
        )
        
        # 配置参数
        self.prompt_speech_path = self.config.get_config('audio_path')
        # self.prompt_text = self.config.get_config('prompt_text')
        self.pitch = self.config.get_config('tts_pitch')
        self.speed = self.config.get_config('tts_speed')

    # def text_conversation(self, text):
    #     messages = [
    #         {'role': Role.SYSTEM, 'content': self.config.get_config('agent_prompt')},
    #         {'role': Role.USER, 'content': text}
    #     ]
    #     response = Generation.call(
    #         model="qwen1.5-0.5b-chat",
    #         messages=messages,
    #         seed=random.randint(1, 10000),
    #         result_format='message'
    #     )
    #     if response.status_code == HTTPStatus.OK:
    #         return response.output['choices'][0]['message']['content']
    #     else:
    #         error_info = f"Status code: {response.status_code}, error: {response.message}"
    #         logging.error(error_info)
    #         return error_info

    def generate_tts(self, output_text):
        try:
            # 执行SparkTTS推理
            wav = self.spark_tts.inference(
                text=output_text,
                prompt_speech_path=self.prompt_speech_path,
                pitch=self.pitch,
                speed=self.speed
            )
            
            # 保存音频文件
            output_path = os.path.join('results', f'output_{datetime.now().strftime("%Y%m%d%H%M%S")}.wav')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, wav, samplerate=16000)
            
            self.play_audio_with_pygame(output_path)
            self.received_audio_path.emit(output_path)
        except Exception as e:
            logging.error(f"语音生成失败: {str(e)}")

    def play_audio_with_pygame(self, file_path):
        pygame.mixer.init()
        try:
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        except Exception as e:
            logging.error(f"音频播放失败: {str(e)}")
        finally:
            pygame.mixer.music.stop()
            pygame.mixer.quit()

    def release_resources(self):
        """ 释放所有资源 """
        if hasattr(self.spark_tts, 'model'):
            del self.spark_tts.model
        pygame.mixer.quit()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    rtts = Response_To_TTS()
    input_text = input("请输入要转换的文本：")
    # output_text = rtts.text_conversation(input_text)
    # if not output_text.startswith("Status code"):
    rtts.generate_tts(input_text)
    rtts.release_resources()