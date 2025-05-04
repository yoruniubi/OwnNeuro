import logging
import random
from dashscope import Generation
from http import HTTPStatus
from cosyvoice.cli.cosyvoice import CosyVoice2
import torchaudio
import pygame
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Signal
from dashscope.api_entities.dashscope_response import Role
from configs import ConfigManager
import torch
import os 
import sys
logging.basicConfig(level=logging.DEBUG)

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

def load_wav(wav, target_sr):
    speech, sample_rate = torchaudio.load(wav)
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        assert sample_rate > target_sr, 'wav sample rate {} must be greater than {}'.format(sample_rate, target_sr)
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
    return speech

class Response_To_TTS(QWidget):
    received_audio_path = Signal(str)

    def __init__(self):
        self.cosyvoice = CosyVoice2(
            model_dir=resource_path('pretrained_models/CosyVoice2-0.5B'),
            load_jit=True,
            load_onnx=False,
            load_trt=False
        )
        self.config = ConfigManager()
        self._prompt_speech = load_wav(self.config.get_config('audio_path'), 16000)
    def text_conversation(self, text):
        user_input = text
        messages = [
            {'role': Role.SYSTEM, 'content': self.config.get_config('agent_prompt')},
            {'role': Role.USER, 'content': user_input}
        ]

        response = Generation.call(
            model="qwen1.5-0.5b-chat",
            messages=messages,
            seed=random.randint(1, 10000),
            result_format='message'
        )
        if response.status_code == HTTPStatus.OK:
            output_text = response.output['choices'][0]['message']['content']
            print(output_text)
            # with open('output1.txt', 'w', encoding='utf-8') as f:
            #     f.write("Output Text:\n")
            #     f.write(output_text + "\n\n")
            return output_text
        
        else:
            error_info = f"Request id: {response.request_id}, Status code: {response.status_code}, " \
                         f"error code: {response.code}, error message: {response.message}"
            print(error_info)
            return error_info
    def generate_tts(self, output_text):
        result = self.cosyvoice.inference_zero_shot(
            tts_text=output_text,
            prompt_text= self.config.get_config('prompt_text'),
            prompt_speech_16k=self._prompt_speech,  # 使用属性方法获取prompt_speech
            stream=False
        )

        for i, j in enumerate(result):
            file_name = f'output_{i}.wav'
            torchaudio.save(
                file_name,
                j['tts_speech'],
                self.cosyvoice.sample_rate,
                format="wav",
                encoding="PCM_S",
                bits_per_sample=16
            )
            self.play_audio_with_pygame(file_name)

    def play_audio_with_pygame(self, file_path):
        pygame.mixer.init()
        try:
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        except pygame.error as e:
            print(f"An error occurred while playing audio: {e}")
        finally:
            pygame.mixer.music.stop()
            pygame.mixer.quit()
    def release_resources(self):
        """释放所有占用资源的对象"""
        # 释放 CosyVoice 模型
        if hasattr(self.cosyvoice, 'model'):
            del self.cosyvoice.model
        if hasattr(self.cosyvoice, 'vocos'):
            del self.cosyvoice.vocos
            
        # 释放音频资源
        pygame.mixer.quit()
        
        # 释放配置引用
        self.config = None
        
        # 显式调用 PyTorch 清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # def handle_receive_audio(self):
    #     if hasattr(self, 'received_audio_path'):

        

if __name__ == "__main__":
    rtts = Response_To_TTS()
    input_text = input("请输入要转换的文本：")
    # output_text = rtts.text_conversation(input_text)
    rtts.generate_tts(input_text)