from qfluentwidgets import *
from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Signal,Slot,QThread,QRect,QRunnable,QThreadPool
from PySide6.QtWidgets import QVBoxLayout, QLabel,QWidget,QApplication,QFileDialog
import sys
import os
import json
from fix_motion import load_all_motion_path_from_model_dir, copy_modify_from_motion
import argparse
import numpy as np
import speech_recognition as sr
from faster_whisper import WhisperModel
from queue import Queue
from time import sleep
from sys import platform
import logging
from PySide6.QtCore import QTimer
from openai import OpenAI
logging.basicConfig(level=logging.DEBUG)
from configs import ConfigManager
import shutil
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

class ModelFetcherSignals(QObject):
    result = Signal(list)
    error = Signal(str)

class ModelFetcher(QRunnable):
    def __init__(self, base_url, api_key):
        super().__init__()
        self.signals = ModelFetcherSignals()
        self.base_url = base_url
        self.api_key = api_key

    def run(self):
        try:
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
            models = client.models.list()
            model_ids = [model.id for model in models.data]
            self.signals.result.emit(model_ids)
        except Exception as e:
            self.signals.error.emit(f"获取失败: {str(e)}")
        # finally:
        #     self.deleteLater()  # 关键：确保对象自动销毁
# 新添加一个类用于添加模型到models文件夹
class AddLive2dModel(QObject):
    import_success = Signal(str)
    import_error = Signal(str) 

    def __init__(self):
        super().__init__()
        self.models_dir = resource_path("./models")
        self._ensure_directory()

    def _ensure_directory(self):
        """确保模型目录存在"""
        try:
            os.makedirs(self.models_dir, exist_ok=True)
        except Exception as e:
            self.import_error.emit(f"创建模型目录失败: {str(e)}")

    def start_import(self):
        """启动导入流程"""
        # 选择文件夹对话框
        dir_path = QFileDialog.getExistingDirectory(
            None, 
            "选择Live2D模型文件夹",
            "", 
            QFileDialog.ShowDirsOnly
        )
        if dir_path:
            self._copy_model(dir_path)

    def _copy_model(self, src_path):
        """执行模型复制逻辑"""
        try:
            # 验证源路径有效性
            if not os.path.isdir(src_path):
                raise ValueError("必须选择文件夹")

            # 获取模型名称并处理重名
            base_name = os.path.basename(src_path.rstrip("/\\"))
            dest_path = os.path.join(self.models_dir, base_name)
            
            # 处理重复名称
            counter = 1
            while os.path.exists(dest_path):
                new_name = f"{base_name}_{counter}"
                dest_path = os.path.join(self.models_dir, new_name)
                counter += 1

            # 复制文件
            shutil.copytree(src_path, dest_path)
            self.import_success.emit(os.path.basename(dest_path))
            
        except Exception as e:
            self.import_error.emit(f"导入失败: {str(e)}")

# 负责显示语音识别出来的文字
class TransparentWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.label.setText("请开始说话(please speak)")  # 添加默认文本
        self.adjustSize()
        self.label.setVisible(True)
        self.hide()  # 初始隐藏窗口  
    def initUI(self):
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # 创建标签
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("""
            font-size: 24px;
            color: white;
            background-color: rgba(0, 0, 0, 150);
            border: 2px solid black;
            border-radius: 10px;
        """)
        self.label.adjustSize() 
        # 设置窗口初始尺寸
        width = 800  # 明确定义宽度
        height = 100 # 明确定义高度
        self.resize(width, height)

        # 获取屏幕尺寸
        screen = QApplication.primaryScreen()
        if screen:
            screen_geometry = screen.availableGeometry()
        else:
            # 备用方案：使用默认尺寸
            screen_geometry = QRect(0, 0, 800, 600)

        # 计算窗口位置（右下角留50像素边距）
        window_x = screen_geometry.width() - width - 50
        window_y = screen_geometry.height() - height - 50

        # 设置窗口几何（位置和尺寸）
        self.setGeometry(
            window_x,
            window_y,
            width,
            height
        )
    @Slot(str)
    def update_text(self, text):
        if not text:  # 收到空文本时隐藏
            self.hide()
            return
            
        self.show()
        print(f"更新文本: {text}") 
        self.label.setText(text)
        self.label.adjustSize()
        new_height = self.label.height() + 40
        self.resize(800, min(new_height, 400))  # 限制最大高度
        self.adjustSize()

class Record_Worker(QObject):
    wakeup_detected = Signal(bool)  # 唤醒词检测信号
    text_recognized = Signal(str)  #文本识别信号
    error_occurred = Signal(str)   # 错误信号
    ai_trigger = Signal(str)  # 语音唤醒后是否与AI对话触发信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.args = self.parse_args()
        self.model = WhisperModel(
                model_size_or_path=resource_path('./whisper_model/angelala00/faster-whisper-small'),
                device='auto'
            )
        self.recorder = None
        self.data_queue = Queue()
        self.phrase_time = None
        self.running = True
        self.stop_listening = None
        self.continuous_listening = False  # 新增持续监听状态
        self.config = ConfigManager()

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", default="small", 
                         help="Model to use",
                         choices=["tiny", "base", "small", "medium", "large"])
        parser.add_argument("--energy_threshold", default=1000,
                         type=int, help="Energy level for mic to detect.")
        parser.add_argument("--record_timeout", default=2,
                         type=float, help="Recording window in seconds.")
        parser.add_argument("--phrase_timeout", default=3,
                         type=float, help="Silence duration between phrases.")
        if 'linux' in platform:
            parser.add_argument("--default_microphone", default='pulse',
                             help="Default microphone name for SpeechRecognition.")
        return parser.parse_args([])  # 使用空列表避免命令行参数依赖

    def initialize(self):
        try:
            # 初始化录音设备
            self.setup_microphone()
            # 加载语音识别模型
            self.running = True
            return True
        except Exception as e:
            self.error_occurred.emit(f"初始化失败: {str(e)}")
            return False

    def setup_microphone(self):
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = self.args.energy_threshold
        self.recorder.dynamic_energy_threshold = False

        if 'linux' in platform:
            self.setup_linux_microphone()
        else:
            self.source = sr.Microphone(sample_rate=16000)

        with self.source:
            self.recorder.adjust_for_ambient_noise(self.source)

    def setup_linux_microphone(self):
        mic_name = self.args.default_microphone
        if mic_name == 'list':
            print("Available microphones:")
            for idx, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"{idx}: {name}")
            return
        else:
            for idx, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    self.source = sr.Microphone(sample_rate=16000, device_index=idx)
                    return
            raise ValueError(f"未找到麦克风: {mic_name}")

    def start_listening(self):
        self.stop_listening = self.recorder.listen_in_background(
            self.source, 
            self.record_callback, 
            phrase_time_limit=self.args.record_timeout
        )
        print("开始监听唤醒词...")

    def record_callback(self, _, audio: sr.AudioData):
        data = audio.get_raw_data()
        self.data_queue.put(data)

    @Slot()
    def run(self):
        if not self.initialize():
            return

        self.start_listening()
        while self.running:
            # try:
            self.process_audio()
            sleep(0.1)  # 降低CPU占用
            # except Exception as e:
            #     self.error_occurred.emit(str(e))
            #     self.stop()
    def set_continuous_listening(self, state):
        self.continuous_listening = state
        if state:
            print("进入持续监听模式")
        else:
            print("退出持续监听模式")

    def process_audio(self):
        if self.data_queue.empty():
            return
        # audio_chunks = [chunk for chunk in self.data_queue.queue if chunk is not None]
        # self.data_queue.queue.clear()

        # if not audio_chunks:
        #     return

        # audio_data = b''.join(audio_chunks)

        audio_data = b''.join(self.data_queue.queue)
        self.data_queue.queue.clear()

        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        segments, _ = self.model.transcribe(audio_np)
        text = " ".join([segment.text.strip() for segment in segments]) if segments else ""
        print(f"识别结果: {text}")
        if self.continuous_listening:  # 持续监听模式下直接发送文本
            self.text_recognized.emit(text.strip())
            if text.strip():  # 非空文本才触发AI
                self.ai_trigger.emit(text.strip())  # 新增信号
        else:  # 原有唤醒词检测逻辑
            self.check_wake_words(text)
    # 在 talking_mode.py 的 check_wake_words 方法中增加唤醒处理
    def check_wake_words(self, text):
        # print(f"检测到音频: {text}")  # 调试输出
        for word in self.config.get_config("wake_words", ensure_list=True):
            if word.lower() in text.lower():
                print(f"检测到唤醒词: {word}")  # 调试输出
                self.wakeup_detected.emit(True)
                self.continuous_listening = True  # 新增：立即切换持续监听模式
                self.data_queue.queue.clear()
                return
            
    def stop(self):
        self.running = False
        
        # 停止后台监听（异步）
        if self.stop_listening:
            try:
                self.stop_listening(wait_for_stop=False)
            except Exception as e:
                print(f"停止监听异常: {str(e)}")
        
        # 安全释放麦克风资源
        def safe_release():
            try:
                # # 正确的SpeechRecognition释放方式
                # if hasattr(self, 'source') and self.source:
                #     self.source.__exit__(None, None, None)  # 触发上下文管理器退出
                    
                # 强制关闭PyAudio实例
                import pyaudio
                p = pyaudio.PyAudio()
                p.terminate()
                
                print("[SUCCESS] 麦克风资源已释放")
            except Exception as e:
                print(f"释放异常: {str(e)}")
        
        # 延迟100ms确保异步执行
        QTimer.singleShot(100, safe_release)
        # # 终止faster_whisperrr
        # self.model.terminate()
        # 清空队列
        self.data_queue.queue.clear()
            
# 用于打开文件管理器
class FilePicker(QWidget):
    audio_path = Signal(str)
    prompt_text = Signal(str)
    error_signal = Signal(str)  # 新增错误信号
    def __init__(self, whisper_model=None):
        super().__init__()
        self.whisper = whisper_model
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.confiig = ConfigManager()
        # 创建组件
        self.button_open = PushButton("打开文件")
        self.label = QLabel()
        audio_path = self.confiig.get_config("audio_path")
        if audio_path:
            self.label.setText(f"当前音源路径: {audio_path}")
        else:
            self.label.setText("当前音源路径: 未设置")
        # self.label.setText(f"当前音源路径: {self.confiig.get_config("audio_path")}")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("""
            color: white;
            background-color: rgba(0, 0, 0, 150);
            border-radius: 5px;
            padding: 8px;
        """)
        # 连接信号
        self.button_open.clicked.connect(self.open_file)
        
        # 布局
        layout = QVBoxLayout()
        layout.addWidget(self.button_open)
        layout.addWidget(self.label)
        self.setLayout(layout)  
        self.resize(300, 100)   # 设置窗口初始尺寸

    def handle_audio_file(self, audio_path):
        # 使用工作线程处理耗时操作
        worker = AudioWorker(audio_path, self.whisper)
        worker.signals.result.connect(self.prompt_text.emit)
        worker.signals.error.connect(self.error_signal.emit)
        QThreadPool.globalInstance().start(worker)

    def open_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "打开文件")
        if file_name:
            self.label.setText(f"当前音源路径: {file_name}")
            self.audio_path.emit(file_name)  
            self.handle_audio_file(file_name)
# 音频处理工作线程
class AudioSignals(QObject):
    result = Signal(str)
    error = Signal(str)
# 音频处理工作线程
class AudioWorker(QRunnable):
    def __init__(self, path, whisper_model):
        super().__init__()
        self.signals = AudioSignals()
        self.path = path
        self.whisper = whisper_model

    def run(self):
        try:
            segments, _ = self.whisper.transcribe(self.path)
            text = " ".join([segment.text.strip() for segment in segments])
            self.signals.result.emit(text)
            print(f"识别结果: {text}")
        except Exception as e:
            self.signals.error.emit(f"音频处理失败: {str(e)}")

class TalkingMode(QWidget):
    model_changed = Signal(str)
    wakeup_signal = Signal(bool)
    tts_signal = Signal(bool)
    model_list_ready = Signal(list)  # 模型列表获取完成信号
    model_fetch_error = Signal(str) 
    ai_request_signal = Signal(str)  # 新增信号
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Talking Mode")
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.Tool)
        self.model_paths = []
        self.worker = Record_Worker()
        self.whisper_model = self.worker.model 
        self.config = ConfigManager()
        # 布局设置
        layout = QVBoxLayout()
        layout.setSpacing(15)
        label = QLabel("聊天模式设置", self)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(label)
        # 创建live2d 模型选择框
        self.comboBox = EditableComboBox()
        self.comboBox.setPlaceholderText("请选择 live2d 模型")
        self.comboBox.activated.connect(self.on_combobox_activated)
        # 添加live2d模型
        self.live2d_label = QLabel("添加live2d模型")
        self.live2d_label.setStyleSheet("font-size: 13px; color: #666;")
        self.live2d_button = PushButton("添加模型", self)
        self.live2d_button.clicked.connect(self.add_live2d_model_from_file)
        # 模型选择说明
        model_label = QLabel("live2d模型选择:", self)
        model_label.setStyleSheet("font-size: 13px; color: #666;")
        # 语音唤醒开关
        self.checkBox = SwitchButton("启用语音唤醒", self)
        self.checkBox.setChecked(False)
        self.checkBox.checkedChanged.connect(self.toggle_voice_wakeup)
        self.checkBox.setOnText("已启用语音唤醒")
        # tts开关
        self.tts_checkBox = SwitchButton("启用语音合成", self)
        self.tts_checkBox.setChecked(False)
        self.tts_checkBox.setOnText("已启用语音合成")
        self.tts_checkBox.checkedChanged.connect(self.toggle_tts)
        # 布局添加组件
        layout.addWidget(self.checkBox)
        layout.addWidget(self.tts_checkBox)
        layout.addWidget(self.live2d_label)
        layout.addWidget(self.live2d_button)
        layout.addWidget(model_label)
        layout.addWidget(self.comboBox)
        # api_key 设置
        api_label = QLabel("API 密钥(回车后保存修改):", self)
        api_label.setStyleSheet("font-size: 13px; color: #666;")
        layout.addWidget(api_label)
        self.api_key_line = EditableComboBox()
        self.api_key_line.setPlaceholderText("请输入你的 API Key")
        # 加载api_key
        current_api_keys = self.config.get_config("api_key", ensure_list=True)
        self.api_key_line.addItems(current_api_keys)
        if current_api_keys:
            self.api_key_line.setCurrentText(current_api_keys[-1])  # 默认选中最新密钥
            # 并将最新密钥添加到selected_api_key
            self.config.update_config("selected_api_key", current_api_keys[-1])
        layout.addWidget(self.api_key_line)
        # base_url 设置
        url_label = QLabel("服务地址：", self)
        url_label.setStyleSheet("font-size: 13px; color: #666;")
        layout.addWidget(url_label)
        self.base_url_line = EditableComboBox()
        self.items = ['qwen', 'openai', 'deepseek', 'anthropic','google_gemini']
        self.base_url_line.setPlaceholderText("请选择你的base_url")
        self.base_url_line.addItems(self.items)
        layout.addWidget(self.base_url_line)
        # 语音唤醒提示词设置
        self.wakeup_label = QLabel("语音唤醒词(回车后保存修改):", self)
        self.wakeup_label.setStyleSheet("font-size: 13px; color: #666;")
        self.wakeup_line = EditableComboBox()
        self.wakeup_line.setPlaceholderText("请输入语音唤醒词")
        wake_up_list = self.config.get_config("wake_words",ensure_list=True)
        self.wakeup_line.addItems(wake_up_list)
        layout.addWidget(self.wakeup_label)
        layout.addWidget(self.wakeup_line)
        # Agent提示词
        self.agent_label = QLabel("Agent提示词(回车后保存修改):", self)
        self.agent_label.setStyleSheet("font-size: 13px; color: #666;")
        self.agent_line = EditableComboBox()
        self.agent_line.setPlaceholderText("请输入Agent提示词")
        agent_list = self.config.get_config("agent_prompt", ensure_list=True)
        self.agent_line.addItems(agent_list)
        layout.addWidget(self.agent_label)
        layout.addWidget(self.agent_line)
        # 选择聊天模型
        self.model_label = QLabel("选择聊天模型:", self)
        self.model_label.setStyleSheet("font-size: 13px; color: #666;")
        layout.addWidget(self.model_label)
        self.model_line = ComboBox()
        self.model_line.setPlaceholderText("请选择聊天模型")
        model_list = self.config.get_config("chat_models", ensure_list=True)
        selected_model = self.config.get_config("selected_model", ensure_list=True)
        # 初始化模型列表时设置默认选中项
        self.model_line.addItems(model_list)
        if selected_model:
            self.model_line.setCurrentText(selected_model[0])
        else:
            self.model_line.setCurrentIndex(-1)
        layout.addWidget(self.model_line)
        # 初始化透明窗口
        self.text_win = TransparentWindow()
        # 初始化线程（替换QThreadPool为QThread）
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)  
        self.setLayout(layout)
        self.load_live2d_folder()
        self.worker.text_recognized.connect(self.text_win.update_text)
        self.worker.wakeup_detected.connect(self.wakeup_signal)
        self.setup_url_mapping()
        # 先断开默认的信号
        self.agent_line.returnPressed.disconnect()
        self.wakeup_line.returnPressed.disconnect()
        self.api_key_line.returnPressed.disconnect()
        self.agent_line.returnPressed.connect(self.handle_agent_edit_finished)
        self.wakeup_line.returnPressed.connect(self.handle_wakeup_edit_finished)
        self.api_key_line.returnPressed.connect(self.handle_api_key_edit_finished)
        self.worker.ai_trigger.connect(self.handle_ai_trigger)
        self.agent_line.currentTextChanged.connect(self.update_selected_prompt) 
        self.model_line.currentTextChanged.connect(self.update_selected_model)
        self.model_importer = AddLive2dModel()
        self.model_importer.import_success.connect(self._handle_import_success)
        self.model_importer.import_error.connect(self._handle_import_error)
        self.api_key_line.currentTextChanged.connect(self.update_selected_api_key)
        self.model_list_ready.connect(self.update_model_combobox)
        self.model_fetch_error.connect(self.show_error_toast)
        self.active_threads = []
        self.thread_pool = QThreadPool()
        self.thread_pool = QThreadPool.globalInstance() 
        self.get_model_list() 
    def get_model_list(self):
        base_url = self.config.get_config("base_url")
        api_key = self.config.get_config("selected_api_key")
        
        if not base_url or not api_key:
            self.model_fetch_error.emit("请先配置API密钥和服务地址")
            return

        # 使用QRunnable代替QThread
        worker = ModelFetcher(base_url, api_key)
        worker.signals.result.connect(self.model_list_ready)
        worker.signals.error.connect(self.model_fetch_error)
        
        # 自动管理线程生命周期
        self.thread_pool.start(worker)
    @Slot(list)
    def update_model_combobox(self, model_ids):
        current_selected = self.model_line.currentText()
        self.model_line.clear()
        self.model_line.addItems(model_ids)
        if current_selected in model_ids:
            self.model_line.setCurrentText(current_selected)
    @Slot(str)
    def show_error_toast(self, msg):
        Flyout.create(
            title="⚠️ 错误",
            content=msg,
            target=self.model_label,
            parent=self,
        )
    @Slot(str)
    def update_selected_api_key(self, api_key):
        """更新当前选中的API Key到配置文件"""
        if api_key:
            self.config.update_config("selected_api_key", api_key)
            print(f"[INFO] 已切换API密钥 -> {api_key[:6]}****")
            self.get_model_list()  # 切换密钥时刷新模型列表
    def add_live2d_model_from_file(self):
        """按钮点击触发的槽函数"""
        self.model_importer.start_import()

    @Slot(str)
    def _handle_import_success(self, model_name):
        """处理导入成功"""
        # 显示浮动提示
        Flyout.create(
            title="🎉 导入成功",
            content=f"模型 {model_name} 已添加至模型库",
            target=self.live2d_button,  # 在按钮旁显示提示
            parent=self,
        )
        # 刷新模型列表
        self.load_live2d_folder()

    @Slot(str)
    def _handle_import_error(self, error_msg):
        """处理导入失败"""
        Flyout.create(
            title="❌ 导入失败",
            content=error_msg,
            target=self.live2d_button,
            parent=self,
            isError=True,
            duration=3000
        )
    @Slot()
    def update_selected_prompt(self):
        """更新选中的prompt_text"""
        selected_text = self.agent_line.currentText().strip()
        if selected_text:
            self.config.update_config("selected_prompt_text", selected_text)
            print(f"[INFO] 已选择提示词: {selected_text}")
    # 添加新的处理函数
    @Slot(str)
    def update_selected_model(self, model_name):
        """更新当前选择的聊天模型到配置文件"""
        if not model_name:
            return
        
        # 获取当前配置
        current_selected = self.config.get_config("selected_model")
        
        # 避免重复添加
        if model_name in current_selected:
            return
        
        # 更新配置（保留历史记录）
        updated_selected = model_name
        self.config.update_config("selected_model", updated_selected)
        
        print(f"[INFO] 已选择聊天模型: {model_name}")
    @Slot()
    def handle_api_key_edit_finished(self):
        """处理API Key编辑完成"""
        text = self.api_key_line.currentText().strip()
        print(f"API密钥已更新为：{text}")
        if text:
            self.update_api_key(text)
            self.get_model_list()  # API key变化时刷新模型列表
    def update_api_key(self, new_api_key):
        # 获取当前配置中的API密钥列表（兼容旧版配置）
        current_api_keys = self.config.get_config("api_key", ensure_list=True)
        
        # 检查新密钥是否有效且未重复
        new_api_key = new_api_key.strip()
        if not new_api_key:
            print("[WARN] 无效的API密钥：空值")
            return
        
        if new_api_key in current_api_keys:
            print("[INFO] 该API密钥已存在")
            return
        
        # 更新配置和UI
        try:
            # 追加新密钥到列表
            updated_api_keys = current_api_keys + [new_api_key]
            self.config.update_config("api_key", updated_api_keys)
            
            # 更新UI控件（EditableComboBox）
            if new_api_key not in [self.api_key_line.itemText(i) for i in range(self.api_key_line.count())]:
                self.api_key_line.addItem(new_api_key)
                self.api_key_line.setCurrentText(new_api_key)
            
            print(f"[INFO] 已追加新API密钥 -> {new_api_key[:6]}****")
        except Exception as e:
            print(f"[ERROR] 更新API密钥失败: {str(e)}")
    def handle_agent_edit_finished(self):
        """处理Agent提示词编辑完成"""
        text = self.agent_line.currentText().strip()
        print(f"Agent提示词已更新为：{text}")
        if text:
            self.update_agent_words(text)
            
    def handle_wakeup_edit_finished(self):
        """处理唤醒词编辑完成"""
        text = self.wakeup_line.currentText().strip()
        print(f"唤醒词已更新为：{text}")
        if text:
            self.update_wake_words(text)

    @Slot(str)
    def handle_ai_trigger(self, text):
        """处理持续监听获得的文本"""
        self.text_win.update_text(text)
        # 将文本传递给AI处理
        self.ai_request_signal.emit(text)  # 需要定义这个信号
    def update_agent_words(self, agent_words):
        """更新Agent提示词（回车确认后生效）"""
        current_config = self.config.get_config("agent_prompt", ensure_list=True)
        current_items = [self.agent_line.itemText(i) for i in range(self.agent_line.count())]
        
        # 过滤空值和已有项
        if not agent_words or agent_words in current_items:
            return
            
        # 处理多词情况（按逗号分隔）
        new_words = [w.strip() for w in agent_words.split(",") if w.strip()]
        added_words = [word for word in new_words 
                      if word not in current_config 
                      and word not in current_items]
        
        if added_words:
            updated_config = current_config + added_words
            self.config.update_config("agent_prompt", updated_config)
            self.agent_line.addItems(added_words)
            print(f"[INFO] 新增Agent提示词: {added_words}")

    def update_wake_words(self, wake_words):
        """更新唤醒词（回车确认后生效）"""
        current_config = self.config.get_config("wake_words", ensure_list=True)
        current_items = [self.wakeup_line.itemText(i) for i in range(self.wakeup_line.count())]
        
        if not wake_words or wake_words in current_items:
            return
            
        # 处理多词情况（按逗号分隔）
        new_words = [w.strip() for w in wake_words.split(",") if w.strip()]
        added_words = [word for word in new_words 
                      if word not in current_config 
                      and word not in current_items]
        
        if added_words:
            updated_config = current_config + added_words
            self.config.update_config("wake_words", updated_config)
            # self.worker.args.wake_words = updated_config
            self.wakeup_line.addItems(added_words)
            print(f"[INFO] 新增唤醒词: {added_words}")

    def setup_url_mapping(self):
        self.base_url_line.currentTextChanged.connect(self.update_base_url)
        # self.base_url_line.currentTextChanged.connect(self.get_model_list)  # 新增

    def handle_baseurl(self, base_url):
        """更新配置文件"""
        if base_url:
            self.config.update_config("base_url", base_url)
            print(f"[INFO] 已保存服务地址: {base_url}")
        else:
            return

    def update_base_url(self, text):
        """根据用户输入自动映射预设地址或保存自定义地址"""
        if text in self.items:
            mapped_url = self.get_base_url(text)
            self.config.update_config("base_url", mapped_url)
            print(f"[INFO] 已映射服务地址: {text} → {mapped_url}")
            self.get_model_list()  # 地址变更时自动刷新
    @Slot(str)
    def get_base_url(self, api_type):
        base_urls = {
            "openai": "https://api.openai.com/v1",
            "deepseek": "https://api.deepseek.com/v1",
            "google_gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "anthropic": "https://api.anthropic.com/v1",
            "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1"
        }
        return base_urls.get(api_type.lower(), "")

    def load_live2d_folder(self):
        """加载模型目录并自动修复动作文件"""
        self.live2d_folder = resource_path('models')
        if not os.path.exists(self.live2d_folder):
            print(f"[ERROR] 模型目录 '{self.live2d_folder}' 不存在")
            return
        
        self.comboBox.clear()
        self.model_paths = []
        
        for filename in os.listdir(self.live2d_folder):
            full_path = os.path.join(self.live2d_folder, filename)
            if not os.path.isdir(full_path):
                print(f"[WARN] 跳过非目录项: {filename}")
                continue
            
            # 添加到下拉框
            self.comboBox.addItem(filename, userData=full_path)
            self.model_paths.append(full_path)
            print(f"[INFO] 加载模型: {filename}")
            
            # 自动检查并修复动作文件
            self.check_Motion_json(full_path)
            model3_json_path = self.get_model3_json_path(full_path)
            self.check_Idle(model3_json_path)

    def get_model3_json_path(self, model_path: str) -> str:
        """生成或检查model3.json路径"""
        model_folder_name = os.path.basename(model_path)
        expected_json_name = f"{model_folder_name}.model3.json"
        expected_json_path = os.path.join(model_path, expected_json_name)
        
        # 检查是否存在完全匹配的文件
        for filename in os.listdir(model_path):
            if filename == expected_json_name:
                return os.path.join(model_path, filename)
        
        # 如果没有找到完全匹配的文件，尝试找到类似的文件（忽略大小写）
        matched_files = []
        for filename in os.listdir(model_path):
            if filename.lower().endswith("model3.json"):
                matched_files.append(filename)
        
        if matched_files:
            # 如果有多个匹配文件，选择第一个
            selected_file = matched_files[0]
            print(f"[WARN] 找到多个可能的model3.json文件，选择第一个: {selected_file}")
            return os.path.join(model_path, selected_file)
        # 如果没有找到任何相关的文件，返回预期的路径
        return expected_json_path
        

    def check_Idle(self, model_path: str):
        """确保Idle动画配置存在（兼容键名大小写）"""
        if not os.path.exists(model_path):
            print(f"[WARN] 配置文件不存在: {model_path}")
            return
        
        try:
            with open(model_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 兼容大小写：查找 "Motions" 或 "motions" 键
            motion_key = next((k for k in data.keys() if k.lower() == "motions"), None)
            if not motion_key:
                # 如果不存在 Motion/motion 键，则创建一个
                data["Motions"] = {}
                motion_key = "Motions"
            
            motions = data[motion_key]
            if "Idle" not in motions and "idle" not in motions:
                # 统一使用首字母大写的键名
                motions["Idle"] = {}
                with open(model_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                print(f"[INFO] 已为 {model_path} 添加Idle动画")
        except Exception as e:
            print(f"[ERROR] 检查Idle失败: {str(e)}")
    def check_Motion_json(self, model_path: str):
        """修复模型的动作文件"""
        motion_files = load_all_motion_path_from_model_dir(model_path)
        if not motion_files:
            print(f"[WARN] 模型 '{model_path}' 无动作文件")
            return
        
        save_dir = os.path.join(model_path, "motions")
        for path in motion_files:
            if not path.endswith(".json"):
                continue
            copy_modify_from_motion(path, save_root=save_dir)

    def on_combobox_activated(self, index: int):
        """用户选择模型时触发信号"""
        selected_path = self.comboBox.itemData(index)
        self.model_changed.emit(selected_path)
        print(f"[INFO] 已切换到模型: {self.comboBox.currentText()}")
    def toggle_voice_wakeup(self, state):
        """直接发送布尔值信号"""
        state == self.checkBox.isChecked()
        print("[INFO] 语音唤醒状态: ", state)
        if state:
            print("[INFO] 启用语音唤醒")
            # self.worker.set_continuous_listening(True)
            if not self.worker_thread.isRunning():
                self.worker_thread.started.connect(self.worker.run)
                self.worker_thread.start()
                self.text_win.show()
        else:
            print("[INFO] 禁用语音唤醒")
            self.worker.stop()
            self.text_win.hide()
            # # 关闭麦克风
            # if self.worker.stop_listening:
            #     self.worker.stop_listening(wait_for_stop=False)
            self.worker_thread.quit()
    def toggle_tts(self, state):
        """直接发送布尔值信号"""
        state == self.tts_checkBox.isChecked()
        print("[INFO] TTS状态: ", state)
        if state:
            self.tts_signal.emit(True)
            print("[INFO] 启用TTS")
        else:
            self.tts_signal.emit(False)
            print("[INFO] 禁用TTS")
            return

    def closeEvent(self, event):
        # self.stop_voice_recognition()
        event.ignore()
        self.hide()
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = TalkingMode()
    win.show()
    sys.exit(app.exec()) 