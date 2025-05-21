from qfluentwidgets import *
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget,QLabel,QVBoxLayout,QFileDialog,QApplication
from configs import ConfigManager
from PySide6.QtCore import Signal, Slot
# 用于打开文件管理器
class FilePicker(QWidget):
    audio_path = Signal(str)
    # prompt_text = Signal(str)
    error_signal = Signal(str)  # 新增错误信号
    def __init__(self):
        super().__init__()
        # self.whisper = whisper_model
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.WindowCloseButtonHint | Qt.WindowType.Tool)
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

    def open_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "打开文件")
        if file_name:
            self.label.setText(f"当前音源路径: {file_name}")
            self.audio_path.emit(file_name)  
            # self.handle_audio_file(file_name)

class TTS_Setting(QWidget):
    def __init__(self):
        super().__init__()
        self.config = ConfigManager()
        self.initUI()
    def initUI(self):
        self.setWindowTitle("TTS设置")
        self.setGeometry(100, 100, 400, 300)
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint | 
            Qt.WindowCloseButtonHint | 
            Qt.WindowType.Tool  # 修正后的窗口标志
        )

        # 主布局
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)  # 设置边距
        layout.setSpacing(15)  # 设置控件间距

        # TTS音频选择部分
        self.tts_audio_label = QLabel("TTS参考音频:", self)
        self.tts_audio_label.setStyleSheet("font-size: 13px; color: #666;")
        layout.addWidget(self.tts_audio_label)
        self.tts_audio_line = FilePicker()
        layout.addWidget(self.tts_audio_line)
        self.tts_audio_line.audio_path.connect(self.handle_audio_path)
        # self.tts_audio_line.prompt_text.connect(self.handle_prompt_text)
        self.tts_audio_line.error_signal.connect(self.show_error_toast)

        # 音高设置部分
        self.tts_pitch_label = QLabel("音高:", self)
        self.tts_pitch_label.setStyleSheet("font-size: 13px; color: #666;")
        layout.addWidget(self.tts_pitch_label)
        self.tts_pitch_line = EditableComboBox()
        self.tts_pitch_line.addItems(["very_low", "low", "moderate", "high", "very_high"])
        # 默认选中moderate
        self.tts_pitch_line.setCurrentText("moderate")
        self.tts_pitch_line.currentTextChanged.connect(self.update_pitch)
        layout.addWidget(self.tts_pitch_line)

        # 语速设置部分
        self.tts_speed_label = QLabel("语速:", self)
        self.tts_speed_label.setStyleSheet("font-size: 13px; color: #666;")
        layout.addWidget(self.tts_speed_label)
        self.tts_speed_line = EditableComboBox()
        self.tts_speed_line.addItems(["very_low", "low", "moderate", "high", "very_high"])
        # 默认选中moderate
        self.tts_speed_line.setCurrentText("moderate")
        self.tts_speed_line.currentTextChanged.connect(self.update_speed)
        layout.addWidget(self.tts_speed_line)

        # 添加拉伸，使控件集中在顶部，下方留白
        layout.addStretch(1)

        # 设置主布局
        self.setLayout(layout)
    
    # def handle_prompt_text(self, text):
    #     if text:
    #         self.config.update_config("prompt_text", text)
    #         print(f"[INFO] 已保存提示文本: {text}")
    #     else:
    #         return
    def handle_audio_path(self, audio_path):
        if audio_path:
            self.config.update_config("audio_path", audio_path)
            print(f"[INFO] 已保存音频路径: {audio_path}")
        else:
            return
    def update_pitch(self, text):
        # 存在配置文件中
        if self.config.get_config("tts_pitch") != text:
            self.config.update_config("tts_pitch", text)
    def update_speed(self, text):
        if self.config.get_config("tts_speed") != text:
            self.config.update_config("tts_speed", text)
    @Slot(str)
    def show_error_toast(self, msg):
        Flyout.create(
            title="⚠️ 错误",
            content=msg,
            target=self.model_label,
            parent=self,
        )

if __name__ == "__main__":
    app = QApplication([])
    window = TTS_Setting()
    window.show()
    app.exec()
