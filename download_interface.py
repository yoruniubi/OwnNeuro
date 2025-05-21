# download_interface.py
import os
import sys
import logging
from pathlib import Path
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtWidgets import QWidget, QVBoxLayout, QApplication
from qfluentwidgets import (
    ProgressBar, BodyLabel, TitleLabel, FluentWindow,
    MessageBox, StateToolTip
)
from modelscope import snapshot_download
from modelscope.hub.api import HubApi

def resource_path(relative_path):
    """获取资源绝对路径"""
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.normpath(os.path.join(base_path, relative_path))

class DownloadWorker(QThread):
    """后台下载线程"""
    finished = Signal()
    error = Signal(str)

    def __init__(self, model_id, cache_dir):
        super().__init__()
        self.model_id = model_id
        self.cache_dir = cache_dir
        self._is_running = True

    def run(self):
        try:
            snapshot_download(
                model_id=self.model_id,
                cache_dir=self.cache_dir,
                # resume_from_disk=True,
                # user_agent={"user_agent": "ownneuro-app/1.0.0"}
            )
            self.finished.emit()
        except Exception as e:
            self.error.emit(f"下载失败: {str(e)}")
            logging.error(f"Download error: {str(e)}")

class DownloadWindow(FluentWindow):
    """模型下载界面"""
    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 300)
        
        # 创建下载容器
        self.download_widget = QWidget()
        self.download_widget.setObjectName("downloadContainer")  # 修复objectName问题
        self.download_layout = QVBoxLayout(self.download_widget)
        
        # 界面元素
        self.title_label = TitleLabel("首次运行需要下载模型", self)
        self.tip_label = BodyLabel("请保持网络畅通，下载完成后将自动进入程序", self)
        self.progress_bar = ProgressBar(self)
        self.state_tooltip = StateToolTip("正在初始化", "", self)
        
        # 布局设置
        self.download_layout.addSpacing(20)
        self.download_layout.addWidget(self.title_label, 0, Qt.AlignHCenter)
        self.download_layout.addSpacing(10)
        self.download_layout.addWidget(self.tip_label, 0, Qt.AlignHCenter)
        self.download_layout.addSpacing(20)
        self.download_layout.addWidget(self.progress_bar)
        self.download_layout.addSpacing(30)
        
        self.addSubInterface(self.download_widget, "model_download", "模型下载")
        self.progress_bar.setValue(0)
        
        # 初始化下载参数
        self.worker = None
        self.model_path = ""
        self.downloading = False
        self.total_size = 0
        self.downloaded_size = 0
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress_manually)

    def start_download(self, model_id, save_path):
        """启动下载任务"""
        if self.downloading:
            return
        
        self.model_path = str(save_path)
        self.downloading = True
        self.state_tooltip = StateToolTip("正在下载模型", "", self)
        self.state_tooltip.move(30, 60)
        
        # 获取模型总大小
        try:
            model_info = HubApi().get_model(model_id=model_id)
            self.total_size = model_info['size']
        except Exception as e:
            logging.warning(f"获取模型大小失败: {str(e)}")
            self.total_size = 0
        
        self.worker = DownloadWorker(model_id, self.model_path)
        self.worker.finished.connect(self.on_download_finished)
        self.worker.error.connect(self.show_error)
        self.worker.start()
        
        self.progress_timer.start(1000)  # 每秒更新进度

    def update_progress_manually(self):
        """手动更新下载进度"""
        if not self.downloading:
            return
        
        try:
            # 计算已下载文件大小
            downloaded = sum(
                f.stat().st_size 
                for f in Path(self.model_path).rglob('*') 
                if f.is_file()
            )
            
            if self.total_size > 0:
                progress = int(downloaded / self.total_size * 100)
                status = f"下载中 {downloaded/1024/1024:.1f}MB/{self.total_size/1024/1024:.1f}MB"
                self._update_ui(progress, status)
        except Exception as e:
            logging.warning(f"进度更新失败: {str(e)}")

    def _update_ui(self, value, text):
        """更新界面元素"""
        self.progress_bar.setValue(value)
        self.state_tooltip.setContent(text)
        if value >= 100:
            self.state_tooltip.setContent("下载完成，正在初始化...")
            self.state_tooltip.setState(True)
            self.progress_bar.setVisible(False)

    def on_download_finished(self):
        """下载完成处理"""
        self.downloading = False
        self.progress_timer.stop()
        self.state_tooltip.setState(True)
        self.close()

    def show_error(self, error_msg):
        """显示错误信息"""
        self.downloading = False
        self.progress_timer.stop()
        self.state_tooltip.setState(False)
        MessageBox("错误", error_msg, self).exec()
        sys.exit(1)

    def closeEvent(self, event):
        if self.downloading:
            event.ignore()
        else:
            super().closeEvent(event)


def check_models():
    """检查模型是否存在，返回缺失的模型列表"""
    missing_models = []
    
    # 检查第一个模型
    model1_path = resource_path('pretrained_models/SparkAudio/Spark-TTS-0.5B')
    if not os.path.exists(model1_path):
        missing_models.append(('SparkAudio/Spark-TTS-0.5B', resource_path('pretrained_models')))
    
    # 检查第二个模型
    model2_path = resource_path('whisper_model/angelala00/faster-whisper-small')
    if not os.path.exists(model2_path):
        missing_models.append(('angelala00/faster-whisper-small', resource_path('whisper_model')))
    
    return missing_models
if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = DownloadWindow()
    
    # 测试下载示例模型
    test_path = resource_path('test_download')
    Path(test_path).mkdir(exist_ok=True)
    win.start_download('damo/nlp_structbert_word-segmentation_chinese-base', test_path)
    
    win.show()
    sys.exit(app.exec())