import os
import sys
import OpenGL.GL as gl
import numpy as np
from PIL import Image
from PySide6.QtCore import QTimerEvent, Qt,QThreadPool, QRunnable, Signal, QObject,QTimer
from PySide6.QtGui import QMouseEvent, QCursor,QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QApplication, QVBoxLayout, QLineEdit
from PySide6.QtGui import QGuiApplication,QAction,QIcon
import live2d.v3 as live2d
from openai import OpenAI
from PySide6.QtWidgets import QSystemTrayIcon, QMenu
# from chat_real_time import Response_To_TTS
from qfluentwidgets import *
from working_mode import TimeManager
from talking_mode import TalkingMode
from modelscope import snapshot_download
from pathlib import Path
import logging
from crawl_part import CrawlWindow
import asyncio
from qasync import QEventLoop
from html import escape
from configs import ConfigManager
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
def download_model():
    try:
        model_path = resource_path('pretrained_models/CosyVoice2-0.5B')
        if not Path(model_path).exists():
            Path(model_path).mkdir(parents=True, exist_ok=True)
            snapshot_download(
                model_id='iic/CosyVoice2-0.5B',
                cache_dir=model_path
            )
        return model_path
    except Exception as e:
        logging.error(f"❌ 模型下载失败: {str(e)}")
        raise  # 抛出异常让上层处理
def callback():
    print("Motion ended")

class TTSWorkerSignals(QObject):
    finished = Signal(str)  # 带有参数的完成信号
    error = Signal(str)

class TTSWorker(QRunnable):
    def __init__(self, result):
        super().__init__()
        self.signals = TTSWorkerSignals()
        self.result = result
        self._tts_engine = None 
        self._should_stop = False

    def run(self):
        try:
            # 按需导入
            from chat_real_time import Response_To_TTS
            if self._tts_engine is None:
                self._tts_engine = Response_To_TTS()
            if self._should_stop:  # 检查停止标志
                return
            self._tts_engine.generate_tts(self.result)
            self.signals.finished.emit(self.result)
        except Exception as e:
            self.signals.error.emit(f"TTS 错误: {str(e)}")
        finally:
            if self._tts_engine is not None:
                # 调用自定义清理方法
                self._tts_engine.release_resources()
                del self._tts_engine
                self._tts_engine = None
            # 强制垃圾回收
            import gc
            gc.collect()
    def stop(self):
        """停止当前任务"""
        self._should_stop = True
        if self._tts_engine is not None:
            self._tts_engine.release_resources()

class AIWorkerSignals(QObject):
    finished = Signal(str)
    error = Signal(str)
    received_baseurl = Signal(str)

class AIWorker(QRunnable):
    def __init__(self, user_prompt):
        super().__init__()
        self.signals = AIWorkerSignals()
        self.config = ConfigManager()
        self.user_prompt = user_prompt
        self.system_prompt = self.config.get_config('selected_prompt_text')
        self.api_key = self.config.get_config('selected_api_key')
        self.base_url = self.config.get_config('base_url')
        self.chat_model = self.config.get_config('selected_model')
    def run(self):
        try:
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            completion = client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.user_prompt}
                ],
                temperature=1.0,
                max_tokens=512
            )
            response = completion.choices[0].message.content
            self.signals.finished.emit(response)
        except Exception as e:
            self.signals.error.emit(f"生成回答失败: {str(e)}")
class Win(QOpenGLWidget):
    def __init__(self):
        super().__init__()
        self.isInLA = False
        self.clickInLA = False
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint|Qt.WindowType.Tool)
        # 设置透明窗口以及透明穿透
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        # self.resize(300, 300)
        #将一开始的展示位置设置在右下角
        # self.setGeometry(0, 0, 300, 300)
        self.read = False
        self.clickX = -1
        self.clickY = -1
        self.model: live2d.LAppModel | None = None
        self.systemScale = QGuiApplication.primaryScreen().devicePixelRatio()
        self.movePressed = False
        self.offset = None
        self.lineEdit = LineEdit(self)  # 设置父级为当前窗口
        self.lineEdit.setFixedHeight(40)
        self.lineEdit.setFixedWidth(250)
        self.lineEdit.setPlaceholderText("请输入你想说的话")
        # self.lineEdit.setClearButtonEnabled(True)
        self.user_prompt = self.lineEdit.text()
        self.lineEdit.returnPressed.connect(self.chat_with_AI)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)  # 设置布局边距
        layout.addStretch(1)  # 添加拉伸因子将控件排列到底部
        # 将输入框添加到布局
        action1 = QAction(self)  # 创建一个 QAction
        action1.setIcon(FluentIcon.SEND.icon()) # 设置图标路径
        action1.setText("发送")  # 设置文本
        action1.triggered.connect(self.chat_with_AI)  # 设置触发事件
        # 添加到 QLineEdit
        self.lineEdit.addAction(action1, QLineEdit.TrailingPosition)
        layout.addWidget(self.lineEdit)
        # 设置窗口的位置
        pos_x, pos_y, window_width, window_height = self.calculate_position()
        self.setGeometry(pos_x, pos_y, window_width, window_height)
        # 初始化系统托盘
        self.init_system_tray()
        # 初始化线程池
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(5)  # 设置最大线程数
        # 子窗口的一些初始化
        self.talking_settings = TalkingMode()
        self.work_mode_window = TimeManager()  # 创建时间管理器实例
        self.work_mode_window.rest_triggered.connect(self.handle_rest_notification)
        self.voice_worker = None
        self.rss_window = CrawlWindow(parent=self)
        self.timer_id = None  # 新增
        self.live2d_initialized = False
        self.last_rss_data = None
        self.init_auto_crawl()
        self.tts_enabled = False # 默认不启用 TTS
        self.active_tts_workers = [] # 新增：跟踪活动的TTS线程
        self.config = ConfigManager()
        # 监听器
        self.listening_timer = QTimer()
        self.listening_timer.timeout.connect(self.process_continuous_audio)
        self.is_in_conversation = False  # 对话状态标志
        # 设置对话超时
        self.conversation_timeout = 10000  # 10秒超时
        self.conversation_timer = QTimer()
        self.conversation_timer.timeout.connect(self.stop_continuous_listening)
        # self.lineEdit.setAlignment(Qt.AlignmentFlag)
        # self.time_manager.input_time()
    def handle_tts_enabled(self, enabled):
        self.tts_enabled = enabled
        if not enabled:
            # 停止并清理所有 TTS 资源
            for worker in self.active_tts_workers:
                if isinstance(worker, TTSWorker):
                    worker.stop()  # 调用停止方法
                    if worker._tts_engine is not None:
                        worker._tts_engine.release_resources()
            self.active_tts_workers.clear()  
            # 强制垃圾回收
            import gc
            gc.collect()

    def init_auto_crawl(self):
        # 每3小时（10800000毫秒）触发一次爬取
        self.crawl_timer = self.startTimer(10800000)
        # 立即执行第一次爬取
        QTimer.singleShot(1000, self.start_auto_crawl)
    def start_auto_crawl(self):
        self.rss_window.start_crawling()
    def init_system_tray(self):
        # 创建托盘图标
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(QIcon(resource_path("UI_icons/logo.png"))) 
        self.tray_icon.setToolTip("OwnNeuro")

        # 创建右键菜单
        self.tray_menu = QMenu()
        
        # 添加菜单项
        self.show_action = QAction("显示窗口", self)
        self.work_mode_action = QAction("工作模式设置", self)
        self.talk_mode_action = QAction("聊天模式设置", self)
        self.rss_action = QAction("RSS订阅", self)
        self.quit_action = QAction("退出", self)

        self.tray_menu.addAction(self.show_action)
        self.tray_menu.addAction(self.work_mode_action)
        self.tray_menu.addAction(self.talk_mode_action)
        self.tray_menu.addAction(self.rss_action)
        self.tray_menu.addSeparator()
        self.tray_menu.addAction(self.quit_action)

        # 连接菜单动作
        self.show_action.triggered.connect(self.toggle_visibility)
        self.work_mode_action.triggered.connect(self.open_work_mode_settings)  
        self.quit_action.triggered.connect(self.quit_app)
        self.rss_action.triggered.connect(self.open_rss_settings)
        self.talk_mode_action.triggered.connect(self.open_talking_settings)
        self.tray_menu.aboutToShow.connect(self.update_menu_status)
        # 设置托盘图标的上下文菜单
        self.tray_icon.setContextMenu(self.tray_menu)

        # 显示托盘图标
        self.tray_icon.show()
    def update_menu_status(self):
        """根据窗口可见性更新菜单项勾选状态"""
        if self.isVisible():
            self.show_action.setText("✓ 显示窗口")
        else:
            self.show_action.setText("显示窗口")

    def toggle_visibility(self):
        """切换窗口可见性并更新菜单"""
        if self.isVisible():
            self.hide()
        else:
            self.show()
        self.update_menu_status() 

    def toggle_visibility(self):
        if self.isVisible():
            self.hide()
        else:
            self.show()
    #打开聊天模式设置窗口
    def open_talking_settings(self):
        if hasattr(self.talking_settings, 'model_changed'):
            print("接收到了模型切换信号")
            self.talking_settings.model_changed.connect(self.update_model_path)  # 连接信号
        if hasattr(self.talking_settings, 'tts_signal'):
            print("接收到了TTS信号")
            self.talking_settings.tts_signal.connect(self.handle_tts_enabled)
        self.talking_settings.wakeup_signal.connect(
        self.handle_wakeup,
        Qt.ConnectionType.UniqueConnection  # 防止重复连接
    )
        self.talking_settings.show()
    def update_model_path(self, model_path):
        print(f"切换模型路径到: {model_path}")
        if live2d.LIVE2D_VERSION == 3 and self.model:
            model3_json_path = self.talking_settings.get_model3_json_path(model_path)
            print(f"加载模型配置文件：{model3_json_path}")
            try:
                live2d.dispose()
                live2d.init() 
                self.model = live2d.LAppModel()
                self.model.LoadModelJson(model3_json_path)
                self.resizeGL(self.width(), self.height())
                self.startTimer(int(1000/120))
            except Exception as e:
                print(f"模型加载失败: {str(e)}")
    def cleanup(self):
        # 清理标志位
        if hasattr(self, '_cleaned'):
            return
        self._cleaned = True  # 防止重复执行

        # 停止定时器
        if self.timer_id is not None:
            self.killTimer(self.timer_id)
            self.timer_id = None

        # 释放 Live2D 资源
        if self.live2d_initialized:
            live2d.dispose()
            self.live2d_initialized = False

        # 隐藏窗口和托盘图标
        self.hide()
        self.tray_icon.hide()

        # 关闭子窗口
        self.talking_settings.close()
        self.work_mode_window.close()
        self.rss_window.close()
    def quit_app(self):
        """安全退出应用程序"""
        self.cleanup()
        self.tray_icon.hide()  # 隐藏托盘图标
        self.talking_settings.close()
        self.work_mode_window.close()
        self.rss_window.close()
        QApplication.quit()
    def closeEvent(self, event):
        event.ignore()  # 忽略关闭事件
        self.hide()    # 隐藏窗口而非退出
    def calculate_position(self):
        # 获取屏幕的尺寸
        screen_geometry = QGuiApplication.primaryScreen().geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()
        
        # 窗口的宽度和高度
        window_width = 250
        window_height = 300
        # 计算窗口的右下角位置
        pos_x = screen_width - window_width
        pos_y = screen_height - window_height 
        return pos_x, pos_y, window_width, window_height
    def showTeachingTip(self, result):
        TeachingTip.create(
            target=self.lineEdit,
            title="Neuro:",
            icon=InfoBarIcon.SUCCESS,
            content=result,
            isClosable=True,
            tailPosition=TeachingTipTailPosition.BOTTOM,
            duration=10000,
            parent=self,
        )
    def initializeGL(self):
        self.makeCurrent()
        
        if live2d.LIVE2D_VERSION == 3:
            live2d.glewInit()

        self.model = live2d.LAppModel()
        self.model.SetAutoBlinkEnable(True)
        self.model.SetAutoBreathEnable(True)

        # 初始加载模型路径
        initial_model_path = self.talking_settings.get_model3_json_path(self.talking_settings.model_paths[0])
        self.model_path = initial_model_path
        if live2d.LIVE2D_VERSION == 3:
            self.model.LoadModelJson(self.model_path)
        if not self.live2d_initialized:
            live2d.init()
            self.live2d_initialized = True 
        # 启动定时器
        self.timer_id = self.startTimer(int(1000/120))

    def timerEvent(self, a: QTimerEvent | None):
        if not self.isVisible():
            return

        local_x, local_y = QCursor.pos().x() - self.x(), QCursor.pos().y() - self.y()
        self.isInLA = self.isInL2DArea(local_x, local_y)
        self.play_idle_animation()
        self.update()
        if a.timerId() == self.crawl_timer:
            self.start_auto_crawl()


    def play_idle_animation(self):
        # 检查当前是否有动画正在播放
        if self.model.IsMotionFinished():
            # 随机播放 Idle 动画
            self.model.StartRandomMotion(
                group="Idle",
                priority=live2d.MotionPriority.NORMAL,
                onStartMotionHandler=None,
                onFinishMotionHandler=None,
            )
        else:
            return  # 有动画正在播放，不启动新的动画
    def isInL2DArea(self, click_x, click_y):
        h = self.height()
        alpha = gl.glReadPixels(click_x * self.systemScale, (h - click_y) * self.systemScale, 1, 1, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)[3]
        return alpha > 0

    def resizeGL(self, w: int, h: int):
        if self.model:
            self.model.Resize(w, h)

    def paintGL(self):
        live2d.clearBuffer()
        self.model.Update()
        self.model.Draw()
        if not self.read:
            self.savePng('screenshot.png')
            self.read = True

    def savePng(self, fName):
        data = gl.glReadPixels(0, 0, self.width(), self.height(), gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
        data = np.frombuffer(data, dtype=np.uint8).reshape(self.height(), self.width(), 4)
        data = np.flipud(data)
        new_data = np.zeros_like(data)
        for rid, row in enumerate(data):
            for cid, col in enumerate(row):
                color = None
                new_data[rid][cid] = col
                if cid > 0 and data[rid][cid - 1][3] == 0 and col[3] != 0:
                    color = new_data[rid][cid - 1]
                elif cid > 0 and data[rid][cid - 1][3] != 0 and col[3] == 0:
                    color = new_data[rid][cid]
                if color is not None:
                    color[0] = 255
                    color[1] = 0
                    color[2] = 0
                    color[3] = 255
                color = None
                if rid > 0:
                    if data[rid - 1][cid][3] == 0 and col[3] != 0:
                        color = new_data[rid - 1][cid]
                    elif data[rid - 1][cid][3] != 0 and col[3] == 0:
                        color = new_data[rid][cid]
                elif col[3] != 0:
                    color = new_data[rid][cid]
                if color is not None:
                    color[0] = 255
                    color[1] = 0
                    color[2] = 0
                    color[3] = 255
        img = Image.fromarray(new_data, 'RGBA')
        img.save(fName)
    def mousePressEvent(self, event: QMouseEvent):
        x, y = event.scenePosition().x(), event.scenePosition().y()
        if self.isInL2DArea(x, y):
            self.clickInLA = True
            self.clickX, self.clickY = x, y
            print("Mouse pressed")
            if event.button() == Qt.MouseButton.LeftButton:
                self.movePressed = True
                self.offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseReleaseEvent(self, event: QMouseEvent):
        x, y = event.scenePosition().x(), event.scenePosition().y()
        if self.isInL2DArea(x, y):
            # 用来展示点击了哪个位置
            self.model.Touch(x, y)
            # self.play_Tapbody_animation(x, y)
            self.clickInLA = False
            print("Mouse released")
        if event.button() == Qt.MouseButton.LeftButton:
            self.movePressed = False

    def mouseMoveEvent(self, event: QMouseEvent):
        x, y = event.scenePosition().x(), event.scenePosition().y()
        if self.clickInLA:
            self.move(int(self.x() + x - self.clickX), int(self.y() + y - self.clickY))
        if self.movePressed:
            new_pos = event.globalPosition().toPoint() - self.offset
            self.move(new_pos)
        self.repaint()

    def open_work_mode_settings(self):
        self.work_mode_window.setWindowModality(Qt.NonModal)
        self.work_mode_window.show()
    def handle_rest_notification(self, result):
        """处理来自工作模式窗口的休息通知"""
        if result == 1:
            text = "长休息时间到了"
        elif result == 0:
            text = "短休息时间到了"
        else:
            text = "继续工作"
        if self.tts_enabled:    
            # 启动TTS线程
            tts_worker = TTSWorker(text)
            tts_worker.signals.finished.connect(self.on_tts_finished)
            tts_worker.signals.error.connect(self.on_tts_error)
            self.active_tts_workers.append(tts_worker)  # 跟踪线程
            self.thread_pool.start(tts_worker)
        else:
            return
    def chat_with_AI(self):
        user_prompt = self.lineEdit.text().strip()
        if not user_prompt:
            print("输入内容不能为空")
            return
        self.lineEdit.setEnabled(False)
        
        # 创建 AI 工作线程
        worker = AIWorker(user_prompt)
        worker.signals.finished.connect(self.on_ai_response)
        worker.signals.error.connect(self.on_ai_error)
        self.thread_pool.start(worker)
    
    def on_ai_response(self, response):
        self.lineEdit.setEnabled(True)
        if self.tts_enabled:
            # 启动 TTS 线程
            tts_worker = TTSWorker(response)
            tts_worker.signals.finished.connect(self.on_tts_finished)  # 连接 TTS 完成信号
            tts_worker.signals.error.connect(self.on_tts_error)
            self.active_tts_workers.append(tts_worker)  # 跟踪线程
            self.thread_pool.start(tts_worker)
        else:
            self.showTeachingTip(response)
    def on_ai_error(self, error_msg):
        self.lineEdit.setEnabled(True)
        print(error_msg)
        self.showTeachingTip("抱歉，暂时无法回答这个问题")
    def on_tts_finished(self, response):
        self.showTeachingTip(response)  # 显示信息
        self.active_tts_workers = [w for w in self.active_tts_workers] # 移除已完成的线程

    def on_tts_error(self, error_msg):
        print(error_msg)
        self.showTeachingTip("抱歉,暂时无法生成tts")
        self.active_tts_workers = [w for w in self.active_tts_workers] # 移除错误的线程
    def open_rss_settings(self):
        """打开RSS设置窗口"""
        self.rss_window.show()
    def handle_wakeup(self, is_detected: bool):
        """处理唤醒信号"""
        if not is_detected:
            return  # 未检测到唤醒词时直接返回
        
        text = "Hello!,How can I help you?"
        # 根据 TTS 开关状态选择不同处理方式
        if self.tts_enabled:
            # TTS 启用时生成语音
            tts_worker = TTSWorker(text)
            tts_worker.signals.finished.connect(self.on_tts_finished)
            tts_worker.signals.error.connect(self.on_tts_error)
            self.active_tts_workers.append(tts_worker)  # 跟踪线程
            self.thread_pool.start(tts_worker)
        else:
            # TTS 禁用时直接显示文字
            self.showTeachingTip(text)
        if is_detected and not self.is_in_conversation:
            self.start_continuous_listening()
            # 更新文字框为绿色
            self.talking_settings.text_win.setStyleSheet("border: 2px solid #00FF00;") 
            self.talking_settings.text_win.show()  # 显示识别窗口
            # 重置超时计时器
            self.conversation_timer.start(self.conversation_timeout)
    def start_continuous_listening(self):
        """启动持续监听"""
        self.is_in_conversation = True
        self.talking_settings.worker.set_continuous_listening(True)
        self.listening_timer.start(5000)  # 5秒间隔
        print("进入持续对话模式")

    def stop_continuous_listening(self):
        """停止持续监听"""
        self.is_in_conversation = False
        self.talking_settings.worker.set_continuous_listening(False)
        self.listening_timer.stop()
        self.conversation_timer.stop()
        self.talking_settings.text_win.hide()  # 隐藏识别窗口
        print("退出持续对话模式")
    def process_continuous_audio(self):
        """处理持续监听到的文本"""
        if not self.is_in_conversation:
            return
        # 获取最新识别的文本
        current_text = self.talking_settings.text_win.label.text()
        if current_text and current_text != "请开始说话(please speak)":
            # 重置超时计时器
            self.conversation_timer.start(self.conversation_timeout)
            self.chat_with_ai_continuous(current_text)

    def chat_with_ai_continuous(self, text):
        """自动发起AI请求"""
        worker = AIWorker(text)
        worker.signals.finished.connect(self.handle_ai_response)
        worker.signals.error.connect(self.on_ai_error)
        self.thread_pool.start(worker)

    def handle_ai_response(self, response):
        """处理AI回复"""
        self.showTeachingTip(response)
        if self.tts_enabled:
            tts_worker = TTSWorker(response)
            tts_worker.signals.finished.connect(lambda: self.stop_continuous_listening())
            self.thread_pool.start(tts_worker)
        else:
            self.stop_continuous_listening()
    def handle_Rss_Message(self):
        data = self.rss_window.select_Rss()
        print(f"Received RSS data: {data}")
        if not data:
            return
        
        message = ""
        for item in data:
            message = f'''
            这篇文章不错哦：
            <a href="{escape(item['link'])}" 
            style="color: #2b7af0; text-decoration: none">
            {escape(item['title'])}
            </a><br>
            {escape(item['summary'])}<br><br>
            '''.replace('\n', '')  
        # 创建 QTextBrowser 并设置内容
        browser = TextBrowser(parent=self)
        # browser.setAttribute(Qt.WA_TranslucentBackground, True) 
        browser.setWindowFlags(Qt.WindowStaysOnTopHint)
        browser.setHtml(message)
        browser.setOpenExternalLinks(True)
        browser.setFixedSize(300, 100)

        browser.show()
        
        # 设置定时器，在 5 秒后关闭 QTextBrowser
        QTimer.singleShot(5000, browser.close)
if __name__ == "__main__":
    # 配置基础路径
    whisper_dir = resource_path('whisper_model')
    if not os.path.exists(whisper_dir):
        os.makedirs(whisper_dir, exist_ok=True)
        # 下载语音识别模型
        try:
            snapshot_download(
                model_id='angelala00/faster-whisper-small',
                cache_dir=whisper_dir
            )
        except Exception as e:
            print(f"❌ 语音模型下载失败: {str(e)}")

    # CosyVoice 模型路径
    SAVE_PATH = Path(resource_path('pretrained_models/CosyVoice2-0.5B'))
    
    # 下载主模型（仅在首次运行时下载）
    try:
        if not SAVE_PATH.exists():
            print("⏳ 正在下载语音合成模型...")
            SAVE_PATH.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                model_id='iic/CosyVoice2-0.5B',
                cache_dir=str(SAVE_PATH)
            )
        print(f"✅ 模型已加载：{SAVE_PATH}")
    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        print("⚠️ 部分功能可能受限")
    live2d.init()
    format = QSurfaceFormat.defaultFormat()
    format.setSwapInterval(0)
    QSurfaceFormat.setDefaultFormat(format)
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop) 
    win = Win()
    win.show()
    app.aboutToQuit.connect(win.quit_app)
    with loop:
        loop.run_forever()  # 确保异步任务能执行
    sys.exit(app.exec()) 
    live2d.dispose()