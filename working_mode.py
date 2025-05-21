from qfluentwidgets import *
from PySide6.QtCore import Qt,QTimer
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, 
                              QHBoxLayout, QLabel, QSpacerItem, QSizePolicy)
import sys
from configs import ConfigManager
class TimeManager(QWidget):
    need_reminder = False
    rest_triggered = Signal(int)  # 0: 短休息 1: 长休息 2: 保持
    def __init__(self, parent=None):
        super().__init__(parent)
        # 初始化配置管理器
        self.config_manager = ConfigManager()
        self.default_config = {
            "short_break_gap": self.config_manager.get_config("short_break_gap"),
            "long_break_gap": self.config_manager.get_config("long_break_gap"),
            "short_break_duration": self.config_manager.get_config("short_break_duration"),
            "long_break_duration": self.config_manager.get_config("long_break_duration")
        }
        self.time_controls = {
            "short_break_gap": None,
            "long_break_gap": None,
            "short_break_duration": None,
            "long_break_duration": None
        }
        self.remaining_seconds = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_countdown)
        self.is_working = True
        self.setup_ui()
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.Tool)
        self.short_break_count = 0
        self.long_break_count = 0
        self.is_short_break = True
        # 加载配置
        self.load_config()
    
    def setup_ui(self):
        """初始化界面布局"""
        self.setWindowTitle("Time Settings")
        self.layout = QVBoxLayout(self)
        
        # 倒计时显示标签
        self.countdown_label = QLabel("剩余时间: 00:00:00")
        self.countdown_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.countdown_label)
        
        # 创建时间设置项
        self.time_controls["short_break_gap"] = self.create_time_setting(
            "短休息的时间间隔:", *self.default_config["short_break_gap"])
        self.time_controls["long_break_gap"] = self.create_time_setting(
            "长休息的时间间隔:", *self.default_config["long_break_gap"])
        self.time_controls["short_break_duration"] = self.create_time_setting(
            "短休息时间长度:", *self.default_config["short_break_duration"])
        self.time_controls["long_break_duration"] = self.create_time_setting(
            "长休息时间长度:", *self.default_config["long_break_duration"])

        # 确认按钮
        self.confirm_btn = PrimaryPushButton("确认")
        self.confirm_btn.clicked.connect(self.on_confirm)
        
        # 添加布局
        for layout in self.time_controls.values():
            self.layout.addLayout(layout)
        self.layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
        self.layout.addWidget(self.confirm_btn, alignment=Qt.AlignCenter)
        self.resize(700, 250)
    def load_config(self):
        """从ConfigManager加载配置到UI控件"""
        try:
            for key in self.time_controls:
                value = self.config_manager.get_config(key)
                if value is not None:
                    layout = self.time_controls[key]
                    h, m, s = value
                    layout.itemAt(1).widget().setValue(h)
                    layout.itemAt(3).widget().setValue(m)
                    layout.itemAt(5).widget().setValue(s)
        except Exception as e:
            print(f"加载配置失败: {str(e)}")
    def save_config(self):
        """将UI控件的值保存到ConfigManager"""
        try:
            config = {
                "short_break_gap": self.get_time_values("short_break_gap"),
                "long_break_gap": self.get_time_values("long_break_gap"),
                "short_break_duration": self.get_time_values("short_break_duration"),
                "long_break_duration": self.get_time_values("long_break_duration")
            }
            for key, value in config.items():
                self.config_manager.update_config(key, value)
        except Exception as e:
            MessageBox("保存失败", f"配置文件保存失败: {str(e)}", self).exec()
    def create_time_setting(self, label_text, h, m, s):
        """创建单个时间设置项"""
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label_text))
        
        # 小时控件
        hour = CompactSpinBox()
        hour.setRange(0, 23)
        hour.setValue(h)
        layout.addWidget(hour)
        layout.addWidget(QLabel("Hour(s)"))
        
        # 分钟控件
        minute = CompactSpinBox()
        minute.setRange(0, 59)
        minute.setValue(m)
        layout.addWidget(minute)
        layout.addWidget(QLabel("Minute(s)"))
        
        # 秒控件
        second = CompactSpinBox()
        second.setRange(0, 59)
        second.setValue(s)
        layout.addWidget(second)
        layout.addWidget(QLabel("Second(s)"))
        
        return layout
    def tuple_to_seconds(self,time_tuple):
        """将 (h, m, s) 元组转换为总秒数"""
        h, m, s = time_tuple
        return h * 3600 + m * 60 + s
    def get_time_values(self, layout_name):
        """从指定布局中获取时间元组 (h, m, s)"""
        layout = self.time_controls[layout_name]
        return (
            layout.itemAt(1).widget().value(),  # 小时
            layout.itemAt(3).widget().value(),  # 分钟
            layout.itemAt(5).widget().value()   # 秒
        )
    def start_countdown(self, total_seconds):
        """启动倒计时"""
        self.remaining_seconds = total_seconds
        self.update_countdown_display()
        self.timer.start(1000)  # 每秒更新一次
    def update_countdown(self):
        """更新倒计时"""
        if self.remaining_seconds > 0:
            self.remaining_seconds -= 1
            self.update_countdown_display()
        else:
            self.timer.stop()
            result = self.send_rest()
            
            # 发射信号通知主窗口
            self.rest_triggered.emit(result)
            
            if result == 1:
                duration = self.tuple_to_seconds(self.get_time_values("long_break_duration"))
                self.is_working = False
                self.start_countdown(duration)
            elif result == 0:
                duration = self.tuple_to_seconds(self.get_time_values("short_break_duration"))
                self.is_working = False
                self.start_countdown(duration)
            else:
                gap = self.tuple_to_seconds(self.get_time_values("short_break_gap"))
                self.is_working = True
                self.start_countdown(gap)
                self.countdown_label.setText("")
    def send_rest(self):
        """判断是否需要切换休息模式"""
        # 设置一个标志，初始为 False
        if self.is_short_break:
            self.short_break_count += 1
            # 获取当前设置的间隔和时长
            gap = self.tuple_to_seconds(self.get_time_values("short_break_gap"))
            duration = self.tuple_to_seconds(self.get_time_values("short_break_duration"))
            total_time = self.short_break_count * duration
            
            if total_time >= gap:
                self.is_short_break = False
                self.long_break_count = 0
                self.need_reminder = True
                return 1  # 切换长休息
        else:
            self.long_break_count += 1
            gap = self.tuple_to_seconds(self.get_time_values("long_break_gap"))
            duration = self.tuple_to_seconds(self.get_time_values("long_break_duration"))
            total_time = self.long_break_count * duration
            
            if total_time >= gap:
                self.is_short_break = True
                self.short_break_count = 0
                self.need_reminder = True
                return 0  # 切换短休息
        
        # 如果不需要提醒，则返回 2
        return 2  # 保持当前模式
    def update_countdown_display(self):
        """更新倒计时显示"""
        hours = self.remaining_seconds // 3600
        minutes = (self.remaining_seconds % 3600) // 60
        seconds = self.remaining_seconds % 60
        self.countdown_label.setText(
            f"{'工作时间' if self.is_working else '休息时间'} 剩余: " 
            f"{hours:02}:{minutes:02}:{seconds:02}"
        )
    
    def on_confirm(self):
        """确认按钮点击事件"""
        try:
            self.save_config()  # 改用ConfigManager保存
            settings = {
                "short_break_gap": self.get_time_values("short_break_gap"),
                "long_break_gap": self.get_time_values("long_break_gap"),
                "short_break_duration": self.get_time_values("short_break_duration"),
                "long_break_duration": self.get_time_values("long_break_duration")
            }
            if any(v < 0 for group in settings.values() for v in group):
                raise ValueError("时间参数不能为负值")
            initial_gap = self.tuple_to_seconds(settings["short_break_gap"])
            self.is_working = True
            self.start_countdown(initial_gap)
            MessageBox("设置成功", "时间参数已保存!", self).exec()
        except Exception as e:
            MessageBox("错误", str(e), self).exec()            # self.hide()

    def closeEvent(self, event):
        """重写关闭事件"""
        # super().closeEvent(event)
        event.ignore()
        self.hide()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TimeManager()
    window.show()
    sys.exit(app.exec())