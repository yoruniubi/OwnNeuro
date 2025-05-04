import asyncio
import feedparser
import aiohttp
from PySide6 import QtWidgets, QtCore
from qfluentwidgets import *
from qasync import QEventLoop, asyncSlot
import json
from PySide6.QtGui import QIcon
from PySide6.QtCore import QUrl,QThreadPool,Qt
from sklearn.feature_extraction.text import TfidfVectorizer
from PySide6.QtWidgets import QWidget
import re
from configs import ConfigManager
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

class RSSMessageBox(MessageBoxBase):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.titleLabel = SubtitleLabel('添加RSS订阅源', self)
        self.urlLineEdit = LineEdit(self)

        self.urlLineEdit.setPlaceholderText('输入URL')
        self.urlLineEdit.setClearButtonEnabled(True)

        self.warningLabel = CaptionLabel("URL 不正确")
        self.warningLabel.setTextColor("#cf1010", QColor(255, 28, 32))

        # add widget to view layout
        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self.urlLineEdit)
        self.viewLayout.addWidget(self.warningLabel)
        self.warningLabel.hide()

        self.widget.setMinimumWidth(350)
    def validate(self):
        """ 重写验证表单数据的方法 """
        isValid = QUrl(self.urlLineEdit.text()).isValid()
        self.warningLabel.setHidden(isValid)
        return isValid

class InterestMessageBox(MessageBoxBase):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.titleLabel = SubtitleLabel('添加兴趣', self)
        self.LineEdit = LineEdit(self)

        self.LineEdit.setPlaceholderText('输入兴趣')
        self.LineEdit.setClearButtonEnabled(True)

        self.warningLabel = CaptionLabel("输入的词汇不正确")
        self.warningLabel.setTextColor("#cf1010", QColor(255, 28, 32))

        # add widget to view layout
        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self.LineEdit)
        self.viewLayout.addWidget(self.warningLabel)
        self.warningLabel.hide()

        self.widget.setMinimumWidth(350)

class SettingCard(GroupHeaderCardWidget):
    keywordAdded = QtCore.Signal(str)  # 新增信号用于传递关键词
    keywordRemoved = QtCore.Signal(str)  # 删除关键词信号
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("RSS管理")
        self.setBorderRadius(8)
        # 主布局
        group_widget = QWidget()
        self.group_layout = QtWidgets.QVBoxLayout(group_widget)  # 改为垂直布局
        self.group_layout.setContentsMargins(0, 0, 0, 0)
        self.group_layout.setSpacing(15)
        # 加载配置文件
        self.config = ConfigManager()
        self.interests_map = self.config.get_config('rss_map')
        # 建议和添加关键词
        suggest_row = QtWidgets.QHBoxLayout()
        suggest_row.setContentsMargins(0, 0, 0, 0)
        suggest_row.addStretch(1)
        suggest_words = self.config.get_config('keywords')
        # 将suggest_words中的词传入keyword
        self.suggest_buttons = []
        for word in suggest_words:
            btn = TogglePushButton(word)
            btn.setCheckable(True)
            btn.setChecked(True)  # 默认选中
            btn.toggled.connect(lambda checked, w=word: self._handle_suggest_toggle(w, checked))
            self.suggest_buttons.append(btn)
            suggest_row.addWidget(btn)
            if btn.isChecked():
                print(f"添加关键词：{word}")
                self.keywordAdded.emit(word)
        self.dynamic_tags_container = QtWidgets.QHBoxLayout()
        self.dynamic_tags_container.setContentsMargins(0, 0, 0, 0)
        suggest_row.addLayout(self.dynamic_tags_container)
        suggest_row.addStretch(1)  # 右拉伸
        self.add_btn = PushButton("添加")
        self.add_btn.clicked.connect(self._add_from_input)
        suggest_row.addWidget(self.add_btn)
        self.group_layout.addLayout(suggest_row)
        # 统一添加到卡片组
        interest_group = self.addGroup(resource_path("UI_icons/interest.png"),"兴趣管理","选择需要的关键词", group_widget)
        interest_group.setSeparatorVisible(True)
        # 订阅源管理部分
        self.rss_group = QWidget()
        self.rss_layout = QtWidgets.QVBoxLayout(self.rss_group)
        # 初始化订阅源复选框
        self.rss_checkboxes = []
        for name in self.interests_map:
            cb = CheckBox(name)
            cb.setChecked(True)
            self.rss_checkboxes.append(cb)
            self.rss_layout.addWidget(cb)
        # 添加到卡片
        # self.scroll_layout.addWidget(rss_group)
        rss_group = self.addGroup(resource_path("UI_icons/rss.png"), "订阅源管理", "选择需要采集的订阅源", self.rss_group)
        rss_group.setSeparatorVisible(True)
       
    def _handle_suggest_toggle(self, word, checked):
        if checked:
            print(f"添加关键词：{word}")
            self.keywordAdded.emit(word)
        else:
            print(f"删除关键词：{word}")
            self.keywordRemoved.emit(word)
    def get_selected_keywords(self):
        return [btn.text() for btn in self.suggest_buttons if btn.isChecked()]
    def _add_from_input(self):
        dialog = InterestMessageBox(parent=self)
        try:
            if dialog.exec():
                TeachingTip.create(
                    title="成功",
                    icon=InfoBarIcon.SUCCESS,
                    content="添加成功...",
                    target = self.add_btn,
                    parent=self,
                    isClosable=True,
                    tailPosition=TeachingTipTailPosition.BOTTOM,
                    duration=500,
                )
                self._add_keyword(dialog.LineEdit.text())
        except Exception as e:
            print(e)
    def _add_keyword(self, word: str):
        if not word.strip():
            return
        # 创建标签按钮
        tag = TogglePushButton(word)
        tag.setChecked(True)
        tag.toggled.connect(lambda checked, w=word: self._handle_suggest_toggle(w, checked))
        # 将标签添加到动态容器中
        self.dynamic_tags_container.addWidget(tag)
        current_keywords = self.config.get_config('keywords') or []
        if word not in current_keywords:
            current_keywords.append(word)
            self.config.update_config('keywords', current_keywords)


class CrawlWindow(QtWidgets.QMainWindow):

    def __init__(self,parent=None):
        super().__init__(parent=parent)
        self.thread_pool = QThreadPool()  # 独立线程池
        self.thread_pool.setMaxThreadCount(2)
        self.setGeometry(100, 100, 400, 500)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.Tool)
        self.config = ConfigManager()
        self.keywords = self.config.get_config('keywords')
        self.init_ui()
        self.hide()
        self.setWindowTitle("RSS management")
        self.last_pushed_index = 0  # 记录最近推送的文章索引
        self.article_history = set()  # 记录已推送文章的哈希值
    def init_ui(self):
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        layout = QtWidgets.QVBoxLayout(main_widget)   
        # 实例化SettingCard并连接信号
        self.setting_card = SettingCard(self)
        self.setting_card.keywordAdded.connect(self._handle_keyword_added)
        self.setting_card.keywordRemoved.connect(self._handle_keyword_removed)
        # 添加订阅源按钮
        btn_layout = QtWidgets.QHBoxLayout()
        self.add_btn = PrimaryPushButton(QIcon(resource_path("UI_icons/add.png")), '添加订阅源')
        self.add_btn.clicked.connect(self.add_rss)  # 连接本地信号
        btn_layout.addWidget(self.add_btn)
        layout.addWidget(self.setting_card)
        layout.addLayout(btn_layout)
        self.last_crawled_data = None
    def _handle_keyword_added(self, word: str):
        """处理新增关键词"""
        if word not in self.keywords:
            self.keywords.append(word)
            self.config.update_config('keywords',self.keywords)
    def _handle_keyword_removed(self, word: str):
        """处理删除关键词"""
        if word in self.keywords:
            self.keywords.remove(word)
            self.config.delete_config('keywords', self.keywords)
    async def fetch_rss(self, url):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise ConnectionError(f"HTTP {response.status}")
                    content = await response.text()
                    feed = feedparser.parse(content)
                    if feed.bozo:
                        raise ValueError("无效的RSS格式")
                    data = {
                        'source': url,
                        'title': feed.feed.get('title', ''),
                        'entries': [
                            {
                                'title': entry.get('title', ''),
                                'link': entry.get('link', ''),
                                'published': entry.get('published', ''),
                                'summary': entry.get('summary', '')
                            } for entry in feed.entries[:5]
                        ]
                    }
                    msg = f"成功采集 {url} ({len(data['entries'])}条)"
                    print(msg)
                    for entry in data['entries']:
                        entry['score'] = self.calculate_article_score(entry)
                    return data
        except Exception as e:
            error_msg = f"错误：{url}\n{str(e)}"
            print(error_msg)
            return None
    def calculate_article_score(self, entry):
        print("当前关键词列表:", self.keywords)
        if not self.keywords:
            print("警告：未设置关键词")
            return "0"  # 返回字符串 "0"

        # 使用正则表达式进行中英文分词
        def tokenize(text):
            # 匹配中文词汇（连续中文字符）和英文单词
            return re.findall(r'[\u4e00-\u9fa5]+|[a-zA-Z0-9]+', text)
        
        # 合并标题和摘要
        raw_text = entry['title'] + entry['summary']
        # 进行分词处理
        tokens = tokenize(raw_text)
        processed_text = ' '.join(tokens)
        
        # 配置TF-IDF（允许单个字符）
        vectorizer = TfidfVectorizer(
            token_pattern=r'(?u)\b\w+\b',  # 匹配任何单词字符
            lowercase=True  # 自动转换为小写
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform([processed_text])
        except ValueError:
            return "0"  # 返回字符串 "0"
        
        # 获取词汇表并转换为小写
        feature_names = [word.lower() for word in vectorizer.get_feature_names_out()]
        
        # 计算得分
        score = 0
        for kw in self.keywords:
            kw_lower = kw.strip().lower()
            if kw_lower in feature_names:
                idx = vectorizer.vocabulary_.get(kw_lower)  # 使用 get 避免 KeyError
                if idx is not None:
                    score += tfidf_matrix[0, idx]
        
        # 将得分转换为字符串并返回
        score_str = f"{round(score, 2)}"
        print(f"关键词: {self.keywords}")
        print(f"有效匹配: {[kw for kw in self.keywords if kw.lower() in feature_names]}")
        print(f"文章得分: {score_str}")
        return score_str
    def get_selected_interests(self):
        return [cb.text() for cb in self.setting_card.rss_checkboxes if cb.isChecked()]

    @asyncSlot()
    async def start_crawling(self):
        selected = self.get_selected_interests()
        
        urls = [self.setting_card.interests_map[i] for i in selected]
        
        # 启动异步任务
        tasks = [self.fetch_rss(url) for url in urls]
        results = await asyncio.gather(*tasks)
        
        # 过滤掉 None 值
        valid_results = [result for result in results if result is not None]
        if valid_results == self.last_crawled_data:
            print("数据无变化，不更新")
            return
        self.last_crawled_data = valid_results
        
        # 保存数据到 JSON 文件
        with open(resource_path('rss_data/rss_data.json'), 'w', encoding='utf-8') as f:
            json.dump(valid_results, f, ensure_ascii=False, indent=4)
        if self.parent() and hasattr(self.parent(), 'handle_Rss_Message'):
            self.parent().handle_Rss_Message()  # 直接调用
        print("采集完成！")

    #用于添加RSS订阅源 
    async def fetch_rss_title(self, url):
        """异步获取RSS源标题"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {'User-Agent': 'Mozilla/5.0'}
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        return None
                    content = await response.text()
                    feed = feedparser.parse(content)
                    return feed.feed.get('title', '').strip()
        except Exception as e:
            print(f"获取RSS标题失败: {str(e)}")
            return None

    def add_rss(self):
        dialog = RSSMessageBox(self)

        if dialog.exec():
            url = dialog.urlLineEdit.text().strip()
            if not QUrl(url).isValid():
                TeachingTip.create(
                title="错误",
                icon=InfoBarIcon.ERROR,
                content="URL格式无效",
                target = dialog.urlLineEdit,
                parent=self,
                isClosable=True,
                tailPosition=TeachingTipTailPosition.BOTTOM,
                duration=1000
                )
                return 
            # 启动异步任务获取标题
            async def async_task():
                title = await self.fetch_rss_title(url)
                if title:
                    source_name = f"{title[:15]}..." if len(title) > 15 else title
                    self._add_rss_to_ui(source_name, url)
                    TeachingTip.create(
                        title="添加成功",
                        icon=InfoBarIcon.SUCCESS,
                        content=f"已添加订阅源: {title}",
                        target = dialog.urlLineEdit,
                        parent=self,
                        isClosable=True,
                        tailPosition=TeachingTipTailPosition.BOTTOM,
                        duration=1000
                    )
                else:
                    TeachingTip.create(
                        title="警告",
                        icon=InfoBarIcon.WARNING,
                        content="无法获取订阅源标题，将使用默认名称",
                        target = dialog.urlLineEdit,
                        parent=self,
                        isClosable=True,
                        tailPosition=TeachingTipTailPosition.BOTTOM,
                        duration=1000
                    )
            # 通过QAsync启动异步任务
            loop = QEventLoop(self)
            asyncio.set_event_loop(loop)
            asyncio.ensure_future(async_task())

    def _add_rss_to_ui(self, source_name, url):
        """将订阅源添加到UI"""
        # 防止重复添加
        current_sources = self.setting_card.config.get_config('rss_map')
        if url in current_sources.values():
            TeachingTip.create(
                title="错误",
                icon=InfoBarIcon.ERROR,
                content="该订阅源已存在",
                parent=self,
                isClosable=True,
                target =self.setting_card.add_btn,
                tailPosition=TeachingTipTailPosition.BOTTOM,
                duration=1000
            )
            return
        # 添加到数据源
        self.setting_card.interests_map[source_name] = url
        # 更新配置
        current_rss = self.setting_card.config.get_config('rss_map') or {}
        current_rss.update({source_name: url})
        self.setting_card.config.update_config('rss_map', current_rss)
        # 更新本地映射
        self.setting_card.interests_map = self.setting_card.config.get_config('rss_map')
        # 添加复选框
        cb = CheckBox(source_name)
        cb.setChecked(True)
        self.setting_card.rss_checkboxes.append(cb)  # 添加到SettingCard的复选框列表
        self.setting_card.rss_layout.addWidget(cb)  # 添加到滚动布局
        self.setting_card.adjustSize()
        self.adjustSize()
    # 筛选信息密度高的RSS
    def select_Rss(self):
        with open(resource_path("rss_data/rss_data.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data:
            return []
        
        current_data_hash = hash(json.dumps(data, sort_keys=True))
        
        if hasattr(self, 'last_pushed_hash') and current_data_hash == self.last_pushed_hash:
            print("数据未变化，返回空列表")  # 调试：提示哈希检查
            return []
        self.last_pushed_hash = current_data_hash
        
        all_entries = []
        for feed in data:
            for entry in feed["entries"]:
                try:
                    score = float(entry["score"]) if entry["score"].strip() else 0.0
                except ValueError:
                    score = 0.0
                entry["score"] = score  # 更新为数值类型
                all_entries.append(entry)

        # 按分数降序排序
        sorted_entries = sorted(all_entries, key=lambda x: x["score"], reverse=True)
         # 筛选未推送过的有效文章
        valid_entries = []
        for entry in sorted_entries:
            entry_hash = hash(f"{entry['title']}{entry['link']}")
            if entry_hash not in self.article_history:
                valid_entries.append(entry)

        # 优先选择未推送的最高分文章，若没有则选择次高分
        if valid_entries:
            selected_entry = valid_entries[0]
            self.article_history.add(hash(f"{selected_entry['title']}{selected_entry['link']}"))
        else:
            # 所有文章都已推送过，重置历史并选择当前最高分
            self.article_history.clear()
            if sorted_entries:
                selected_entry = sorted_entries[0]
                self.article_history.add(hash(f"{selected_entry['title']}{selected_entry['link']}"))
            else:
                return []
        # 只需要title，link和summary
        final_articles = [{"title": selected_entry["title"], "link": selected_entry["link"], "summary":selected_entry["summary"]}]
        return final_articles

    def close_event(self,event):
        event.ignore()
        self.hide()
        
if __name__ == '__main__':
    # 配置异步事件循环
    app = QtWidgets.QApplication([])
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    window = CrawlWindow()
    
    with loop:
        loop.run_forever()