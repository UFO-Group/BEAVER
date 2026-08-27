# 干净文本字符串
# 用来把 XML 中的 <paragraph> 节点 → 提取出干净的可用文本。
# 是 DocumentParser 的底层组件之一。
import re
import pylogg
from lxml import etree
import read.TextNormalizer1
from read.TextNormalizer1 import asciiText, normText, innerText

log = pylogg.New('paragraph')

class ParagraphParser(object):
    def __init__(self):
        self.text = ""
        self.body = None

    def _is_reference(self, text: str) -> bool:
        """仅排除明显是参考文献的段落"""
        if not text:
            return False
        t = text.strip().lower()
        # 只有短句且包含 doi/http 才判定为参考
        if len(t) < 120 and ("doi" in t or "http" in t or "www." in t):
            return True
        # 文献格式如 "Smith et al., 2020."
        if re.match(r'^[A-Z][a-z]+ et al\., \d{4}', text):
            return True
        return False


    def _innerText(self, element):
        """递归提取完整文本，包括子节点文本和尾巴"""
        parts = []
        if element.text:
            parts.append(TextNormalizer1.normText(element.text))
        for child in element:
            parts.append(self._innerText(child))
            if child.tail:
                parts.append(TextNormalizer1.normText(child.tail))
        return " ".join([p for p in parts if p.strip()])

    def parse(self, paragraph_element):
        """解析单个段落元素"""
        self.body = paragraph_element
        raw_text = self._innerText(paragraph_element)
        if not self._is_reference(raw_text):
            self.text = TextNormalizer1.normText(raw_text)
        else:
            self.text = ""

    def is_valid(self) -> bool:
        """Return true if the parsed paragraph is valid."""
        txt = (self.text or "").strip()
        # 过滤掉纯空行或纯链接/doi类内容
        if not txt:
            return False
        if re.match(r'^\s*(https?|doi|www\.)', txt.lower()):
            return False
        # 允许短句（如表格标题、图注）
        return len(txt.split()) >= 3
