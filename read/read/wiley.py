import re
import pandas as pd
from lxml import html
from .document1 import HTMLDocumentParser
from .table1 import TableParser


class WileyParser(HTMLDocumentParser):
    """专用于 Wiley HTML 文献格式的解析器"""
    def __init__(self, filepath) -> None:
        super().__init__('wiley', filepath)
        self._tree = self.parse_html(filepath)
        self.tableParser = _wileyTableParser()  # ✅ 初始化表格解析器

        # 基本路径配置
        self.abstract_xpath = '//div[@class="abstract-group"]'
        self.date_xpath     = '//span[@class="epub-date"]'
        self.body_xpath     = '//section[@class="article-section article-section__full"]'
        self.journal_xpath  = '//div[@class="journal-banner-text"]'

    # ------------------------------------------------------
    # HTML解析
    # ------------------------------------------------------
    def parse_html(self, filepath):
        with open(filepath, 'rb') as f:
            contents = f.read()
        return html.fromstring(contents)

    # ------------------------------------------------------
    # 元信息提取
    # ------------------------------------------------------
    def parse_meta(self):
        super().parse_meta()

        # 期刊和标题
        if not self.journal.strip():
            self.journal_xpath = '//h1'
            self.journal = self.xpath_to_string(self.journal_xpath)
            self.title_xpath = '//h2'
            self.title = self.xpath_to_string(self.title_xpath)

        # 摘要容错提取
        if not self.abstract.strip():
            self.abstract_xpath = '//section[@class="article-section article-section__abstract"]//p'
            self.abstract = self.xpath_to_string(self.abstract_xpath)

        # 正文与日期
        self.body = self.xpath_to_string(self.body_xpath)
        self.date = self.xpath_to_string(self.date_xpath)

        # # ✅ 输出日志信息（统一格式）
        # print(f"📘 Title: {self.title}")
        # print(f"📗 Journal: {self.journal}")
        # print(f"📅 Date: {self.date}")
        # print(f"📄 Abstract length: {len(self.abstract)} chars")
        # print(f"📚 Body length: {len(self.body)} chars\n")

        return {
            "title": self.title,
            "journal": self.journal,
            "abstract": self.abstract,
            "body": self.body,
            "date": self.date
        }

    # ------------------------------------------------------
    # 段落提取
    # ------------------------------------------------------
    def parse_paragraphs(self):
        self.para_xpaths = [
            '//article//p',
            '//section[@class="article-section__content"]//p'
        ]
    
        paragraphs = []
        for xpath in self.para_xpaths:
            nodes = self._tree.xpath(xpath)
            for node in nodes:
                text = node.text_content().strip()
                if text:
                    paragraphs.append(text)
    
        # if paragraphs:
        #     print(f"✅ Found {len(paragraphs)} paragraphs.\n")
        #     for i, p in enumerate(paragraphs[:100], 1):  # 已经是文本了，不需要再调用 text_content
        #         print(f"[{i}] {p[:150]}...\n")
        # else:
        #     print("⚠️ No paragraphs found.")
        
        return paragraphs


    # ------------------------------------------------------
    # 表格提取
    # ------------------------------------------------------
    def parse_tables(self):
        """解析 Wiley HTML 中的表格"""
        table_elements = self._tree.xpath('//table')
        print("找到的表格数量:", len(table_elements))

        if len(table_elements) == 0:
            print("⚠️ 没有找到任何表格。")
            return []

        table_data = []
        for i, table in enumerate(table_elements, 1):
            print(f"\n--- 开始解析第 {i} 个表格 ---")
            parsed = self.tableParser.parse(table)
            table_data.append(parsed)
        return table_data


class _wileyTableParser(TableParser):
    """Table parser for Wiley papers (保持原始逻辑 + 统一输出风格)"""
    def __init__(self) -> None:
        super().__init__()

    def parse(self, table_element):
        """解析 Wiley HTML 表格为结构化 JSON 格式"""
        print("表格解析开始:", table_element)

        result = {
            "caption": None,
            "rows": []
        }

        # ✅ 提取表格标题
        caption_xpath = './ancestor::div[@class="article-table-content"]/header[@class="article-table-caption"]'
        captions = table_element.xpath(caption_xpath)
        if captions:
            caption_text = captions[0].text_content().strip()
            self.parse_caption_label(captions[0], label=None)
            result["caption"] = caption_text
        else:
            result["caption"] = "未找到表格标题"

        # ✅ 解析表格行列
        rows = table_element.xpath(".//tr")
        for tr in rows:
            cells = []
            for td in tr.xpath("./th | ./td"):
                text = td.text_content().strip().replace("\xa0", " ")
                rowspan = td.get("rowspan", 1)
                colspan = td.get("colspan", 1)
                try:
                    rowspan = int(rowspan)
                except (TypeError, ValueError):
                    rowspan = 1
                try:
                    colspan = int(colspan)
                except (TypeError, ValueError):
                    colspan = 1
                cells.append({
                    "text": text,
                    "rowspan": rowspan,
                    "colspan": colspan
                })
            result["rows"].append(cells)

        # ✅ 打印结构信息
        n_rows = len(result["rows"])
        n_cols = max((len(r) for r in result["rows"]), default=0)
        print(f"📊 表格标题: {result['caption']}")
        print(f"✅ 表格行数: {n_rows}, 列数: {n_cols}")

        if n_rows > 0:
            print(f"预览首行内容: {result['rows'][0]}")

        # ✅ 保存DataFrame尝试（用于后处理）
        try:
            df = pd.DataFrame([[cell["text"] for cell in row] for row in result["rows"]])
            self.dataframe = df
        except Exception as e:
            print("⚠️ 无法构建DataFrame:", e)

        return result
