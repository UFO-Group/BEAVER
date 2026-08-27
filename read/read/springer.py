import re
import pandas as pd
from lxml import html
from .document1 import HTMLDocumentParser
from .table1 import TableParser


class SpringerParser(HTMLDocumentParser):
    """专用于 Springer HTML 文献格式的解析器"""
    def __init__(self, filepath) -> None:
        super().__init__('springer', filepath)
        self._tree = self.parse_html(filepath)
        self.tableParser = _springerTableParser()  # ✅ 初始化表格解析器

        # 基本路径配置
        self.title_xpath = '//header/h1'
        self.journal_xpath = '//header/p[@class="c-article-info-details"]//i'
        self.date_xpath = '//header//time'
        self.abstract_xpath = '//section[@data-title="Abstract"]//p'
        self.body_xpath = '//div[@class="c-article-body"]'

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

        # 多级容错机制（与ElsevierParser一致风格）
        if not self.title.strip():
            for xp in ['//h1', '//div[@class="ArticleTitle"]']:
                val = self.xpath_to_string(xp)
                if val:
                    self.title = val.strip()
                    break

        if not self.abstract.strip():
            for xp in [
                '//section[@data-title="Abstract"]//p',
                '//div[@id="Abs1-content"]//p',
                '//section[@class="Abstract"]//p',
                '//section[@class="abstract"]//p',
                '//div[@class="section--abstract"]//p'
            ]:
                val = self.xpath_to_string(xp)
                if val:
                    self.abstract = val.strip()
                    break

        if not self.date.strip():
            self.date = self.xpath_to_string('//time')

        if not self.journal.strip():
            for xp in [
                '//a[@data-test="ConfSeriesLink"]',
                '//div[@class="ArticleHeader main-context"]//a'
            ]:
                val = self.xpath_to_string(xp)
                if val:
                    self.journal = val.strip()
                    break

        # ✅ Body容错提取
        if not self.body.strip():
            possible_body_xpaths = [
                '//div[@class="c-article-body"]',
                '//div[contains(@class, "article-body")]',
                '//section[contains(@class, "c-article-section")]',
                '//div[@id="body"]'
            ]
            for xp in possible_body_xpaths:
                text = self.xpath_to_string(xp)
                if text.strip():
                    self.body_xpath = xp
                    self.body = text.strip()
                    print(f"✅ Body text extracted successfully from {xp}. Length: {len(self.body)} chars.\n")
                    break

            if not self.body.strip():
                print("⚠️ No body text found. Please check the HTML structure or update XPath.\n")

    # ------------------------------------------------------
    # 段落提取
    # ------------------------------------------------------
    def parse_paragraphs(self):
        self.para_xpaths = ['//p', '//span']

        paragraphs = []
        for xpath in self.para_xpaths:
            nodes = self._tree.xpath(xpath)
            for node in nodes:
                text = node.text_content().strip()
                if text:
                    paragraphs.append(text)

        if paragraphs:
            print(f"✅ Found {len(paragraphs)} paragraphs.\n")
            for i, p in enumerate(paragraphs[:100], 1):
                print(f"[{i}] {p[:150]}...\n")
        else:
            print("⚠️ No paragraphs found.")

        return paragraphs

    # ------------------------------------------------------
    # 表格提取
    # ------------------------------------------------------
    def parse_tables(self):
        """Check if tables exist and then call the table parser to parse them."""
        table_elements = self._tree.xpath('//table')
        print("找到的表格数量:", len(table_elements))

        if len(table_elements) == 0:
            print("⚠️ 没有找到任何表格。")
            return

        # 调用表格解析器逐个处理
        for i, table in enumerate(table_elements, 1):
            print(f"\n--- 开始解析第 {i} 个表格 ---")
            self.tableParser.parse(table)


class _springerTableParser(TableParser):
    """Table parser for Springer papers to extract captions and table content."""
    def __init__(self) -> None:
        super().__init__()

    def parse(self, table_element):
        """Parse the table element and extract its caption."""
        super().parse(table_element)

        print("表格解析开始:", table_element)

        result = {
            "caption": None,
            "rows": []
        }

        # ✅ 提取表格标题
        caption_xpath = './/caption | ./preceding::p[contains(text(),"Table")][1]'
        caption_nodes = table_element.xpath(caption_xpath)
        if caption_nodes:
            caption_text = caption_nodes[0].text_content().strip()
            self.parse_caption_label(caption_nodes[0], label=None)
            result["caption"] = caption_text
        else:
            result["caption"] = "未找到表格标题"

        # ✅ 解析表格内容
        rows = table_element.xpath(".//tr")
        for tr in rows:
            cells = []
            for td in tr.xpath("./th|./td"):
                text = td.text_content().strip()
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

        if not result["rows"]:
            print("⚠️ 表格为空或没有正确解析。")
            return

        try:
            self.dataframe = pd.DataFrame(result["rows"])
        except Exception as e:
            print(f"⚠️ 无法构建DataFrame: {e}")
            return

        print(f"✅ 表格行数: {len(self.dataframe)}, 列数: {self.dataframe.shape[1]}")

        # 打印表头与第二列数据
        try:
            val = self.dataframe.iloc[0, :].values
            self.header = "\n".join([str(s) for s in val])
            print("表头:", self.header)

            val = self.dataframe.iloc[:, 1].values
            print("表格第二列数据:", val)
        except IndexError as e:
            print(f"⚠️ 错误: {e}")

        return result
