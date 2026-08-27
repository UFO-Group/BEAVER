import re
import pandas as pd
from read.document1 import HTMLDocumentParser
from read.table1 import TableParser


class AIPParser(HTMLDocumentParser):
    """HTML document parser for AIP (American Institute of Physics) papers."""
    def __init__(self, filepath) -> None:
        super().__init__('aip', filepath)
        self.tableParser = _aipTableParser  # 实例化类传递给 TableParser

        # ---- Meta XPaths ----
        self.title_xpath = '//h1'
        self.journal_xpath = '//div[@class="header-journal-title"]/a'
        self.date_xpath = '//div[contains(@class, "publicationContentEpubDate")]'
        self.abstract_xpath = '//div[contains(@class, "abstractSection")]/div'
        self.body_xpath = '//div[@class="hlFld-Fulltext"]'

    # ----------------------------------------------------------------------
    def _full_table_links(self, tree):
        """获取全文中可能的补充数据或表格链接"""
        return tree.xpath('//a[contains(@href, "table") or contains(@href, "suppl")]')

    # ----------------------------------------------------------------------
    def parse_meta(self):
        """解析 AIP 文章的标题、期刊、摘要、日期"""
        # 先让父类做一遍基础解析，通常会填 self.title / self.journal / self.abstract / self.date
        base_meta = super().parse_meta() or {}

        print("\n🧩 --- META INFO ---")
        print("[DEBUG] Primary XPath hits:")
        for name, xpath in {
            "title": self.title_xpath,
            "journal": self.journal_xpath,
            "abstract": self.abstract_xpath,
            "date": self.date_xpath,
        }.items():
            try:
                print(f"  {name}: {len(self._tree.xpath(xpath))} nodes")
            except Exception:
                print(f"  {name}: XPath error")

        # ---- Abstract fallback ----
        if len(self.abstract.strip()) <= 12:
            for xpath in [
                '//div[@class="hlFld-Abstract"]',
                '//section[@id="abstract"]/p',
                '//meta[@name="description"]/@content'
            ]:
                text = self.xpath_to_string(xpath)
                if text.strip():
                    self.abstract = text
                    break

        # ---- Pub date fallback ----
        # ---- Pub date fallback ----
        if not self.date.strip():
            for xpath in [
                # 1️⃣ 你原来的主 XPath
                '//div[contains(@class,"publicationContentEpubDate")]',

                # 2️⃣ 常见 AIP meta 标签
                '//meta[@name="citation_online_date"]/@content',
                '//meta[@name="citation_publication_date"]/@content',
                '//meta[@name="citation_date"]/@content',

                # 3️⃣ DC/PRISM 风格的日期
                '//meta[@name="dc.Date"]/@content',
                '//meta[@name="DC.date"]/@content',
                '//meta[@name="dc.date"]/@content',

                # 4️⃣ 各种 epub date span/p
                '//span[contains(@class,"epubdate")]',
                '//span[contains(@class,"epub-date")]',
                '//p[contains(@class,"epub-date")]',  # <p class="epub-date">

                # 5️⃣ 文本里包含 "Published online"
                '//p[contains(text(),"Published online")]',

                # 6️⃣ 兜底：带 Published 且有年份的 <p>
                '//p[contains(text(),"Published") and contains(text(),"20")]',
            ]:
                text = self.xpath_to_string(xpath)
                if text.strip():
                    self.date = text.strip()
                    break

        # 7️⃣ 最后再兜底：任何 name 里带 date 的 meta
        if not self.date.strip():
            text = self.xpath_to_string(
                '//meta[contains(translate(@name,"DATE","date"),"date")]/@content'
            )
            if text.strip():
                self.date = text.strip()

        # ---- Fallback for title / journal via <meta> tags ----
        if not self.title.strip():
            text = self.xpath_to_string('//meta[@name="citation_title"]/@content')
            if text.strip():
                self.title = text

        if not self.journal.strip():
            text = self.xpath_to_string('//meta[@name="citation_journal_title"]/@content')
            if text.strip():
                self.journal = text

        # 清洗文本
        self.clean()

        # ✅ 最终统一返回 dict（关键点）
        meta = {
            "title": self.title,
            "journal": self.journal,
            "abstract": self.abstract,
            "date": self.date,
        }
        # 把父类解析到的其它字段也合进去（如果有的话）
        base_meta.update(meta)
        return base_meta


    # ----------------------------------------------------------------------
    def parse_paragraphs(self):
        """解析正文段落"""
        self.para_xpaths = [
            '//div[@class="NLM_paragraph"]',
            '//div[contains(@class,"hlFld-Fulltext")]//p',
            '//div[contains(@class,"section")]//p'
        ]
        paragraphs = super().parse_paragraphs()

        # 打印结果
        if paragraphs:
            print(f"✅ Found {len(paragraphs)} paragraphs.\n")
            for i, para in enumerate(paragraphs, 1):
                text = para.text_content().strip()
                # if text:
                #     print(f"[{i}] {text}\n")
        else:
            print("No paragraphs found.\n")

        return paragraphs if paragraphs is not None else []

    # ----------------------------------------------------------------------
    def parse_tables(self):
        """解析表格和表格链接"""
        print("\n📊 --- TABLES ---")
        tables = self._tree.xpath('//table')
        if not tables:
            print("No <table> elements found.")
            return []
    
        print(f"✅ Found {len(tables)} HTML tables.")
        parsed_tables = []
        for i, table in enumerate(tables, 1):
            try:
                tp = self.tableParser()
                data = tp.parse(table)
                parsed_tables.append(data)
                print(f"✅ Parsed table {i}: {data['caption'] or 'No caption'} (rows: {len(data['rows'])})")
            except Exception as e:
                print(f"⚠️ Error parsing table {i}: {e}")
    
        print(f"✅ Parsed {len(parsed_tables)} tables.")
        return parsed_tables


class _aipTableParser(TableParser):
    """增强版 AIP HTML 表格解析器：支持 caption、aria-label、隐藏graphic-wrap、嵌套脚注、rowspan、colspan"""
    def __init__(self) -> None:
        super().__init__()

    def parse(self, table_element):
        result = {"caption": None, "rows": [], "footnote": None}

        # 🧩 Step 1. 找到 table 外层 wrap
        wrap = table_element.xpath('./ancestor::div[contains(@class,"table-wrap")][1]')
        wrap = wrap[0] if wrap else None

        # 🧩 Step 2. 提取 caption
        caption_text = None

        # 2.1 <caption>
        captions = table_element.xpath('./caption')
        if captions:
            caption_text = captions[0].text_content().strip()

        # 2.2 <div class="table-wrap-head">
        if not caption_text and wrap is not None:
            pre_caption = wrap.xpath('./preceding-sibling::div[contains(@class,"table-wrap-head")]')
            if pre_caption:
                caption_text = pre_caption[0].text_content().strip()

        # 2.3 ✅ <div class="graphic-wrap hide"> 中的 aria-label
        if not caption_text:
            aria_links = table_element.xpath(
                './following-sibling::div[contains(@class,"graphic-wrap") or contains(@class,"graphic-wrap hide")]//a[@aria-label]'
            )
            if not aria_links and wrap is not None:
                aria_links = wrap.xpath(
                    './following-sibling::div[contains(@class,"graphic-wrap") or contains(@class,"graphic-wrap hide")]//a[@aria-label]'
                )
            for a in aria_links:
                label = a.attrib.get("aria-label", "")
                if "TABLE" in label.upper():
                    caption_text = label.strip()
                    break

        # 2.4 <strong>、<span class="tableLabel"> 等
        if not caption_text:
            possible_caps = table_element.xpath('.//span[contains(@class,"tableLabel")] | .//strong | .//b')
            for node in possible_caps:
                text = node.text_content().strip()
                if re.match(r'(?i)table\s*\d+', text):
                    caption_text = text
                    break

        # 2.5 首行 <th>
        if not caption_text:
            first_th = table_element.xpath('.//tr[1]/th[1]')
            if first_th:
                text = first_th[0].text_content().strip()
                if re.match(r'(?i)table\s*\d+', text):
                    caption_text = text

        if caption_text:
            caption_text = re.sub(r'\s+', ' ', caption_text)
            result["caption"] = caption_text
        else:
            print(f"⚠️ Warning: No caption found for table (rows ≈ {len(table_element.xpath('.//tr'))})")

        # 🧩 Step 3. 提取表格行，处理 rowspan 和 colspan
        for tr in table_element.xpath(".//tr"):
            row = []
            cells = tr.xpath("./th|./td")
            for cell in cells:
                text = re.sub(r'\s+', ' ', cell.text_content().strip())
                rowspan = int(cell.get("rowspan", 1))
                colspan = int(cell.get("colspan", 1))

                row.append({
                    "text": text,
                    "rowspan": rowspan,
                    "colspan": colspan
                })
            result["rows"].append(row)

        # 🧩 Step 4. 提取脚注（兼容多层结构）
        fn_xpath_candidates = [
            './following-sibling::div[contains(@class,"table-wrap-foot")]//p',
            './following-sibling::div[contains(@class,"table-wrap-foot")]//div[contains(@class,"fn")]//p',
        ]
        if wrap is not None:
            for path in fn_xpath_candidates:
                footnotes = wrap.xpath(path)
                if footnotes:
                    fn_text = " ".join(fn.text_content().strip() for fn in footnotes)
                    fn_text = re.sub(r'\s+', ' ', fn_text)
                    result["footnote"] = fn_text
                    break

        # 🧩 Step 5. 调试输出
        print("\n[DEBUG] AIP Table Parsed:")
        print(f"Caption: {result['caption'] or '(none)'}")
        print(f"Rows: {len(result['rows'])} | Footnote: {bool(result['footnote'])}")
        if result["footnote"]:
            print(f"Footnote: {result['footnote'][:100]}...")
        else:
            print("No footnote found.")
        print("Preview first row:", result["rows"][0] if result["rows"] else "(empty table)")
        print("-" * 80)

        return result
