import re
import pandas as pd
from read.document1 import HTMLDocumentParser
from read.table1 import TableParser
import lxml.etree as ET

class ACSParser(HTMLDocumentParser):
    """
    HTML document parser for ACS papers.
    """
    def __init__(self, filepath) -> None:
        super().__init__('acs', filepath)

        # ACS-specific XPath configurations
        self.table_xpath = '//*[local-name()="table" or local-name()="table-wrap"]'
        self.title_xpath = '//*[local-name()="span" and @class="hlFld-Title"]'  # Title extraction
        self.date_xpath = '//*[local-name()="time" and @datetime]//text()'  # For date extraction (e.g., October 21, 2016)
        self.journal_xpath = '//*[local-name()="h2" and text()="Macromolecules"]'  # Journal name extraction

    def parse_meta(self):
        # 直接处理和打印元数据
        self.title = self.xpath_to_string(self.title_xpath)
        self.journal = self.xpath_to_string(self.journal_xpath)
        self.date = self.xpath_to_string(self.date_xpath)
        
        print(f"Title: {self.title}")
        print(f"Journal: {self.journal}")
        print(f"Date: {self.date}")

    def parse_tables(self):
        """解析 ACS 表格（支持 rowspan/colspan + 命名空间）"""
        tables = self.html_tree.xpath(self.table_xpath)
        print(f"✅ Found {len(tables)} tables.")

        parsed_tables = []

        for idx, table in enumerate(tables, start=1):
            table_data = {"caption": None, "rows": [], "n_rows": 0, "n_cols": 0, "index": idx}

            # ---------- 1️⃣ 提取表格标题 ----------
            caption_xpath = (
                './preceding-sibling::div[contains(@class,"tableCaption")]'
                '| ./preceding-sibling::div[contains(@class,"caption")]'
                '| ./preceding-sibling::*[local-name()="div" and contains(text(), "Table")]'
            )
            captions = table.xpath(caption_xpath)
            if captions:
                table_data["caption"] = captions[0].text_content().strip()
            else:
                parent_caption = table.xpath('./ancestor::*[local-name()="table-wrap"]//div[contains(@class,"caption")]')
                if parent_caption:
                    table_data["caption"] = parent_caption[0].text_content().strip()

            # ---------- 2️⃣ 提取表格行列 ----------
            rows = []
            tr_elements = table.xpath('.//*[local-name()="tr"]')
            for tr in tr_elements:
                row_data = []
                cells = tr.xpath('./*[local-name()="th" or local-name()="td"]')
                for td in cells:
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

                    row_data.append({
                        "text": text,
                        "rowspan": rowspan,
                        "colspan": colspan
                    })
                if any(c["text"] for c in row_data):  # 过滤空行
                    rows.append(row_data)

            table_data["rows"] = rows
            table_data["n_rows"] = len(rows)
            table_data["n_cols"] = max((len(r) for r in rows), default=0)

            # ---------- 3️⃣ DataFrame 构建与调试输出 ----------
            if rows:
                df = pd.DataFrame([[cell["text"] for cell in r] for r in rows])
                table_data["data"] = df.values.tolist()  # ✅ 转成 JSON 可序列化结构
                table_data["columns"] = df.columns.tolist() if not df.empty else []


                print(f"\n📊 Table {idx}: {table_data['caption'] or '(No caption)'}")
                print(f"Rows: {table_data['n_rows']} | Cols: {table_data['n_cols']}")
                print(f"First Row: {df.iloc[0, :].tolist()}")
            else:
                print(f"\n⚠️ Table {idx} seems empty (no <td>/<th> detected).")

            parsed_tables.append(table_data)

        self.tables = parsed_tables
        print(f"\n✅ Parsed {len(parsed_tables)} tables successfully.")
        return parsed_tables


    def parse_paragraphs(self):
        paragraphs = []
        seen = set()

        # ========= 1️⃣ 抽象（Conspectus / Abstract）=========
        abstract_nodes = self.html_tree.xpath(
            '//div[contains(@class,"article_abstract-content")]'
            '//p[contains(@class,"articleBody_abstractText")]'
        )

        for node in abstract_nodes:
            text = " ".join(node.text_content().split())
            if text and text not in seen:
                seen.add(text)
                paragraphs.append(node)

        # ========= 2️⃣ 正文（只取 NLM_p，不取 <p>，尤其不是 p.inline）=========
        sec_nodes = self.html_tree.xpath(
            '//div[contains(@class,"article_content-left")]'
            '//div[contains(@class,"NLM_sec") and contains(@class,"NLM_sec_level_1")]'
        )

        for sec in sec_nodes:
            # Section 标题
            h2_nodes = sec.xpath('.//h2')
            title = "".join(h2_nodes[0].itertext()).strip() if h2_nodes else ""
            title_lower = title.lower()

            # 跳过 References / Key References 区域
            if "key references" in title_lower or title_lower == "references":
                continue

            # 只要 div.NLM_p，不要 p.inline
            p_nodes = sec.xpath('.//div[contains(@class,"NLM_p")]')

            for p in p_nodes:
                text = " ".join(p.text_content().split())
                if not text:
                    continue
                if text in seen:
                    continue
                seen.add(text)
                paragraphs.append(p)

        # ========= 3️⃣ 过滤 Back Matter（References、Ack 等）=========
        final = []
        for node in paragraphs:
            if not node.xpath('ancestor::*[contains(@class,"NLM_back")]'):
                final.append(node)

        print(f"✅ ACS 正文段落数量: {len(final)}\n")
        return final


        # ========= 4️⃣ 调试输出 =========
        if paragraphs:
            print(f"✅ Found {len(paragraphs)} ACS body paragraphs (abstract + sections, deduplicated).\n")
            for i, node in enumerate(paragraphs[:40], 1):
                txt = " ".join(node.text_content().split())
                print(f"[{i}] {txt[:120]}...\n")
        else:
            print("⚠️ No body paragraphs found in ACS article.\n")

        # ✅ 返回元素列表（外部继续 text_content() 即可）
        return paragraphs
