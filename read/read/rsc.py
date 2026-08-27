import re
from .document1 import HTMLDocumentParser
from .table1 import TableParser

class RSCParser(HTMLDocumentParser):
    def __init__(self, filepath) -> None:
        super().__init__('rsc', filepath)
        self.tableParser = _rscTableParser()
        self.table_xpath = '//table[contains(@class, "tgroup") or contains(@class, "rtable")]'
        self.body_xpath = '//div[@id="articlebody" or @id="wrapper" or contains(@class,"article__body")]'
        self.abstract_xpath = '//div[@id="abstract" or contains(@class,"abstract")]//p'
        self.title_xpath = '//meta[@name="citation_title"]/@content | //h1'

        # 🔴 关键：补上期刊和日期的 XPath
        # 「From the journal: Journal of Materials Chemistry A」
        self.journal_xpath = '//*[contains(text(),"From the journal")]'

        # 「First published 18 Mar 2013」
        self.date_xpath = '//*[contains(text(),"First published")]'


    def parse_meta(self):
        """解析标题、摘要、期刊、日期（RSC 专用）"""

        # 先让父类跑一遍，拿到一个基础 meta
        meta = super().parse_meta() or {}

        # ========= 1) 标题 =========
        if not getattr(self, "title", None):
            title = self.xpath_to_string('//meta[@name="citation_title"]/@content')
            if not title:
                title = self.xpath_to_string('//h1')
            self.title = (title or "").strip()

        # ========= 2) 摘要 =========
        if not getattr(self, "abstract", None):
            abstract = self.xpath_to_string(
                '//meta[@name="citation_abstract"]/@content'
            )
            if not abstract:
                abstract = self.xpath_to_string(
                    '//div[@id="abstract" or contains(@class,"abstract")]//p'
                )
            self.abstract = (abstract or "").strip()

        # ========= 3) 期刊（重点修这里） =========
        # 1）先尝试 meta（新一点的 RSC 页面可能有）
        journal = self.xpath_to_string(
            '//meta[@name="citation_journal_title"]/@content'
        )

        # 2）没有 meta 的话 → 用你贴出来的这段结构兜底：
        #    <span class="italic"><a title="Link to journal home page">J. Mater. Chem. A</a></span>
        if not journal:
            journal = self.xpath_to_string(
                '//span[@class="italic"]/a[@title="Link to journal home page"]/text()'
            )
        # 再兜底一层（有的页面没有 title 属性）
        if not journal:
            journal = self.xpath_to_string(
                '//span[@class="italic"]/a/text()'
            )

        self.journal = (journal or "").strip()

        # ========= 4) 日期 =========
        # 1）先尝试 meta
        date = self.xpath_to_string(
            '//meta[@name="citation_publication_date"]/@content | '
            '//meta[@name="citation_date"]/@content | '
            '//meta[@name="citation_online_date"]/@content'
        )

        # 2）没有 meta 的话 → 利用你给出的那段：
        #    <span class="italic">J. Mater. Chem. A</span>, 2013, <strong>1</strong>, ...
        if not date:
            year_text = self.xpath_to_string(
                'normalize-space(//span[@class="italic"]/following-sibling::text()[1])'
            )
            # year_text 一般类似 ", 2013, 1,"，用正则抓年份
            m = re.search(r'(19|20)\d{2}', year_text or "")
            if m:
                date = m.group(0)
            else:
                date = year_text  # 实在不行就全塞进去

        self.date = (date or "").strip()

        # ========= 5) 清洗 + 写回 =========
        try:
            self.clean()  # 如果父类有 clean() 就顺便用一下
        except Exception:
            pass

        meta["title"] = getattr(self, "title", "") or meta.get("title", "")
        meta["journal"] = getattr(self, "journal", "") or meta.get("journal", "")
        meta["date"] = getattr(self, "date", "") or meta.get("date", "")
        meta["abstract"] = getattr(self, "abstract", "") or meta.get("abstract", "")

        # # 调试输出（先保留几次看看，OK 后可以注释掉）
        # print("📘 Title:", meta["title"])
        # print("📗 Journal:", meta["journal"])
        # print("📅 Date:", meta["date"])

        return meta

    def parse_paragraphs(self):
        """
        RSC 专用段落解析（统一结构化逻辑）
        - 提取 <p> 与 <span> 标签文本
        - 支持控制台打印段落数量和示例
        - 返回空列表而非 None
        """
        # ✅ 设置 XPath 列表
        self.para_xpaths = [
            '//p',
            '//span'
        ]
    
        # ✅ 调用父类解析逻辑（会遍历 para_xpaths）
        paragraphs = super().parse_paragraphs()
    
        # ✅ 控制台打印信息
        if paragraphs:
            print(f"✅ Found {len(paragraphs)} paragraphs.\n")
            for i, para in enumerate(paragraphs, 1):
                text = para.text_content().strip()
                # if text:
                #     print(f"[{i}] {text}\n")
        else:
            print("No paragraphs found.\n")
    
        # ✅ 容错返回空列表而不是 None
        return paragraphs if paragraphs is not None else []

class _rscTableParser(TableParser):
    def __init__(self) -> None:
        super().__init__()

    def parse(self, table_element):
        """解析RSC表格，提取caption、rowspan、colspan，并与统一逻辑对齐"""
        result = {
            "caption": None,
            "rows": [],
            "n_rows": 0,
            "n_cols": 0
        }

        # ✅ 提取caption（支持div或caption标签）
        caption_xpath = (
            './ancestor::div/preceding-sibling::div[@class="table_caption"][1] | '
            './preceding-sibling::div[@class="table_caption"][1] | '
            './caption'
        )
        captions = table_element.xpath(caption_xpath)
        if captions:
            caption_text = captions[0].text_content().strip()
            self.parse_caption_label(captions[0], label=None)
            result["caption"] = caption_text
        else:
            result["caption"] = "未找到表格标题"

        # ✅ 提取表格行列
        rows = []
        for tr in table_element.xpath(".//*[local-name()='tr']"):
            cells = []
            for td in tr.xpath("./*[local-name()='th' or local-name()='td']"):
                txt = td.text_content().strip().replace("\xa0", " ")

                # 安全整数解析
                def safe_int(val, default=1):
                    try:
                        return int(float(re.findall(r"[\d.]+", val)[0]))
                    except Exception:
                        return default

                rowspan = safe_int(td.get("rowspan", "1"))
                colspan = safe_int(td.get("colspan", "1"))

                cells.append({
                    "text": txt,
                    "rowspan": rowspan,
                    "colspan": colspan
                })

            if any(c["text"] for c in cells):  # ✅ 过滤空行
                rows.append(cells)

        result["rows"] = rows
        result["n_rows"] = len(rows)
        result["n_cols"] = max((len(r) for r in rows), default=0)

        # ✅ 结构化预览（与 Elsevier / AIP 一致）
        if rows:
            import pandas as pd
            df = pd.DataFrame([[c["text"] for c in r] for r in rows])
            result["data"] = df.values.tolist()
            print(f"✅ Parsed table with {result['n_rows']} rows, {result['n_cols']} cols")
            print("First Row:", df.iloc[0, :].tolist())
        else:
            print("⚠️ 表格为空或未识别到有效单元格")

        return result
