import re
from read.document1 import HTMLDocumentParser
from read.table1 import TableParser 

class IOPParser(HTMLDocumentParser):
    """HTML Parser for IOP publications (e.g. Smart Materials, J. Phys., etc.)."""

    def __init__(self, filepath) -> None:
        super().__init__('iop_publishing', filepath)
        self.tableParser = _iopTableParser()

        # ---- Meta ----
        self.title_xpath = '//h1'
        self.journal_xpath = '//span[@itemid="periodical"]'
        self.date_xpath = '//span[@class="wd-jnl-art-pub-date"]'
        self.abstract_xpath = (
            '//section[contains(@class,"article-section__abstract")]//p | '
            '//*[contains(@class,"wd-jnl-art-abstract")]//p'
        )
        self.body_xpath = '//div[@class="article-content"]'

    # ----------------------------------------------------------------------
    def parse_meta(self):
        """解析 IOP 文章的元信息（含摘要），并返回 dict"""
        # 先让父类解析：通常会填充 self.title / self.journal / self.date / self.abstract
        super().parse_meta()
    
        # ---- Fallback: meta 标签信息 ----
        if not self.title:
            self.title = self.xpath_to_string('//meta[@name="citation_title"]/@content')
        if not self.journal:
            self.journal = self.xpath_to_string('//meta[@name="citation_journal_title"]/@content')
        if not self.date:
            self.date = self.xpath_to_string('//meta[@name="citation_publication_date"]/@content')
        if not self.abstract:
            self.abstract = self.xpath_to_string('//meta[@name="citation_abstract"]/@content')
    
        # 清洗字符串（父类里应该有，比如去掉多余空白）
        self.clean()
    
        # ✅ 关键：返回一个 dict，方便外面像 AIP 一样用
        return {
            "title": self.title or "",
            "journal": self.journal or "",
            "date": self.date or "",
            "abstract": self.abstract or ""
        }

    def parse_tables(self):
        """修复父类 parse_tables() 中的调用错误"""
        tables = self._tree.xpath('//table')
        print(f"📊 --- TABLES ---\nFound {len(tables)} tables.")
        results = []
    
        for i, table_element in enumerate(tables, 1):
            try:
                result = self.tableParser.parse(table_element)  # ✅ 正确调用
                results.append(result)
                print(f"✅ Table {i} parsed: {result['n_rows']} rows, {result['n_cols']} cols")
            except Exception as e:
                print(f"⚠️ Error parsing table {i}: {e}")
        
        return results

    # ----------------------------------------------------------------------
    def _full_table_links(self, tree) -> list:
        """IOP 无需单独下载表格链接"""
        return []

    # ----------------------------------------------------------------------
    def parse_paragraphs(self):
        """
        IOP 专用段落解析：
        - 提取 <p> 与 <span> 标签文本
        - 打印段落数量和前若干段内容
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


class _iopTableParser(TableParser):
    """解析 IOP HTML 文件中的表格及标题"""
    def __init__(self) -> None:
        super().__init__()

    def parse(self, table_element):
        """解析 IOP 表格（支持 <p><b>Table X.</b>、<figcaption> 与 <caption> 格式）"""
        result = {
            "caption": None,
            "rows": [],
            "n_rows": 0,
            "n_cols": 0
        }

        # ✅ 1. 寻找表格标题（扩展范围）
        caption_xpath = (
            './preceding-sibling::p[1]/b | '
            './preceding-sibling::p[1] | '
            './ancestor::figure/figcaption | '
            './caption | '
            './preceding-sibling::*[contains(@class,"table-caption")][1]'
        )
        captions = table_element.xpath(caption_xpath)
        if captions:
            caption_text = captions[0].text_content().strip()
            self.parse_caption_label(captions[0], label=None)
            result["caption"] = caption_text
        else:
            result["caption"] = "未找到表格标题"

        # ✅ 2. 解析表格单元格（命名空间无关）
        rows = []
        for tr in table_element.xpath(".//*[local-name()='tr']"):
            cells = []
            for td in tr.xpath("./*[local-name()='th' or local-name()='td']"):
                txt = td.text_content().strip().replace("\xa0", " ")

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

            if any(c["text"] for c in cells):  # ✅ 过滤空白行
                rows.append(cells)

        # ✅ 3. 汇总信息
        result["rows"] = rows
        result["n_rows"] = len(rows)
        result["n_cols"] = max((len(r) for r in rows), default=0)

        # ✅ 4. 生成 JSON 友好结构（嵌套列表）
        if rows:
            import pandas as pd
            df = pd.DataFrame([[c["text"] for c in r] for r in rows])
            result["data"] = df.values.tolist()
            print(f"✅ Parsed IOP table: {result['caption']} ({result['n_rows']} rows × {result['n_cols']} cols)")
            print("First Row:", df.iloc[0, :].tolist())
        else:
            print(f"⚠️ Empty or malformed table: {result['caption']}")

        return result
