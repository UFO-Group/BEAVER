# 把一篇论文（XML 或 HTML 文件）解析成结构化信息：元数据、正文、段落、表格等，便于后续清洗、检索或做 RAG。
# XML2JSON
# DocumentParser.py
import os
import re
import json
from datetime import datetime
from typing import List

from lxml import html, etree

from read.table1 import TableParser 
import read.Paragraph1
from read.TextNormalizer1 import asciiText, normText, innerText
import read.TextNormalizer1

class DocumentParser(object):
    """
    Base DocumentParser class.
    Args:
        ftype (str): XML or HTML
        publisher (str):    Name of the publisher
        filepath (str):    Path to the html document
    
    """


    def __init__(self, ftype, publisher, filepath) -> None:
        self._tree = None 

        # Paper table mapping
        self._mapping = {
            'doi': 'docname',
            'format': 'doctype',
            'date_pub': 'date',
        }

        # Document ids
        self.docpath = filepath
        self.doctype = ftype
        self.publisher = publisher
        self.docname = filepath.split("/")[-1]

        # Texts
        self.sections = {}

        # Meta
        self.title = ""
        self.date = ""
        self.date_added = datetime.now()
        self.abstract = ""
        self.journal = ""
        self.body = ""
        self.wordcount = {}

        # Parsed items
        self.tables : list[tabular.TableParser] = []
        self.paragraphs : list[Paragraph1.ParagraphParser] = []
        self.figures: list = []
        self.tablesfound = 0

    @property
    def n_tables(self):
        return len(self.tables)

    @property
    def n_figures(self):
        return len(self.figures)
    
    @property
    def length(self):
        return len(self.body)

    def serialize(self):
        js = {
            'title': self.title,
            'date': self.date,
            'abstract': self.abstract,
            'journal': self.journal,
            'publisher': self.publisher,
            'ntables': len(self.tables),
            'tables_found': self.tablesfound,
            'type': self.doctype,
            'path': self.docpath,
            'file_name': self.docname,
            'body': self.body,
        }

        return js
    
    def to_json(self, outfile):
        with open(outfile, 'w+') as fp:
            data = self.serialize()
            json.dump(data, fp, indent=4)

        print("Save OK:", outfile)

    def __repr__(self) -> str:
        pr = f"\nDocument ({self.publisher}) :: ["
        if len(self.title) > 0:
            pr += "\n\tTitle: " + self.title
        if len(self.abstract) > 0:
            pr += "\n\tAbstract: " + self.abstract
        if len(self.date) > 0:
            pr += "\n\tDate: " + self.date
        if len(self.journal) > 0:
            pr += "\n\tJournal: " + self.journal
        if len(self.body) > 0:
            pr += "\n\tLength: %d" %len(self.body)
        if self.tablesfound > 0:
            pr += "\n\tTable tags: %d" %self.tablesfound
        if len(self.tables) > 0:
            pr += "\n\tValid tables: %d" %len(self.tables)
        pr += "\n]"
        return pr

    def xpath_to_string(self, xpath):
        """ Get the element specified by XPath and return its inner text. """
        elems = self._tree.xpath(xpath)

        if len(elems) > 0:
            return innerText(elems[0])
        else:
            return ""

    def parse_meta(self):
        raise NotImplementedError()
    
    def clean(self):
        """ Run some simple text normalization. """
        self.abstract = self.abstract.lstrip('Abstract').lstrip("ABSTRACT")
        self.body = self.body.lstrip('Introduction').lstrip("INTRODUCTION")
    
    def parse_tables(self):
        raise NotImplementedError()
    
    def parse_paragraphs(self):
        """ Parse all the paragraphs from an XML document. """
        for para_xpath in self.para_xpaths:
            selected_elements = self._tree.xpath(para_xpath)

            for item in selected_elements:
                para = Paragraph1.ParagraphParser()
                para.parse(item)

                if para.is_valid():
                    self.paragraphs.append(para)

    def add_section(self, name, tag, body):
        """ Add a text section. """
        self.sections[name] = {
            'name': name, 'tag': tag, 'body': body
        }
        
    def parse(self, parse_tables=True, parse_paragraphs=True):
        """
        Parse the loaded document. This is the method that
        controls how a document is parsed.

        """
        self.parse_meta()

        if parse_tables:
            self.parse_tables()
            self.find_word_count(r'Table (\d+|[IVXLCDM]+)[\.:]*')

            # Find discussion sentences about each table.
            for tab in self.tables:
                needle = f"Table {tab.number}"
                tab.descriptions = self.find_references(needle)

            self.remove_duplicate_tables()

        if parse_paragraphs:
            self.parse_paragraphs()
            self.remove_duplicate_paragraphs()

        self.clean()


    def remove_duplicate_tables(self):
        """
        Loop over the detected tables, and remove duplicates.
        Check for duplicates using table caption.
        
        """
        tables = []
        captions = []
        for tabl in self.tables:
            if tabl.caption not in captions:
                tables.append(tabl)
                captions.append(tabl.caption)
        self.tables = tables

    def remove_duplicate_paragraphs(self):
        """
        Loop over the detected paragraphs, and remove duplicates.
        Check for duplicates using text.
        """
        paras = []
        texts = []
        for para in self.paragraphs:
            part_of_texts = [para.text in t for t in texts]
            if para.text not in texts and not any(part_of_texts):
                paras.append(para)
                texts.append(para.text)

        self.paragraphs = paras

    def find_word_count(self, word):
        """ Count the number of times a word occurs in the document. """

        doc = self.body.strip()

        if len(doc) == 0:
            fp = open(self.docpath, 'r')
            doc = fp.read()
            fp.close()

        # Search for the pattern in the document
        pattern = rf'\b{word}\b'
        match = re.findall(pattern, doc, re.IGNORECASE)

        self.wordcount[word] = len(match)
        return len(match)
    
    def find_references(self, object : str) -> list[str]:
        """
        Find the sentences in the document body referencing a table or figure.
        Case insensitive. The reference texts will be converted to ASCII.
        Args:
            object: The word(s) to search for.
                    Ex. 'Table II.', 'Figure 2' etc.

        """
        doc = asciiText(self.body.strip())

        if len(doc) == 0:
            print("Warning: document empty.")
            return []
        
        # Haystack = valid sentences
        pattern = re.compile(r'([a-z][^\.!?]*[\.!?]) ', re.M | re.IGNORECASE)
        sentenses = pattern.findall(doc)

        # Needle
        pattern = rf'\b{object}\b'

        results = []

        # Search for the needle in the sentences.
        for i, sent in enumerate(sentenses):
            match = re.search(pattern, sent, re.IGNORECASE)
            if match is not None:
                context = []

                # include the previous sentence for context
                if i > 0:
                    context.append(sentenses[i-1].strip())

                # include the matching sentence
                context.append(sent.strip())

                # include the next sentence for context
                if i < len(sentenses) - 1:
                    context.append(sentenses[i+1].strip())

                # combine everything
                results.append("\n".join(context))

        # If sentewise search fails, fallback to brute force search. 
        # This will not capture previous and next sentences.
        if len(results) == 0:
            pattern = re.compile(rf'([^.!?]*{object}[^.!?]*[.!?])', re.M | re.IGNORECASE)
            results = pattern.findall(doc)

        return results


    def errors(self):
        """
        Run basic checks after parsing and return a list of missing items.
        
        """

        pr = ""
        if len(self.title.strip()) == 0:
            pr += "- title not set\n"
        if len(self.abstract.strip()) == 0:
            pr += "- abstract not set\n"
        if len(self.body.strip()) == 0:
            pr += "- body not set\n"
        if len(self.date.strip()) == 0:
            pr += "- date not set\n"
        if len(self.journal.strip()) == 0:
            pr += "- journal not set\n"
        if len(self.paragraphs) <= 3:
            pr += "- paragraphs not set/only few set\n"

        return pr

from lxml import etree

class XMLDocumentParser(DocumentParser):
    """
    A DocumentParser to parse XML documents.
    Args:
        publisher (str):    Name of the publisher.
        filepath (str):    Path to the XML document.
    """

    def __init__(self, publisher, filepath) -> None:
        super().__init__('xml', publisher, filepath)

        # Parse XML document tree with recover=True to tolerate malformed XML
        with open(filepath, 'rb') as f:
            self._tree = etree.parse(f, parser=etree.XMLParser(recover=True))  # Add recover=True

        # Define XPath expressions for various elements
        self.table_xpath = '//*[local-name()="table"]'
        self.title_xpath = '//*[local-name()="title"]'
        self.abstract_xpath = '//*[local-name()="abstract"]'
        self.body_xpath = '//*[local-name()="body"]'
        self.date_xpath = '//*[local-name()="date"]'
        self.journal_xpath = '//*[local-name()="journal"]'
        self.para_xpaths = ['//*[local-name()="p"]']

    def parse_tables(self):
        # Tables from any XML namespace.
        tables = self._tree.xpath(self.table_xpath)

        self.tablesfound = len(tables)

        for table in tables:
            tabl = tabular.XMLTableParser()
            tabl.parse(table)
            if tabl.is_valid():
                self.tables.append(tabl)

    def parse_meta(self):
        self.title = self.xpath_to_string(self.title_xpath)
        self.abstract = self.xpath_to_string(self.abstract_xpath)
        self.date = self.xpath_to_string(self.date_xpath)
        self.body = self.xpath_to_string(self.body_xpath)
        self.journal = self.xpath_to_string(self.journal_xpath)


class HTMLDocumentParser:
    """
    A base HTML document parser.
    Args:
        publisher (str):  Publisher name.
        filepath (str):   Path to the HTML file.
    """
    def __init__(self, publisher, filepath) -> None:
        self.publisher = publisher
        self.filepath = filepath

        # 读取HTML文件并解析为HTML树
        with open(filepath, 'rb') as file:
            contents = file.read()
        self._tree = html.fromstring(contents)  # ✅ 改为 _tree，便于子类兼容调用
        self.html_tree = self._tree             # ✅ 保留旧名兼容性

        # 初始化元信息
        self.title = ""
        self.abstract = ""
        self.journal = ""
        self.body = ""
        self.date = ""

        # 默认 XPATH 配置
        self.title_xpath = '//h1'
        self.abstract_xpath = '//div[contains(@class, "abstract")]'
        self.journal_xpath = '//div[contains(@class, "journal")]'
        self.date_xpath = '//div[contains(@class, "date")]'
        self.body_xpath = '//div[contains(@class, "fulltext")]'
        self.table_xpath = '//table'
        self.para_xpaths = ['//p']

        # 表格解析器（由子类指定）
        self.tableParser = None

    # -------------------------
    # 🔹 通用工具函数
    # -------------------------
    def xpath_to_string(self, xpath):
        elements = self._tree.xpath(xpath)
        if not elements:
            return ""
        
        texts = []
        for el in elements:
            # 如果返回的是普通字符串（如属性值）
            if isinstance(el, str):
                text = el.strip()
            else:
                # 否则是 Element 节点
                text = el.text_content().strip()
            if text:
                texts.append(text)
        
        return ' '.join(texts)


    # -------------------------
    # 🔹 元信息解析
    # -------------------------
    def parse_meta(self):
        """解析文章的基本信息"""
        self.title = self.xpath_to_string(self.title_xpath)
        self.abstract = self.xpath_to_string(self.abstract_xpath)
        self.journal = self.xpath_to_string(self.journal_xpath)
        self.date = self.xpath_to_string(self.date_xpath)
        self.body = self.xpath_to_string(self.body_xpath)

    # -------------------------
    # 🔹 段落解析
    # -------------------------
    def parse_paragraphs(self):
        """返回所有段落节点"""
        paragraphs = []
        for xpath in self.para_xpaths:
            paras = self._tree.xpath(xpath)
            paragraphs.extend(paras)
        return paragraphs

    # -------------------------
    # 🔹 表格解析（新增）
    # -------------------------
    def parse_tables(self):
        """通用表格解析逻辑"""
        tables = self._tree.xpath(self.table_xpath)
        if not tables:
            print("No tables found.")
            return []

        print(f"Found {len(tables)} tables.")
        parsed_tables = []

        if not self.tableParser:
            print("⚠️ No table parser defined for this publisher.")
            return tables

        for i, table in enumerate(tables, 1):
            try:
                tp = self.tableParser()  # ✅ 调用子类表格解析器
                tp.parse(table)
                parsed_tables.append(tp)
                print(f"✅ Parsed table {i}")
            except Exception as e:
                print(f"⚠️ Error parsing table {i}: {e}")

        return parsed_tables

    def clean(self):
        """
        通用清理函数：用于去除HTML文本中的多余空格、换行符等。
        各PublisherParser可按需重写。
        """
        def _clean_text(text):
            if not text:
                return ""
            return ' '.join(text.split())  # 去掉多余空白
        
        self.title = _clean_text(self.title)
        self.abstract = _clean_text(self.abstract)
        self.journal = _clean_text(self.journal)
        self.date = _clean_text(self.date)
        self.body = _clean_text(self.body)
