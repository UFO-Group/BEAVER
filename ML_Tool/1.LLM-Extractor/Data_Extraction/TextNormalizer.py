# Text Semantic Normalization

import import_ipynb
import re
import unidecode
from lxml import etree
from textacy import preprocessing

def normalize_parentheses(text: str) -> str:
    """ Normalize and clean up parentheses and brackets by completing pairs.

        text :
            The text to clean up.
    """
    if text.count('}') == text.count('{')-1:
        text = text+'}'
    if text.count(')') == text.count('(')-1:
        # Assumes the missing closing bracket is in the end which is reasonable
        text = text+')'
    elif text.count(')') == text.count('(')+1:
        # Last ) is being removed from the list of tokens which is ok
        text = text[:-1]
    return text


def asciiText(unicode_text: str):
    """ Convert unicode text to ASCII. """
    raise RuntimeError("Please use TextNormalizer.")


def normText(text: str):
    """ Normalize a string to remove extra spaces and special characters etc."""
    norm = TextNormalizer()
    return norm.norm_chars(text)


def innerText(elem):
    """ Return the innerText of a XML element. Normalize using normText. """
    value = etree.tostring(elem, method="text", encoding="unicode")
    value = normText(value)
    return value


class TextNormalizer:
    def __init__(self):
        self.TILDES = {
            '~',  # \u007e Tilde
            '˜',  # \u02dc Small tilde
            '⁓',  # \u2053 Swung dash
            '∼',  # \u223c Tilde operator
            '∽',  # \u223d Reversed tilde
            '∿',  # \u223f Sine wave
            '〜',  # \u301c Wave dash
            '～',  # \uff5e Full-width tilde
        }
        self.SLASHES = {
            '/',  # \u002f Solidus
            '⁄',  # \u2044 Fraction slash
            '∕',  # \u2215 Division slash
        }
        self.CONTROLS = {
            r'\u0001', r'\u0002', r'\u0003', r'\u0004', r'\u0005', r'\u0006',
            r'\u0007', r'\u0008', r'\u000e', r'\u000f', r'\u0011', r'\u0012',
            r'\u0013', r'\u0014', r'\u0015', r'\u0016', r'\u0017', r'\u0018',
            r'\u0019', r'\u001a', r'\u001b',
        }
        self.replace_char_list = [r'[]', r'[,]', '()', '( )', r'[ ]', ' - ']
        self.remove_dot_list = [
            'Dr.', 'Mr.', ',Mrs.', 'et al.', 'cf.', 'viz.', 'etc.', 'Corp.',
            'Inc.', 'spp.', 'Co.', 'Ltd.', 'eg.', 'ex.',
        ]
        self.remove_all_dot_list = ['A.R.', 'A. R.', 'i.e.', 'e.g.']

        # unicode character codes
        self.degrees = [186, 730, 778, 8304, 8728,
                        9702, 9675]  
        self.to_remove = [775, 8224, 8234, 8855, 8482, 9839]

        self.formatting = [i for i in range(
            8288, 8298)] + [i for i in range(8299, 8304)] + [i for i in range(8232, 8239)]

        self.re_copyright = re.compile(
            r'© \d{4} .+', flags=re.UNICODE | re.IGNORECASE)

        self.re_consecutive_spaces = re.compile(
            r'\s+', flags=re.UNICODE | re.IGNORECASE)

    def normalize(self, text, unidec=True, chem=True, spaces=True, fig_ref=True,
              numbers=False, lower_case=False):
        # Handle plain text (removing HTML-related operations)
        
        for char_seq in self.replace_char_list:
            text = text.replace(char_seq, '')
    
        # Remove some fixed symbols
        re_str = ''.join([chr(c) for c in self.to_remove])
        re_str = r'[' + re_str + ']'
        text = re.sub(re_str, '', text)
    
        # Replace dots in abbreviations (e.g., Dr., Mr.)
        for chr_seq in self.remove_dot_list:
            text = text.replace(chr_seq, chr_seq[:-1])
    
        # Handle cases where there might be spaces between superscripts/subscripts and variables
        text = re.sub(r'\s+(\^{\w*})', r'\\1', text)
        text = re.sub(r'(\^\{1\}H)', ' \\1', text)
        text = re.sub(r'(\^\{13\}C)', ' \\1', text)
        text = re.sub(r'(\^\{7\}Li)', ' \\1', text)
        text = re.sub(r'\s+(_{\w*})', r'\\1', text)
        text = re.sub(r'(\^{\w*)\s+}', r'\\1}', text)
        text = re.sub(r'(_{\w*\s+)}', r'\\1}', text)
    
        # Remove dot symbols that might affect sentence splitting
        for chr_seq in self.remove_all_dot_list:
            chr_dotless = chr_seq.replace('.', '')
            text = text.replace(chr_seq, chr_dotless)
    
        # Normalize degree symbols
        re_str = r'[' + ''.join([chr(c) for c in self.degrees]) + ']'
        text = re.sub(re_str, chr(176), text)
        text = text.replace('° C', '°C')
        text = text.replace('°C', ' °C')
    
        # Remove copyright statements, etc.
        text = self.re_copyright.sub('', text)
    
        # Handle figures and references
        if fig_ref:
            text = re.sub(r'Fig.\s*([0-9]+)', 'Figure \\1', text)
            text = re.sub(r'[Rr]ef.\s*([0-9]+)', 'reference \\1', text)
            text = re.sub(r'\sal\.\s*[0-9\s]+', ' al ', text)
    
        # Normalize spaces
        if spaces:
            text = text.replace(r'\u000b', ' ').replace(
                r'\u000c', ' ').replace(u'\u0085', ' ')
            text = text.replace(r'\u2028', r'\n').replace(
                r'\u2029', r'\n').replace(r'\r\n', r'\n').replace(r'\r', r'\n')
            text = text.replace(r'\n', ' ')
    
            text = re.sub(r'\s+,', ',', text)
            text = re.sub(r'([1-9]),([0-9]{3})', r'\\1\\2', text)
            text = re.sub(r'(\w+)=(\d+)', r'\\1 = \\2', text)
            text = re.sub(r'\(\s+', '(', text)
            text = text.replace('%', ' %')  # Normalize percentage signs
            text = text.replace(r'\s+)', ')')
            text = re.sub(r'(\d+),([A-Za-z])', r'\\1, \\2', text)
    
        if chem:
            text = self.chem_normalizer(text)
    
        if unidec:
            pass
    
        else:
            text = preprocessing.normalize.unicode(text)
            text = preprocessing.normalize.hyphenated_words(text)
            text = preprocessing.normalize.quotation_marks(text)
            text = preprocessing.normalize.whitespace(text)
            text = preprocessing.remove.remove_accents(text)
            text = text.replace('…', '...').replace(' . . . ', ' ... ')
    
            re_str = ''.join([chr(c) for c in self.formatting])
            re_str = r'[' + re_str + ']'
            text = re.sub(re_str, '', text)
    
            for tilde in self.TILDES:
                text = text.replace(tilde, '~')
    
            for slash in self.SLASHES:
                text = text.replace(slash, '/')
    
            for control in self.CONTROLS:
                text = text.replace(control, '')
    
        if numbers:
            text = preprocessing.replace.replace_numbers(text)
            regex_fraction = r'[-]?\d+[/]\d+'
            text = re.sub(regex_fraction, ' _FRAC_ ', text)
    
        if lower_case:
            text = text.lower()
    
        text = self.re_consecutive_spaces.sub(' ', text).strip()
    
        return text


    def unidecode_normalize_string(self, string : str) -> str:
        ret_string = ''
        for char in string:
            if re.match(u'[Α-Ωα-ωÅ°±≈⋯∞∆ϵ⋅≫≡≅≃∙≠]', char) is not None:
                ret_string += str(char)
            else:
                ret_string += str(
                    unidecode.unidecode_expect_nonascii(str(char)))

        return ret_string


    def chem_normalizer(self, text : str) -> str:
        """Normalizes certain very common chemical name variations"""
        # Can probably add to this list of rules.
        text = re.sub(r'sulph', r'sulf', text, flags=re.I)
        text = re.sub(r'aluminum', r'aluminium', text, flags=re.I)
        text = re.sub(r'cesium', r'caesium', text, flags=re.I)

        return text

    def norm_chars(self, text : str) -> str:
        """ Manually normalize special characters and spaces. """
        ntext = ""
        i = 0
        while i < len(text):
            # current char
            c = text[i]
            i += 1

            if c == 'â':
                ntext += "-"
            elif c == 'Â':
                ntext += ""
            elif c == r'\x80':
                ntext += ""
            elif c == r'\x85':
                ntext += ""
            elif c == r'\x86':
                ntext += ""
            elif c == r'\x88':
                ntext += ""
            elif c == r'\x89':
                ntext += ""
            elif c == r'\x90':
                ntext += ""
            elif c == r'\x92':
                ntext += ""
            elif c == r'\x93':
                ntext += ""
            elif c == r'\x94':
                ntext += ""
            elif c == r'\x97':
                ntext += ""
            elif c == r'\x98':
                ntext += ""
            elif c == r'\x99':
                ntext += ""
            elif c == r'\x8d':
                ntext += ""
            elif c == r'\x9c':
                ntext += ""
            elif c == r'\x9d':
                ntext += ""
            elif c == r'\x96':
                ntext += ""
            elif c == r'\x8b':
                ntext += ""
            elif c == r'\xa0':
                ntext += " "
            elif c == '©':
                ntext += "(c)"
            elif c == '¼':
                ntext += "1/4"
            elif c == 'Ã':
                ntext += ""
            elif c == '®':
                ntext += "(R)"
            elif c == '¶':
                ntext += "\n"
            elif c == 'Ä':
                ntext += ""
            elif c == '²':
                ntext += "^{2}"
            elif c == 'µ':
                ntext += "\\mu"
            elif c == '′':
                ntext += "'"
            elif c == '“':
                ntext += '"'
            elif c == '‐':
                ntext += "-"
            elif c == '–':
                ntext += "-"
            elif c == '°':
                ntext += "°"
            elif c == '±':
                ntext += "+/-"
            elif c == '−':
                ntext += "-"
            elif c == r'\uf8ff':
                ntext += "-"
            elif c == '¤':
                ntext += ""
            elif c == '≈':
                ntext += "~"
            elif c == 'α':
                ntext += "\\alpha"
            elif c == '’':
                ntext += "'"
            elif c == 'Î' and self._text_char(text, i) == '¼':
                ntext += "\\mu "
                i += 1
            elif c == 'Ï' and self._text_char(text, i) == 'â':
                ntext += "pi-"
                i += 1
            elif c == 'Î' and self._text_char(text, i) == '±':
                ntext += "\\alpha"
                i += 1
            elif c == 'Ï' and self._text_char(text, i) == 'â':
                ntext += "pi-"
                i += 1
            elif c == 'Ï' and self._text_char(text, i) == 'â':
                ntext += "pi-"
                i += 1
            elif c == 'Ï' and self._text_char(text, i) == 'â':
                ntext += "pi-"
                i += 1
            elif c == 'Ï' and self._text_char(text, i) == 'â':
                ntext += "pi-"
                i += 1
            elif c == '×':
                ntext += "x"
            elif c == r'\uf8fe':
                ntext += "="
            elif c == r'\u2005':
                ntext += " "
            elif c == 'β':
                ntext += "\\beta"
            elif c == 'ζ':
                ntext += "\\zeta"
            elif c == 'ï' and self._text_char(text, i) == '£' \
                    and self._text_char(text, i+1) == '½':
                ntext += "---"
                i += 2
            elif c == 'ö':
                ntext += "o"
            elif c == 'ü':
                ntext += "u"
            # elif c == 'ö':
            #     ntext += "o"
            # ... (commented out duplicate lines retained as is)
            else:
                try:
                    c = c.encode('utf-8').decode('utf-8')
                    ntext += c
                except:
                    # If additional special characters are found,
                    # add them to the above elif cases.
                    print(text[i-50:i+50])
                    raise ValueError("Unknown Character:", ntext[-20:], c)

        # Add space before (
        ntext = ntext.replace("(", " (")

        # Remove multiple consecutive spaces.
        ntext = re.sub(r'\s+', ' ', ntext).strip()

        # Cleanup some extra spaces.
        ntext = ntext.replace(" isa ", " is a ")
        ntext = ntext.replace("( ", "(")
        ntext = ntext.replace(" )", ")")
        ntext = ntext.replace("- ", "-")
        ntext = ntext.replace(" ^", "^")
        ntext = ntext.replace(" _", "_")
        ntext = ntext.replace("}=", "} =")
        ntext = ntext.replace(" , ", ", ")
        ntext = ntext.replace("^{[}", "[")
        ntext = ntext.replace("^{]}", "]")
        ntext = ntext.replace("-\\x89", "")

        return ntext

    def _text_char(self, text: str, position: int = 1) -> str:
        try:
            return text[position]
        except IndexError:
            return ''