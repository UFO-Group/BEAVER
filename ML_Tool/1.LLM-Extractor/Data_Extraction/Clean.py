import pylogg
from TextNormalizer import TextNormalizer, normText
from lxml import etree

# Initialize log (can be commented out as needed)
log = pylogg.New('paragraph')

class ParagraphParser(object):
    def __init__(self, name=None, debug=False):
        self.name = name
        self.text = None
        self.body = None
        self.debug = debug
        self.normalizer = TextNormalizer()  # Initialize TextNormalizer instance

    def _log(self, message):
        if self.debug:
            print(f"[{self.name or 'Unnamed'}] {message}")

    def _is_reference(self, text: str) -> bool:
        """
        Determine if it is a reference more accurately 
        (only checks short text containing keywords).
        """
        if text is None:
            return False
        text_lower = text.lower()
        
        # Modification: Only classify as reference if text contains DOI or URL and is short
        if any(kw in text_lower for kw in ["http", "doi"]) and len(text) < 150:
            if len(text.split()) < 10:  # Further refinement: If it is a very short text and contains DOI, consider it a reference
                return True
        return False

    def _clean_text(self, text: str) -> str:
        """Clean up a text string using both normText() and TextNormalizer()."""
        if text is None:
            return ""
        
        if self._is_reference(text):
            return ""  # Skip references directly
        
        # Step 1: Perform structural and semantic normalization
        text = self.normalizer.normalize(text)
        # Step 2: Perform basic character cleaning
        text = normText(text)
    
        return text

    def parse(self, paragraph_text: str):
        """Parse plain text paragraph."""
        self.body = paragraph_text
        self.text = self._clean_text(paragraph_text)
        if self.debug:
            if not self.text.strip():
                print("Warning: Cleaned text is empty!")

    def is_valid(self) -> bool:
        text = self.text.strip()
        
        # ✅ Check both English and Chinese punctuation
        contains_punctuation = any(p in text for p in "。！？.?!")
        
        # ✅ Slightly increase length and word count requirements to avoid misclassifying phrases
        is_long_enough = len(text) > 30
        num_words = len(text.split())
    
        return contains_punctuation and is_long_enough and num_words > 10


    def save(self, outfile):
        with open(outfile, "w+", encoding="utf-8") as fp:
            fp.write(self.text or "")
            fp.write("\n")
        print("Save OK:", outfile)