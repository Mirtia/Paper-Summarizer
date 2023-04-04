from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

from converter import PDFToTextConverter


class PDFSummarizer(PDFToTextConverter):

    LANGUAGE = "english"
    NUM_SENTENCES = 20

    def __init__(self, filename) -> None:
        super().__init__(filename)
        self.summary = ""

    def summarize(self) -> None:
        stemmer = Stemmer(self.LANGUAGE)
        summarizer = Summarizer(stemmer)
        parser = PlaintextParser.from_string(self.text,
                                             Tokenizer(self.LANGUAGE))
        summarizer.stop_words = get_stop_words(self.LANGUAGE)
        for sentence in summarizer(parser.document, self.NUM_SENTENCES):
            self.summary += sentence._text

    def export(self, filename: str) -> None:
        with open(filename, mode="w", encoding="utf-8") as f:
            f.write(self.summary)