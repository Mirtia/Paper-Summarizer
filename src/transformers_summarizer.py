from transformers import AutoTokenizer, BigBirdPegasusForConditionalGeneration

from converter import PDFToTextConverter


class PDFSummarizer(PDFToTextConverter):

    CHUNK_SIZE = 4096
    MAX_LENGTH = 100

    def __init__(self, filename, model="google/bigbird-pegasus-large-arxiv"):
        super().__init__(filename)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = BigBirdPegasusForConditionalGeneration.from_pretrained(
            model)

    def _split_text(self):
        self.chunks = [
            self.text[i:i + self.CHUNK_SIZE]
            for i in range(0, len(self.text), self.CHUNK_SIZE)
        ]

    def summarize(self, quiet=False):
        self._split_text()
        self.summary = ""
        for i, chunk in enumerate(self.chunks):
            if not quiet:
                print(f"Processing chunk {i + 1}/{len(self.chunks)}...")
            inputs = self.tokenizer.encode(chunk,
                                           return_tensors="pt",
                                           max_length=self.CHUNK_SIZE,
                                           truncation=True)
            summary_ids = self.model.generate(inputs,
                                              num_beams=4,
                                              max_length=self.MAX_LENGTH,
                                              early_stopping=True)
            self.summary += self.tokenizer.decode(summary_ids[0],
                                                  skip_special_tokens=True)

    def export(self, filename) -> None:
        with open(filename, mode="w", encoding="utf-8") as f:
            f.write(self.summary)