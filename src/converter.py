import os
import pypdf
import re

class PDFToTextConverter:
    """
    A class that converts .pdf files to .txt files.

    Attributes:
        filename (str): The path to the .pdf file.
        text (str): The content of the .pdf file.
    """

    def __init__(self, filename: str) -> None:
        self.filename = self._validate_file(filename)
        self.text = self._read_file(filename)

    def _validate_file(self, filename: str) -> str:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"The file '{filename}' does not exist.")
        with open(filename, mode="rb") as f:
            if not f.read(4) == b"%PDF":
                raise ValueError(f"The file '{filename}' is not a PDF file.")
        return filename

    def _read_file(self, filename: str) -> str:
        self._validate_file(filename)
        with open(filename, mode="rb") as f:
            writer = pypdf.PdfWriter(clone_from=f)
        return " ".join(page.extract_text().replace("-", "")
                       for page in writer.pages)

    def export(self, filename: str) -> None:
        with open(filename, mode="w", encoding="utf-8") as f:
            f.write(self.text)
