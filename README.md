# Paper Summarizer

This repository contains various methods to perform summarization of scientific articles. It's still on an experimental stage so don't expect it to work as it should.

## PDFToTextConverter

Reads the pdf using [**pypdf**](https://github.com/py-pdf/pypdf) and performs minimal sanitization:

- removes **pdf** annotations
- removes **URLs** and **e-mails**
- removes **-** character (hyphen)
- ignores text after **References** section

You can export the content to a **.txt** file using **export** class method.

## PDFSummarizer

Its base class is **PDFToTextConverter**. I implemented three options summarize text:

- ### NLTK + sshleifer/distilbart-cnn-12-6

First, I tokenized the text and using **frequency analysis** I found the most important sentences in the document. Then, I used [**sshleifer/distilbart-cnn-12-6**](https://huggingface.co/sshleifer/distilbart-cnn-12-6) to the target sentences (after resizing the chunks to fit the model) which is the default model for summarization tasks using the **transformers** library. Because, many words were incorrectly merged together, I used [**wordninja**](https://github.com/keredson/wordninja) which probabilistically splits concatenated words using **NLP**  to make final corrections in the document. To make the process faster I tried utilizing **concurrent** features as much as I could.

- ### Big Bird Pegasus

I chose BigBird, [**google/bigbird-pegasus-large-arxiv**](https://huggingface.co/google/bigbird-pegasus-large-arxiv), available via hugging face.
Note: It runs very slowly...

- ### sumy

There is an already existent implementation of text summarization in this [**repository**](https://github.com/miso-belica/sumy) so I simply integrated their solution.
