import argparse
import nltk_summarizer
import converter
import transformers_summarizer
import sumy_summarizer

def main():
    parser = argparse.ArgumentParser(description='Get input .pdf')
    parser.add_argument('-f',
                        '--file',
                        type=str,
                        help='the name of the input .pdf file',
                        required=True)
    parser.add_argument('-o',
                        '--output',
                        type=str,
                        help='the name of the output file',
                        required=True)
    parser.add_argument('-m',
                        '--mode',
                        type=str,
                        help='the mode of summarization\n -pegasus\n -nltk\n -sumy\n',
                        required=True,
                        default="nltk")

    args = parser.parse_args()

    if args.mode == "nltk":
        summarizer_text = converter.PDFToTextConverter(args.file)
        summarizer_text.export(args.output)
        # summarizer_NTLK = nltk_summarizer.PDFSummarizer(args.file)
        # summarizer_NTLK.summarize()
        # summarizer_NTLK.export(args.output)
    elif args.mode == "pegasus":
        summarizer_pegasus = transformers_summarizer.PDFSummarizer(args.file)
        summarizer_pegasus.summarize()
        summarizer_pegasus.export(args.output)
    elif args.mode == "sumy":
        summarizer_sumy = sumy_summarizer.PDFSummarizer(args.file)
        summarizer_sumy.summarize()
        summarizer_sumy.export(args.output)

if __name__ == "__main__":
    main()
