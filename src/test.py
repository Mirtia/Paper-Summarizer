import argparse
import nltk_summarizer
import transformers_summarizer
import converter

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
                        help='the mode of summarization\n -pegasus\n -nltk\n',
                        required=False,
                        default="ntlk")

    args = parser.parse_args()


    converter_PDF = converter.PDFToTextConverter(args.file)
    if args.mode == "ntlk":
        summarizer_NTLK = nltk_summarizer.PDFSummarizer(args.file)
        summarizer_NTLK.summarize()
        summarizer_NTLK.export(args.output)
    elif args.mode == "pegasus":
        summarizer_pegasus = transformers_summarizer.PDFSummarizer(args.file)
        summarizer_pegasus.summarize()
        summarizer_pegasus.export(args.output)


if __name__ == "__main__":
    main()
