
import os
from urllib.parse import urlparse, urljoin
import pdftotext
import requests
from bs4 import BeautifulSoup
from haystack import Document
from haystack.nodes import (PreProcessor, PDFToTextConverter, TextConverter, DocxToTextConverter)
from haystack.utils import convert_files_to_docs

# converter = TextConverter(remove_numeric_tables=True, valid_languages=["en"])
# doc_txt = converter.convert(file_path="data/tutorial8/classics.txt", meta=None)[0]

# converter = DocxToTextConverter(remove_numeric_tables=False, valid_languages=["en"])
# doc_docx = converter.convert(file_path="data/tutorial8/heavy_metal.docx", meta=None)[0]

# https://www.medicare.gov/publications/10050-Medicare-and-You.pdf


def process_url(url, output_dir):
    """
    Downloads the content of the URL, preprocesses it, and saves it to a local text file with the same name as the URL.
    """
    # Parse the URL to get the file name
    parsed_url = urlparse(url)
    file_name = os.path.basename(parsed_url.path)
    if not file_name:
        file_name = parsed_url.netloc + '_' + "index.html"

    # Download the content
    response = requests.get(url)
    content = response.content

    soup = BeautifulSoup(content, "html.parser")
    url_text = soup.get_text()

    # create haystack document object with text content and doc metadata
    doc = Document(
        content=url_text,
        meta={
            "name": file_name
        }
    )

    # DeprecationWarning: Using a single Document as argument to the 'documents' parameter is deprecated. Use a list of (a single) Document instead.
    docs = [doc]
    tokenized_sentences = preprocessor.process(docs)

    #  Save the content to a local text file so we dont have to download it again
    file_path = os.path.join(output_dir, file_name)

    for i, value in enumerate(tokenized_sentences):
        with open(file_path + str(i), "w", encoding="utf-8") as f:
            f.write(str(tokenized_sentences[i].to_json()))


def process_files(input_dir):
    """
    Tokenizes the text content of the files in the input directory and saves them to a local text file with the same name as the file.
    """
    converter = PDFToTextConverter(
        remove_numeric_tables=True, valid_languages=["en"])

    for filename in os.listdir(input_dir):

        f = os.path.join(input_dir, os.path.basename(filename))

        if ((os.path.isfile(f)) and (f.endswith(".pdf"))):
            print(f)
            doc_pdf = converter.convert(file_path=f, meta=None)[0]
            docs = [doc_pdf]
            tokenized_sentences = preprocessor.process(docs)

            #  Save the content to a local text file so we dont have to process it again
            file_path = os.path.join(output_dir, filename)

            for i, value in enumerate(tokenized_sentences):
                with open(file_path + str(i), "w", encoding="utf-8") as f:
                    f.write(str(tokenized_sentences[i].to_json()))


def traverse_urls(start_url, output_dir, max_depth=1, current_depth=0):
    """
    Traverses the child URLs of the start_url up to max_depth and saves their content to a local text file with the same name as the URL.
    """
    # Process the start_url
    process_url(start_url, output_dir)

    # Traverse the child URLs up to max_depth
    if current_depth < max_depth:
        response = requests.get(start_url)
        soup = BeautifulSoup(response.text, "html.parser")
        for link in soup.find_all("a"):
            child_url = link.get("href")
            if child_url and not child_url.startswith("#") and not child_url.startswith("mailto:"):
                child_url = urljoin(start_url, child_url)
                traverse_urls(child_url, output_dir,
                              max_depth, current_depth + 1)


def main():
    # Set up the starting URL and max traversal depth
    max_depth = 1
    start_url = "https://www.clovercollab.com/"
    output_dir = "./output6"
    input_dir = "./input"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # traverse_urls(start_url, max_depth)
    process_files(input_dir)


if __name__ == "__main__":
    main()


# Initialize the PreProcessor
preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=True,
    split_by="word",
    split_length=90,
    split_overlap=6,
    split_respect_sentence_boundary=True,
)
