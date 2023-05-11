import os
import time
from urllib.parse import urlparse, urljoin
import requests
import tiktoken
import pinecone
from bs4 import BeautifulSoup
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from haystack import Document
from haystack.nodes import (PreProcessor, PDFToTextConverter)


# find API key in console at app.pinecone.io
PINECONE_API_KEY = ''
# find ENV (cloud region) next to API key in console
PINECONE_ENV = 'us-west1-gcp'

INDEX_NAME = 'medicare'



class InputDirectoryHandler(FileSystemEventHandler):
    def __init__(self, input_dir, output_dir, preprocessor):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.preprocessor = preprocessor

    def on_created(self, event):
        if event.is_directory:
            return

        file_path = event.src_path
        if file_path.endswith(".pdf"):
            preprocess_files(file_path, self.output_dir, self.preprocessor)


def preprocess_url(url, output_dir, preprocessor):
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


def preprocess_files(file_path, output_dir, preprocessor):
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
    preprocess_url(start_url, output_dir)

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


# find the length of text chunk in tokens
def tiktoken_len(text):
    # Get the encoding for the model we are using and create a tokenizer
    tokenizer = tiktoken.get_encoding(tiktoken.encoding_for_model('gpt-3.5-turbo'))
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)


def initialize_index(PINECONE_API_KEY, PINECONE_ENV, index_name) -> pinecone.Index: 
    index_name = index_name
    pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

    if index_name not in pinecone.list_indexes():
    # we create a new index
        pinecone.create_index(
        name=index_name,
        metric='dotproduct',
        dimension=1536  # 1536 dim of text-embedding-ada-002
    )
    # connect to the index    
    index = pinecone.GRPCIndex(index_name)
    # get index info
    # print(index.describe_index_stats())
    return index


doc_store = initialize_index(PINECONE_API_KEY, PINECONE_ENV, 'medicare')


# upsert the data document to pinecone index
document_store.write_documents(docs)





def main(preprocessor):
    # ...
    # Set up the starting URL and max traversal depth
    output_dir = "./output6"
    input_dir = "./input"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    event_handler = InputDirectoryHandler(input_dir, output_dir, preprocessor)
    observer = Observer()
    observer.schedule(event_handler, input_dir, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()












if __name__ == "__main__":
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

    main(preprocessor)
