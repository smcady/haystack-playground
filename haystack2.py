import os
import time
from urllib.parse import urlparse, urljoin
import requests
import tiktoken
import pinecone
import configparser
from bs4 import BeautifulSoup
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from haystack import Document 
from haystack.nodes import (PreProcessor, PDFToTextConverter, EmbeddingRetriever)
from langchain.embeddings.openai import OpenAIEmbeddings
from tqdm.auto import tqdm

# Read API keys and other sensitive information from environment variables
cfg = configparser.ConfigParser()
cfg.read('secrets.ini')
PINECONE_API_KEY = os.environ = cfg['PINECONE']['PINECONE_API_KEY']
PINECONE_ENV = os.environ = cfg['PINECONE']['PINECONE_ENV']
OPENAI_API_KEY = os.environ = cfg['OPENAI']['OPENAI_API_KEY']
INDEX_NAME = 'medicare'
MODEL_NAME = 'text-embedding-ada-002'



class InputDirectoryHandler(FileSystemEventHandler):
    def __init__(self, input_dir, preprocessor):
        self.input_dir = input_dir
        self.preprocessor = preprocessor

    def on_created(self, event):
        if event.is_directory:
            return
        print_debug('Directory change detected') 
        file_path = event.src_path
        if file_path.endswith(".pdf"):
            process_files(file_path, self.preprocessor)




def preprocess_url(url, preprocessor):
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
            "name": file_name,
            "Role": "Support"
        }
    )

    # DeprecationWarning: Using a single Document as argument to the 'documents' parameter is deprecated. Use a list of (a single) Document instead.
    docs = [doc]
    docs = preprocessor.process(docs)

    doc_store = initialize_index(PINECONE_API_KEY, PINECONE_ENV, 'medicare')

    embeddings = embed.embed_documents([doc.content for doc in docs])

    # Insert the embeddings into the Pinecone vector store using the document IDs as keys
    doc_store.upsert(vectors=zip([doc.id for doc in docs], embeddings))








def process_files(file_path, preprocessor):
    """
    Tokenizes the text content of the files in the input directory and saves them to a local text file with the same name as the file.
    """
    converter = PDFToTextConverter(
        remove_numeric_tables=True, valid_languages=["en"])

          
    if ((os.path.isfile(file_path)) and (file_path.endswith(".pdf"))):
        print_debug('PDF file detected.  Processing...') 
        print_debug(file_path)
        
        doc = converter.convert(file_path, meta=None)[0]
        print_debug('>>>--- Token length: ' + str(tiktoken_len(str(doc.content))))
        # create haystack document object with text content and doc metadata
        # doc = Document(content=str(doc_text), content_type="text") 

        doc.meta = {
            "name": os.path.basename(file_path),
            "Role": "Executive"
        }
        
        # DeprecationWarning: Using a single Document as argument to the 'documents' parameter is deprecated. Use a list of (a single) Document instead.
        docs = [doc]
        print_debug('>>>--- Sending to preprocessor...')  
        chunked_docs = preprocessor.process(docs)

        doc_store = initialize_index(PINECONE_API_KEY, PINECONE_ENV, 'medicare')

        # # Retriever: A Fast and simple algo to identify the most promising candidate documents
        # # OpenAI EmbeddingRetriever
        # retriever = EmbeddingRetriever(
        #    document_store=doc_store,
        #    batch_size=100,
        #    embedding_model="text-embedding-ada-002",
        #    api_key="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",
        #    max_seq_len=1536
        # )

        # generate embeddings in batches
        batch_size = 100  # how many embeddings we create and insert at once

        for i in tqdm(range(0, len(chunked_docs), batch_size)):
            # find end of batch
            i_end = min(i+batch_size, len(chunked_docs))
            # extract batch
            batch = docs[i:i_end]

            print_debug('storing doc content in pinecode')

        # doc_store.write_documents(batch) # pass in batch or will batching be handled by write_documents?
        # print_debug('Generating embeddings...')
        # doc_store.update_embeddings(retriever)
        # print_debug('storage complete')


        print_debug('>>>--- Generating embeddings...')
        embeddings = embed.embed_documents([doc.content for doc in docs])

        print_debug('Upserting embeddings to Pinecone...')
        # Insert the embeddings into the Pinecone vector store using the document IDs as keys
        doc_store.upsert(vectors=zip([doc.id for doc in docs], embeddings))
        print_debug('>>>--- Upsert complete.')


        # check that we have all vectors in index
        doc_store.describe_index_stats()

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



def print_debug(message):
    if __debug__:   
        print('>>>--- ' + message)
              

# find the length of text chunk in tokens
def tiktoken_len(text):
    # Get the encoding for the model we are using and create a tokenizer
    # tiktoken.encoding_for_model('gpt-3.5-turbo')
    tokenizer = tiktoken.get_encoding('cl100k_base')
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)


def initialize_index(PINECONE_API_KEY, PINECONE_ENV, index_name) -> pinecone.Index: 
    pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

    if index_name not in str(pinecone.list_indexes()):
    # we create a new index
        pinecone.create_index(
        name=index_name,
        metric='dotproduct',
        dimension=1536  # 1536 dim of text-embedding-ada-002
    )
        # if in debug then print the index info
        print_debug('Pinecone index created') 
            
    # connect to the index    
    index = pinecone.Index(index_name)
    # get index info
    
    print_debug('Pinecone index initialized') 
    print(index.describe_index_stats())   

    return index



def main(preprocessor):
    input_dir = "./input"

    event_handler = InputDirectoryHandler(input_dir, preprocessor)
    observer = Observer()
    observer.schedule(event_handler, input_dir, recursive=False)
    observer.start()
    print_debug('Monitoring input directory for new files...') 
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






# pipeline = ExtractiveQAPipeline(reader, retriever)

# query = "What is Medicare?"
# result = pipeline.run(query=query, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 1}})

# # create a doc_store to hold data
# document_store = ElasticsearchDocumentStore()

# # Clean & load your documents into the DocumentStore
# dicts = convert_files_to_dicts(doc_dir, clean_func=clean_wiki_text)
# document_store.write_documents(dicts)

# # Retriever: A Fast and simple algo to identify the most promising candidate documents
# # OpenAI EmbeddingRetriever
# retriever = EmbeddingRetriever(
#    document_store=doc_store,
#    batch_size=8,
#    embedding_model="text-embedding-ada-002",
#    api_key="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",
#    max_seq_len=1536
# )


# # Reader: Powerful but slower neural network trained for QA
# model_name = "deepset/roberta-base-squad2"
# reader = FARMReader(model_name)

# # Pipeline: Combines all the components
# pipe = ExtractiveQAPipeline(reader, retriever)

# # Voilà! Ask a question!
# question = "Who is the father of Sansa Stark?"
# prediction = pipe.run(query=question)
# print_answers(prediction)