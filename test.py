from langchain.document_loaders import GoogleDriveLoader
import os

# loader = GoogleDriveLoader(document_ids=["1BT5apJMTUvG9_59-ceHbuZXVTJKeyyknQsz9ZNIEwQ8"],
#                           credentials_path="../../desktop_credetnaisl.json")


loader = GoogleDriveLoader(
        folder_id="1yucgL9WGgWZdM1TOuKkeghlPizuzMYb5",
        credentials_path= os.path.join(r'C:\Users\smcad\.credentials', 'credentials.json'),
        # Optional: configure whether to recursively fetch files from subfolders. Defaults to False.
        recursive=False
    )




docs = loader.load()

# %set_env OPENAI_API_KEY = ...
# Summarizing

from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain

llm = OpenAI(temperature=0, openai_api_key=os.environ['OPENAI_API_KEY'])
chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
chain.run(docs)