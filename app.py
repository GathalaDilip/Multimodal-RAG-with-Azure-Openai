import os
import uuid
import pytesseract
import pandas as pd
from pdfminer.utils import open_filename
import ssl
from unstructured.partition.pdf import partition_pdf
from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
import openai
from PIL import Image
import io
from langchain_community.chat_models import AzureChatOpenAI
from azure.core.credentials import AzureKeyCredential
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from PIL import Image as PILImage
from unstructured.documents.elements import Image
import streamlit as st
import tempfile
import uuid
import base64
from azure.search.documents.models import QueryType
import requests
from azure.storage.blob import ContentSettings
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.core.credentials import AzureKeyCredential
import streamlit as st
import tempfile
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from langchain_text_splitters import RecursiveCharacterTextSplitter
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential


from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access environment variables
azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
azure_search_api_key = os.getenv("AZURE_SEARCH_API_KEY")
azure_search_index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")

azure_blob_connection_string = os.getenv("AZURE_BLOB_CONNECTION_STRING")
azure_blob_container_name = os.getenv("AZURE_BLOB_CONTAINER_NAME")

azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_chatgpt_deployment = os.getenv("AZURE_OPENAI_CHATGPT_DEPLOYMENT")
azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
azure_openai_embedding_deployment = os.getenv("AZURE_OPENAI_CHATGPT_EMBEDDING_DEPLOYMENT")

vision_endpoint = os.getenv("VISION_ENDPOINT")
vision_api_key = os.getenv("VISION_API_KEY")





# Set the values of your computer vision endpoint and key
os.environ["VISION_ENDPOINT"] = vision_endpoint
os.environ["VISION_KEY"] =  vision_api_key

try:
    endpoint = os.environ["VISION_ENDPOINT"]
    key = os.environ["VISION_KEY"]
except KeyError:
    print("Missing environment variable 'VISION_ENDPOINT' or 'VISION_KEY'")
    exit()
    
# Initialize the client
client = ImageAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))


# Blob Storage client
blob_service_client = BlobServiceClient.from_connection_string(os.environ["AZURE_BLOB_CONNECTION_STRING"])
container_name = os.environ["AZURE_BLOB_CONTAINER_NAME"]
container_client = blob_service_client.get_container_client(container_name)

# Azure OpenAI initialization for summarization

llm = AzureChatOpenAI(
    azure_deployment="Keep your deployment name",   
    api_version="2023-03-15-preview or mention according your model",  
    temperature=0,
    max_tokens=None,
    timeout=120,
    max_retries=5,
)


# Azure Cognitive Search client setup
search_client = SearchClient(
    endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
    index_name=os.environ["AZURE_SEARCH_INDEX_NAME"],
    credential=AzureKeyCredential(os.environ["AZURE_SEARCH_API_KEY"])
)


# Initialize gpt-35-turbo and our embedding model
#embeddings = OpenAIEmbeddings(deployment_name="text-embedding-3-large", chunk_size=1)

embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-large or Keep your deployed modal name",
    azure_endpoint= azure_search_endpoint, 
    api_key= os.environ["AZURE_OPENAI_API_KEY"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"]
)

# Tesseract setup for OCR (if needed for image-based text extraction)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
#pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract.exe'

 
# Disable SSL Verification (if necessary)
ssl._create_default_https_context = ssl._create_stdlib_context


# Function to chunk text into smaller parts
def chunk_text(text, chunk_size=900):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Function to extract text from images using OCR
def ocr_from_image(image):
    return pytesseract.image_to_string(image)

# Function to extract text from the PDF file
def extract_text_from_pdf(pdf_path):
    extracted_text = ""
    
    # Open the PDF file using pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        # Extract text from PDF pages with text content
        for page in pdf.pages:
            text = page.extract_text()
            if text:  # If the page has text, add it to the extracted_text
                extracted_text += text
            else:
                # If no text is found, extract text using OCR
                images = convert_from_path(pdf_path, first_page=page.page_number, last_page=page.page_number)
                for img in images:
                    extracted_text += ocr_from_image(img)
    return extracted_text


# # Function to chunk text into smaller parts using RecursiveCharacterTextSplitter
# def chunk_text(combined_text, chunk_size=600, chunk_overlap=20):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         length_function=len,
#         is_separator_regex=False,
#     )
    
#     # Create documents (chunks) from the input text
#     text_chunks = text_splitter.create_documents(combined_text)
    
#     # Return the list of chunks
#     return text_chunks

    
def analyze_image_with_azure_vision(image_url):
    """
    Analyzes the image using Azure Computer Vision and extracts description, tags, etc.
    """
    try:
        # Perform image analysis using the client
        result = client.analyze_from_url(
            image_url=image_url,
            visual_features=[VisualFeatures.CAPTION, VisualFeatures.READ],
            gender_neutral_caption=True,
        )

        # Extract description (caption) and tags (text)
        description = result.caption.text if result.caption else "No description available."
        tags = [line.text for block in result.read.blocks for line in block.lines] if result.read else []

        return {"description": description, "tags": tags}

    except Exception as e:
        raise Exception(f"Error analyzing image: {e}")
    
    
def image_summarize(image_url, prompt):
    """
    Generates an enhanced summary for the image using LLM (Azure OpenAI).
    Returns the summary in the format: 'Summary: {summary} | Image URL: {image_url}'
    """
    try:
        # Step 1: Get vision data from Azure Computer Vision
        vision_data = analyze_image_with_azure_vision(image_url)

        # Step 2: Enhance the summary with LLM
        detailed_prompt = (
            f"{prompt}\n"
            f"Image Description: {vision_data['description']}\n"
            f"Tags: {', '.join(vision_data['tags'])}\n"
            f"Image URL: {image_url}"
        )
        response = llm.predict_messages([{"role": "user", "content": detailed_prompt}])
        summary = response.content.strip()

        # Step 3: Format the output as required
        return f"Summary: {summary} | Image URL: {image_url}"

    except Exception as e:
        # Handle errors and provide meaningful feedback
        return f"An error occurred: {e}"
    

# Function to generate image URL in Azure Blob Storage
def upload_image_to_azure(image_data, image_name):
    blob_client = container_client.get_blob_client(image_name)
    blob_client.upload_blob(image_data, overwrite=True)
    return f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{image_name}"


# Function to create an index document for image summary
def create_search_document(doc_id, content_type, content, embedding=None, summary=None):
    """
    Create a document formatted for Azure Cognitive Search, adhering to the JSON index structure.

    Args:
        doc_id (str): Unique identifier for the document.
        content_type (str): The type of content ('text', 'table', 'image').
        content (str): The main content of the document (text, table, or image summary).
        embedding (list, optional): Vector embedding for the content. Defaults to None.
        summary (str, optional): Summary of the image or other content. Defaults to None.

    Returns:
        dict: A document object formatted for Azure Cognitive Search.
    """
    # Basic structure adhering to JSON index
    document = {
        "id": doc_id,  # Unique identifier for the document
        "content_type": content_type,  # Type of the content
        "content":content ,  # Main content (text or summary)
        "embedding": embedding,  # Vector representation for similarity search
    }

    # Additional fields for images or summaries
    # if content_type == "image" and summary:
    #     document["image_summary"] = summary  # Include summary for the image
    
    return document



# Function to generate embeddings for the content
def generate_embeddings(content):
    # Using Azure OpenAI's embeddings API
    embedding = embeddings.embed_query(content)
    return embedding


# Function to add embeddings to documents
def add_embeddings_to_documents(documents):
    for doc in documents:
        # Generate embeddings for each document's content
        doc['embedding'] = generate_embeddings(doc['content'])
    return documents


# Function for Page 1 (Upload and Index PDF)
# Function to extract and upload images to Azure Blob Storage
def page1():
    st.title("Upload your PDF File") 

    uploaded_file = st.file_uploader(" ", type="pdf")
    st.write("Powered by GenAI, so surprises and mistakes are possible. Please share your feedback so we can improve")
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(uploaded_file.read())
            temp_pdf_path = temp_pdf.name
        # Extract the PDF elements (text, images, etc.)
        elements = partition_pdf(
            filename=temp_pdf_path,
            strategy="hi_res",  # High-resolution strategy
            extract_images_in_pdf=True,
            extract_image_block_types=["Image", "Table"],
            extract_image_block_to_payload=True  # Ensures images are extracted as base64 data
        )
        
         # Extract text from the uploaded PDF
        combined_text = extract_text_from_pdf(temp_pdf_path)

        # Filter elements that contain image data
        image_elements = [
            element.metadata for element in elements if hasattr(element.metadata, 'image_base64')
        ]

        # Initialize Azure Blob Storage client
        connection_string =  os.environ["AZURE_BLOB_CONNECTION_STRING"]
        container_name =   os.environ["AZURE_BLOB_CONTAINER_NAME"]
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)

        # Ensure the container exists
        if not container_client.exists():
            container_client.create_container()

        image_urls = []
        # Upload each image to Azure Blob Storage
        for i, image_metadata in enumerate(image_elements):
            image_base64 = getattr(image_metadata, 'image_base64', None)
            image_mime_type = getattr(image_metadata, 'image_mime_type', None)

            if not image_base64 or not image_mime_type:
                continue  # Skip if image data is incomplete

            # Decode the base64 image data
            image_bytes = base64.b64decode(image_base64)

            # Generate a unique blob name
            file_extension = image_mime_type.split("/")[-1]
            blob_name = f"image_{i + 1}.{file_extension}"

            # Check if the blob already exists before uploading
            blob_client = container_client.get_blob_client(blob_name)

            if blob_client.exists():
                print(f"The blob '{blob_name}' already exists. Skipping upload.")
                continue  # Skip the upload if the blob already exists
            else:
                # Set the content type for the image
                content_settings = ContentSettings(content_type=image_mime_type)
                # Upload the image to Azure Blob Storage
                blob_client.upload_blob(image_bytes, blob_type="BlockBlob",content_settings=content_settings)

                # Get the image URL after uploading
                image_url = f"https://{container_client.account_name}.blob.core.windows.net/{container_name}/{blob_name}"
                image_urls.append(image_url)
                print(f"Uploaded: {blob_name}, URL: {image_url}") 
                

        # Now you can work with other content like text and tables
        # Extract images, tables, and text
        #img = [element for element in   elements if "unstructured.documents.elements.Image" in str(type(element))]
        tab = [str(element) for element in  elements if "unstructured.documents.elements.Table" in str(type(element))]
        # narrative_text = [str(element) for element in  elements if "unstructured.documents.elements.NarrativeText" in str(type(element))]
        # title_text = [str(element) for element in elements if "unstructured.documents.elements.Title" in str(type(element))]
        # text_text = [str(element) for element in elements if "unstructured.documents.elements.Text" in str(type(element))]
        
        # # Combine all text elements into one list
        # combined_text = title_text + text_text + narrative_text
  

        # Display or process extracted images (e.g., for displaying URLs)
        #st.write("Image URLs uploaded to Azure Blob Storage:")
        # for url in image_urls:
        #     st.image(url)

        # Summarizing images and uploading to Blob Storage
        image_summaries = []
        prompt = """ You are an assistant tasked with summarizing images for retrieval. Provide a concise summary of the image based on input text from vision model.

                    Generate a single, clear paragraph starting with the word "Image" followed by the description based on the image's content. The description should be simple, concise, and informative, summarizing what is seen in the image.

                    Add word "Image" in the begining of Image summary
                
                 """
        
        for idx, element in enumerate(image_urls):
            try:
                # If 'text' attribute exists, summarize the image
                image_summary = image_summarize(element, prompt)

                # Add the image URL from Azure Blob Storage to the image summary
                image_url = image_urls[idx] if idx < len(image_urls) else "No URL available"
                image_summary_with_url = f"{image_summary} Image URL: {image_url}"
                
                # Append the image summary with URL to the list
                image_summaries.append(image_summary_with_url)

                # Generate embeddings for the image summary (excluding the URL)
                doc_id = str(uuid.uuid4())
                embedding = generate_embeddings(image_summary)

                # Create and upload document for the image summary only (with embedding and content type)
                document = create_search_document(doc_id,"image",image_summary_with_url,embedding)
                search_client.upload_documents([document])
            
            except Exception as e:
                 print(f"Error processing image {idx + 1}: {e}")
        # Now create chunks of the combined text
        text_chunks = []
        # Create documents (chunks) from the input text
        text_chunks = chunk_text(combined_text)
        # for text in combined_text:
        #     text_chunks.extend(chunk_text(text,chunk_size=1000))  # Break combined text into smaller chunk
        # Generate embeddings for text chunks and create documents
        all_documents = []
        for text_chunk in text_chunks:
            doc_id = str(uuid.uuid4())  # Generate a unique document ID
            embedding = generate_embeddings(text_chunk)  # Generate embeddings for the text chunk
            document = create_search_document(doc_id,"text", text_chunk,embedding=embedding)  # Create document
            all_documents.append(document)  # Add to the document list

        # Add table summaries to Cognitive Search with embeddings (tables don't need to be chunked)
        for tab_summary in tab:
            doc_id = str(uuid.uuid4())  # Generate a unique document ID
            embedding = generate_embeddings(tab_summary)  # Generate embeddings for the table summary
            document = create_search_document(doc_id, "table",tab_summary,embedding=embedding)  # Create document
            all_documents.append(document)  # Add to the document list

        # Upload all documents directly to Azure Cognitive Search
        search_client.upload_documents(all_documents)  # Upload prepared documents

         

# Function for Page 2 (Retrieve and Display Images)
def page2():
    st.title("DocuChat")

    with st.form('my_form'):    
        query = st.text_area("Ask me anything")
        submitted = st.form_submit_button('Submit')

        if submitted:
            
            # Search for relevant documents in Azure Cognitive Search
            # results = search_client.search(query, top=5, query_type=QueryType.SIMPLE)
            
            # relevant_content = []
            # for result in results:  # Iterate through result
            #     relevant_content.append(result)
                
            # st.write(relevant_content)
            # print(relevant_content)
            # Initialize lists for segregating relevant data
            relevant_text = []
            relevant_images = []

            # Fetch and append results
            relevant_content = []
            results = search_client.search(query, top=5, query_type=QueryType.SIMPLE)
            for result in results:
                relevant_content.append(result)
                

            # Process the relevant content
            for item in relevant_content:
                content = item.get("content", "")  # Access the content field
                content_type = item.get("content_type", "")  # Determine the content type

                # Handle text content
                if content_type == "text":
                    relevant_text.append(content)

                # Handle image content
                elif content_type == "image":
                    # Extract image URL and summary from content
                    if "Image URL:" in content:
                        image_url = content.split("Image URL:")[-1].strip()
                        image_summary = content.split("Summary:")[-1].split("|")[0].strip()
                    else:
                        image_url = ""
                        image_summary = "No summary available"

                    if image_url:  # Add only if a valid URL exists
                        relevant_images.append((image_url, image_summary))

            # # Display relevant text content in Streamlit
            # if relevant_text:
            #     st.write("Relevant Text Content:")
            #     for text in relevant_text:
            #         st.write(text)

            # Display relevant images and their summaries in Streamlit
            if relevant_images:
                for image_url, image_summary in relevant_images:
                    st.image(image_url, caption="Retrieved Image")
                    st.write("Summary of", image_summary)

            # Define the LLM prompt
            prompt = """
            You are a mechanical engineer and an expert in analyzing Bearing Manufacturing. 
            Answer the question based only on the provided context, which can include text, images, and tables. 
            If you are not sure, decline to answer and say 'Sorry, I don't have much information about it'. 
            Just return the helpful answer in as much detail as possible. 
            Answer:
            """
            
            # prompt ="""
            #         "Return the data without paraphrasing."
            #         "Do not interpret or summarize."
            #         "Provide an exact match to the text."
            #         Answer:
            #         """

            # Combine the retrieved text content and image summaries for the LLM context
            context = "\n\n".join(relevant_text) + "\n\n" + "\n".join([f"Image Summary: {img[1]}" for img in relevant_images])

            # Display the context for debugging or review
            # st.write("LLM Context:")
            # st.write(context)

            # Invoke the LLM with the query
            response = llm.invoke(
                model="gpt-4o",
                input=f"{prompt}\n{context}\nQuestion: {query}",
                max_tokens=None
            )

            # Display the LLM response
            st.write("Response:")
            st.write(response.content.strip())


            

# Initialize session state for current page
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Page 1"

# Sidebar navigation
with st.sidebar:
    if st.button("Data Ingestion"):
        st.session_state["current_page"] = "Page 1"
    if st.button("Data Retrieval"):
        st.session_state["current_page"] = "Page 2"

# Render the selected page
if st.session_state["current_page"] == "Page 1":
    page1()
elif st.session_state["current_page"] == "Page 2":
    page2()
    
    
      