# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 15:39:34 2024

@author: P00121384
"""
##############################################################################    
#Initialization of repository and location
PROJECT_ID = "team-cookie-monsters"
REGION = "us-central1"
LOCATION = "us-central1"
BUCKET_URI = "gs://customer_voice_packets"
###############################################################################
### Importing python modules
import pandas as pd
from os import listdir
from pathlib import Path as p
from os.path import isfile, join
import streamlit as st
import os, glob, io, gc
import PIL.Image
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from google.cloud import storage
import google.auth
from pathlib import Path as p
import librosa
import pandas as pd
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from google.api_core.client_options import ClientOptions
import time
import json
from pprint import pprint
import jiwer
from io import BytesIO
from google.cloud import aiplatform
import vertexai
vertexai.init(project=PROJECT_ID, location=LOCATION)
from vertexai.generative_models import GenerationConfig, GenerativeModel, Image, Part
from vertexai.preview.vision_models import ImageGenerationModel
import google.auth
from google.cloud import storage
from vertexai.generative_models import GenerationConfig, GenerativeModel, Image, Part
import math
import matplotlib.pyplot as plt
###############################################################################
#Initializing Directory
data_folder = p.cwd() / "data"
p(data_folder).mkdir(parents=True, exist_ok=True)
transcriptions_folder = p.cwd() / "transcriptions"
p(transcriptions_folder).mkdir(parents=True, exist_ok=True)
image_folder = p.cwd() / "images"
p(image_folder).mkdir(parents=True, exist_ok=True)
###############################################################################
def upload_audio():
    # Audio Uploading Tab
    bucket_name = 'customer_voice_packets'
    # 1. Authenticate to Google Cloud
    credentials, project = google.auth.default()
    # 2. Create a storage client
    storage_client = storage.Client(project=project)
    # 3. Get a reference to the bucket (check existence)
    bucket = storage_client.bucket(bucket_name)
    if not bucket:
        raise ValueError(f"Bucket '{bucket_name}' not found")
    # 4. Handle uploaded file
    with st.form("my-form", clear_on_submit=True):
        uploaded_file = st.file_uploader("Choose a audio file:", type=[".wav",".wave",".flac",".mp3"], accept_multiple_files=False)
        submitted = st.form_submit_button("SUBMIT")
    filename = ""
    print(uploaded_file)
    if uploaded_file is not None:
        # Read audio file:
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes, format='audio/wav')
        # 5. Generate unique filename
        filename = f"{uploaded_file.name}"
        # 6. Create blob and upload the file
        blob = bucket.blob(filename)
        blob.upload_from_string(audio_bytes, content_type=uploaded_file.type)
        # Write on local
        with open("data/" + filename, mode="wb") as f:
            f.write(audio_bytes)        
        # Print success message
        st.success(f"File uploaded successfully: {filename}")
    else:
        st.info("Please upload a WAV file.")
    return filename
###############################################################################
def speech_to_text(audio_path, filename):
    if filename != "":
        data_folder = p.cwd() / "data"
        #long_audio_duration = librosa.get_duration(path=audio_path)  
        client = SpeechClient(client_options=ClientOptions(api_endpoint=f"{REGION}-speech.googleapis.com"))
        language_code = "en-US"
        recognizer_id = f"chirp-{language_code.lower()}-test"
        recognizer_request = cloud_speech.GetRecognizerRequest(
            name= f"projects/{PROJECT_ID}/locations/{REGION}/recognizers/{recognizer_id}"
        )
        get_operation = client.get_recognizer(request=recognizer_request)
        recognizer = get_operation
        long_audio_config = cloud_speech.RecognitionConfig(
        features=cloud_speech.RecognitionFeatures(
            enable_automatic_punctuation=True, enable_word_time_offsets=True
        ),
        auto_decoding_config={},
        )
        long_audio_uri = "gs://customer_voice_packets/"+filename
        long_audio_request = cloud_speech.BatchRecognizeRequest(
            recognizer=recognizer.name,
            recognition_output_config={
                "gcs_output_config": {"uri": f"{BUCKET_URI}/transcriptions"}
            },
            files=[{"config": long_audio_config, "uri": long_audio_uri}],
        )
        long_audio_operation = client.batch_recognize(request=long_audio_request)
        long_audio_result = long_audio_operation.result()
        transcriptions_uri = long_audio_result.results[long_audio_uri].uri
        transcriptions_file_path = str(data_folder / "transcriptions.text")        
        # Audio Uploading Tab
        bucket_name = 'customer_voice_packets'
        # 1. Authenticate to Google Cloud
        credentials, project = google.auth.default()
        # 2. Create a storage client
        storage_client = storage.Client(project=project)
        # 3. Get a reference to the bucket (check existence)
        bucket = storage_client.bucket(bucket_name)
        # Get the blob (file) object
        transcriptions_uri_regex = transcriptions_uri.replace("gs://customer_voice_packets/","")
        blob = bucket.blob(transcriptions_uri_regex) 
        # Download data to local file
        #with open(transcriptions_file_path, "wb") as local_file:
        #    blob.download_as_string()
        blob.download_to_filename(transcriptions_file_path)
        transcriptions = json.loads(open(transcriptions_file_path, "r").read())
        transcriptions = transcriptions["results"]
        transcriptions = [
            transcription["alternatives"][0]["transcript"]
            for transcription in transcriptions
            if "alternatives" in transcription.keys()
        ]
        long_audio_transcription = " ".join(transcriptions)
    else:
        long_audio_transcription = ""
    return long_audio_transcription
###############################################################################
def summary(transcription):
    if transcription!="":
        model = GenerativeModel("gemini-1.0-pro")
        prompt = """You are an expert customer care service agent with good amount of knowledge 
        in interior designing and civil works. You understand the decorative area space and try to build
        beautiful homes for the customers. You should pay attention to detail. Capture all the customer requirements and 
        customer personal details. You should clearly look for the details shared by customer. 
        It may be split due to spaces in between. You should remove the spaces and consider them as a single word. 
        You should decode the text as clearly as possible.
        You should also look if the mobile number or email address is split due to spaces in the text. 
        You should validate the customer email address, customer pin code and customer phone number regex before providing summary.
        If you don't get a 10 digit number for mobile number then say not provided. 
        If you don't get a 6 digit number for pincode then say not provided.
        You should check that the pincode will be of 6 digits and mobile number would be of 10 digits.
        You should also let us know if we can cross sell our product based on the transcription. The cross selling can be done for
        painting, interior design, modular kitchen, bath fittings, furnishing or home decor. You should cross sell only the best fit.
        Your task is to understand the context of the message and give a summary of it. 
        The message is as follows:"""
        final_prompt = prompt + transcription
        final_response = ""
        responses = model.generate_content(final_prompt, stream=True)
    
        for response in responses:
            final_response = final_response + response.text
            #print(response.text, end="")
        
        final_response = final_response.replace("\n"," ")
        final_response = final_response.replace("*"," ")
    else:
        final_response=""
    return final_response
###############################################################################
def img_input(transcription):
    if transcription!='':
        model = GenerativeModel("gemini-1.0-pro")
        prompt = """You are an expert bot. You should get only the relevant work related to interior design, decor,
        bath fittings, modular kitchen and furnishings. You should discard all personal details, appointment details 
        from the text.
        The message is as follows.""" 
        prompt = prompt + transcription
        image_response = ""
        responses = model.generate_content(prompt, stream=True)
        for response in responses:
            image_response = image_response + response.text
            print(response.text, end="")
        image_response = image_response.replace("\n"," ")
        image_response = image_response.replace("*"," ")
    else:
        image_response = ""
    return image_response
###############################################################################
# An axuillary function to display images in grid
def display_images_in_grid(images,file_name):
    """Displays the provided images in a grid format. 4 images per row.

    Args:
        images: A list of PIL Image objects representing the images to display.
    """

    # Determine the number of rows and columns for the grid layout.
    nrows = math.ceil(len(images) / 4)  # Display at most 4 images per row
    ncols = min(len(images) + 1, 4)  # Adjust columns based on the number of images

    # Create a figure and axes for the grid layout.
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 6))

    for i, ax in enumerate(axes.flat):
        if i < len(images):
            # Display the image in the current axis.
            ax.imshow(images[i]._pil_image)

            # Adjust the axis aspect ratio to maintain image proportions.
            ax.set_aspect("equal")

            # Disable axis ticks for a cleaner appearance.
            ax.set_xticks([])
            ax.set_yticks([])
            
            #filename = f"image_{i}.jpg"
            image = images[i]._pil_image.convert('RGB')
            filename = file_name+"image"+str(i)+".jpg"
            # 4. Construct the full path to the destination file
            target_file = os.path.join("images/", filename)  # Adjust the new filename if needed
            # 5. Save the image to the target folder
            image.save(target_file)            
        else:
            # Hide empty subplots to avoid displaying blank axes.
            ax.axis("off")

###############################################################################
def image_creation(image_prompt,file_name):
    if image_prompt!="":
        # prompt = """ You need to strictly restrict images to decorative business like interior design, paints, kitchen products and bathroom products. 
        # Do not generate images for other things. You need to find elements related to decor space in that text and generate images.
        # The message is as follows:"""
        prompt = """You are an interior designing bot with expertise in generating photo-realistic images 
        of living spaces. You are conversant in multiple design language. You are trained on different themes, 
        color schemes, design styles, lighting preferences, and other design elements. 
        """
        prompt = prompt + image_prompt
        generation_model = ImageGenerationModel.from_pretrained("imagegeneration@005")
        response = generation_model.generate_images(
            prompt=prompt,
            number_of_images=4,
        )
        display_images_in_grid(response.images,file_name)
    else:
        print("Error!")
###############################################################################
### Run the application
if __name__ == "__main__":
    ########################## Page configuration #############################
    fp = open("./streamlit/AP.jpg","rb")
    image = PIL.Image.open(fp)
    st.set_page_config(
        page_title="BHS Bot",
        page_icon=image,
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'About': "## Upload the audio file and we will give you the summary and AI generated images of your requirement\n **Contact for updates** : Team Asian Paints"
        })
    ############################ Main Function ################################
    files = glob.glob('data/*')
    for f in files:
        os.remove(f)
    files = glob.glob('transcriptions/*')
    for f in files:
        os.remove(f)
    files = glob.glob('images/*')
    for f in files:
        os.remove(f)
    ############################ Main Function ################################
    # Display the Banner Image
    image = PIL.Image.open("streamlit/ap_bhs.png")
    new_image = image.resize((500, 150))
    st.image(new_image)
    #st.title("Beautiful Home Services - Asian Paints")
    st.markdown("---")
    # Audio Uploading Tab
    bucket_name = 'customer_voice_packets'
    # 1. Authenticate to Google Cloud
    credentials, project = google.auth.default()
    # 2. Create a storage client
    storage_client = storage.Client(project=project)
    # 3. Get a reference to the bucket (check existence)
    bucket = storage_client.bucket(bucket_name)
    if not bucket:
        raise ValueError(f"Bucket '{bucket_name}' not found")
    # 4. Handle uploaded file
    with st.form("my-form", clear_on_submit=True):
        uploaded_file = st.file_uploader("Choose a audio file:", type=[".wav",".wave",".flac",".mp3"], accept_multiple_files=False)
        submitted = st.form_submit_button("SUBMIT")
    filename = ""
    if uploaded_file is not None:
        # Read audio file:
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes, format='audio/wav')
        # 5. Generate unique filename
        filename = f"{uploaded_file.name}"
        # 6. Create blob and upload the file
        blob = bucket.blob(filename)
        blob.upload_from_string(audio_bytes, content_type=uploaded_file.type)
        # Write on local
        with open("data/" + filename, mode="wb") as f:
            f.write(audio_bytes)        
        # Print success message
        st.success(f"File uploaded successfully: {filename}")
        long_audio_transcription = speech_to_text("data/"+filename, filename)
        audio_summary = summary(long_audio_transcription)
        img_prompt = img_input(audio_summary)
        image_creation(img_prompt,filename)
        st.markdown("---")
        st.header("Below is the Transcription of the call with a customer:"+"\n", divider='rainbow')
        st.write(long_audio_transcription+"\n")
        st.markdown("---")
        st.header("Below is the summary of the requirement of the customer:"+"\n", divider='rainbow')
        st.write(audio_summary+"\n")
        st.markdown("---")
        st.header("Below is the A.I. generated interior spaces of a requirement:"+"\n", divider='rainbow')
        list_of_img = glob.glob("images/*")
        st.image(list_of_img, use_column_width=True)
        st.markdown("---")
    else:
        st.info("Please upload a WAV file.")
###############################################################################
