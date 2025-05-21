import os
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import psycopg2
from dotenv import dotenv_values
from ollama import chat, ChatResponse
import streamlit as st
import time

# Function to extract text and images from a PDF file
def extract_content(pdf_path, output_dir, resize_width=None, resize_height=None):
    os.makedirs(output_dir, exist_ok=True)
    extracted_data = {"text": [], "images": []}
    pages = convert_from_path(pdf_path, dpi=300)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=200, length_function=len)

    for i, page in enumerate(pages):
        image_path = os.path.join(output_dir, f"page_{i + 1}.jpg")
        if resize_width or resize_height:
            if not resize_width:
                aspect_ratio = page.width / page.height
                resize_width = int(resize_height * aspect_ratio)
            elif not resize_height:
                aspect_ratio = page.height / page.width
                resize_height = int(resize_width / aspect_ratio)
            page = page.resize((resize_width, resize_height), Image.ANTIALIAS)
        page.save(image_path, "JPEG")
        extracted_data["images"].append(image_path)
        raw_text = pytesseract.image_to_string(Image.open(image_path))
        chunked_text = text_splitter.split_text(raw_text)
        extracted_data["text"].extend(chunked_text)
    return extracted_data

# Function to generate text embeddings
def generate_text_embeddings(text_list, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(text_list, convert_to_numpy=True).tolist()
    return embeddings

# Function to generate embeddings for text and images using CLIP
def generate_embeddings(inputs, model_name="openai/clip-vit-base-patch32"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)

    if isinstance(inputs[0], str) and inputs[0].endswith((".jpg", ".png", ".jpeg")):
        images = [Image.open(image_path).convert("RGB") for image_path in inputs]
        processed_inputs = processor(images=images, return_tensors="pt").to(device)
        embeddings = model.get_image_features(**processed_inputs).cpu().tolist()
    else:
        processed_inputs = processor(text=inputs, return_tensors="pt").to(device)
        embeddings = model.get_text_features(**processed_inputs).cpu().tolist()

    return embeddings

# Function to create embedding tables in PostgreSQL
def create_embedding_tables(db_config):
    connection = None
    try:
        connection = psycopg2.connect(**db_config)
        cursor = connection.cursor()
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cursor.execute("""
            CREATE TABLE text_embeddings (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                embedding VECTOR(384)
            );
        """)
        cursor.execute("""
            CREATE TABLE image_embeddings (
                id SERIAL PRIMARY KEY,
                image_url TEXT NOT NULL,
                embedding VECTOR(512)
            );
        """)
        connection.commit()
        print("Tables successfully created!")
    except Exception as e:
        print(f"Error creating tables: {e}")
    finally:
        if connection:
            cursor.close()
            connection.close()

# Function to load text embeddings into the database
def load_text_embeddings(db_config, text_chunks):
    connection = None
    try:
        embeddings = generate_text_embeddings(text_chunks)
        connection = psycopg2.connect(**db_config)
        cursor = connection.cursor()
        for text, embedding in zip(text_chunks, embeddings):
            cursor.execute(
                "INSERT INTO text_embeddings (content, embedding) VALUES (%s, %s);",
                (text, embedding)
            )
        connection.commit()
        print("Inserted text embeddings successfully into text_embeddings table!")
    except Exception as e:
        print(f"Error inserting text embeddings: {e}")
    finally:
        if connection:
            cursor.close()
            connection.close()

# Function to load image embeddings into the database
def load_image_embeddings(db_config, image_paths):
    connection = None
    try:
        embeddings = generate_embeddings(image_paths)
        connection = psycopg2.connect(**db_config)
        cursor = connection.cursor()
        for image_path, embedding in zip(image_paths, embeddings):
            cursor.execute(
                "INSERT INTO image_embeddings (image_url, embedding) VALUES (%s, %s);",
                (image_path, embedding)
            )
        connection.commit()
        print("Inserted image embeddings successfully into image_embeddings table!")
    except Exception as e:
        print(f"Error inserting image embeddings: {e}")
    finally:
        if connection:
            cursor.close()
            connection.close()

# Function to retrieve similar content from the database
def retrieve_similar_content(db_config, query, top_k=5, retrieve_mode="both"):
    connection = None
    cursor = None
    try:
        text_embedding = generate_text_embeddings([query])[0]
        image_embedding = generate_embeddings([query])[0]

        connection = psycopg2.connect(**db_config)
        cursor = connection.cursor()
        similar_content = []

        if retrieve_mode in ["text", "both"]:
            cursor.execute("""
                SELECT content, embedding <-> %s::vector AS distance
                FROM text_embeddings
                ORDER BY distance ASC
                LIMIT %s;
            """, (text_embedding, top_k))
            text_results = cursor.fetchall()
            for content, distance in text_results:
                similar_content.append({
                    "content": content,
                    "retrieval_type": "text",
                    "distance": round(distance, 4)
                })

        if retrieve_mode in ["image", "both"]:
            cursor.execute("""
                SELECT image_url, embedding <-> %s::vector AS distance
                FROM image_embeddings
                ORDER BY distance ASC
                LIMIT %s;
            """, (image_embedding, top_k))
            image_results = cursor.fetchall()
            for image_url, distance in image_results:
                similar_content.append({
                    "content": image_url,
                    "retrieval_type": "image",
                    "distance": round(distance, 4)
                })

        return similar_content
    except Exception as e:
        print(f"Error retrieving similar content: {e}")
        return []
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

# Function to create a prompt for the LLM
def create_prompt(query, db_config, top_k=5):
    retrieved_content = retrieve_similar_content(db_config, query, top_k=top_k, retrieve_mode="both")
    text_matches = [item for item in retrieved_content if item["retrieval_type"] == "text"]
    image_matches = [item for item in retrieved_content if item["retrieval_type"] == "image"]

    prompt = f"User query: {query}\n\n"
    prompt += "Here are the most relevant text results:\n"
    for text in text_matches:
        prompt += f"- {text['content']} (Similarity Score: {text['distance']})\n"
    prompt += "\nHere are the most relevant images:\n"
    for image in image_matches:
        prompt += f"- {image['content']} (Similarity Score: {image['distance']})\n"
    prompt += "\nGenerate a response based on the query and the above context."
    return prompt

# Function to generate a response using the LLM
def generate_llm_response(query, db_config, model="gemma3:4b", top_k=3, retry_attempts=3):
    prompt = create_prompt(query, db_config, top_k=top_k)
    if not prompt:
        return "Error: Failed to generate a prompt."

    for attempt in range(retry_attempts):
        try:
            response: ChatResponse = chat(model=model, messages=[{'role': 'user', 'content': prompt}])
            if response and hasattr(response, "message"):
                return response.message.content
            return "Error: Received an invalid response."
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            time.sleep(2)

    return "Error: Failed to generate response after multiple attempts."

# Streamlit UI
def inject_custom_css():
    st.markdown("""
        <style>
        .main-title {
            font-size: 30px;
            font-weight: bold;
            color: #ff6f61;
            text-align: center;
        }
        .stButton>button {
            border-radius: 8px;
            background: linear-gradient(135deg, #ff6f61, #ff4b2b);
            color: white;
            padding: 10px;
            border: none;
            font-size: 18px;
        }
        .stTextArea textarea {
            font-size: 16px;
        }
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()

st.markdown('<div class="main-title">🚀 AI-Powered Query System</div>', unsafe_allow_html=True)

query = st.text_area("Enter your query:", placeholder="Type your question here...", height=120)

# Initialize a session state to store the conversation history
if "conversation" not in st.session_state:
    st.session_state.conversation = []

if st.button("Run Query"):
    with st.spinner("Generating AI Response..."):
        db_config = dotenv_values('db_config.env')
        response = generate_llm_response(query, db_config)

        # Retrieve the closest matching image
        retrieved_content = retrieve_similar_content(db_config, query, top_k=1, retrieve_mode="image")
        closest_image = retrieved_content[0] if retrieved_content else None

        # Add the user query and LLM response to the conversation history
        st.session_state.conversation.append({"role": "user", "content": query})
        st.session_state.conversation.append({"role": "assistant", "content": response, "image": closest_image})

# Display the conversation history
for idx, message in enumerate(st.session_state.conversation):
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    elif message["role"] == "assistant":
        # Display the assistant's response in a text box
        st.markdown("**Assistant:**")
        st.text_area(
            label="",
            value=message["content"],
            height=150,
            key=f"response_{idx}",  # Use the index to ensure unique keys
            disabled=True
        )

        # Display the image in a separate block
        if message.get("image"):
            st.markdown("**Closest Matching Image:**")
            st.image(
                message["image"]["content"],
                caption=f"Similarity Score: {message['image']['distance']}",
                use_column_width=True
            )