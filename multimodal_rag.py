import os
import torch
import numpy as np
import psycopg2
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter
from colpali_engine.models import ColQwen2, ColQwen2Processor
from colpali_engine.compression.token_pooling import HierarchicalTokenPooler
from transformers.utils.import_utils import is_flash_attn_2_available
from dotenv import dotenv_values
import streamlit as st
from ollama import chat
from ollama import ChatResponse
import atexit

# Load database configuration
db_config = dotenv_values('db_config.env')
os.environ["OLLAMA_FLASH_ATTENTION"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OLLAMA_GPU_OVERHEAD"] = "1"

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
def generate_text_embeddings(text_list, model_name="vidore/colqwen2-v1.0", pool_factor=3):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ColQwen2.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
    ).eval().to(device)
    processor = ColQwen2Processor.from_pretrained(model_name, use_fast=True)
    batch = processor.process_queries(text_list).to(model.device)
    with torch.no_grad():
        embeddings = model(**batch).cpu().float().numpy()
    pooled_embeddings = np.mean(embeddings, axis=1).tolist()
    return pooled_embeddings

# Function to generate image embeddings
def generate_image_embeddings(image_paths, model_name="vidore/colqwen2-v1.0"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ColQwen2.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
    ).eval().to(device)
    processor = ColQwen2Processor.from_pretrained(model_name, use_fast=True)
    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
    processed_images = processor.process_images(images).to(device)
    with torch.no_grad():
        embeddings = model(**processed_images).cpu().float().numpy()
    pooled_embeddings = np.mean(embeddings, axis=1).tolist()
    return pooled_embeddings

# Function to create the database table
def create_table(db_config):
    try:
        connection = psycopg2.connect(**db_config)
        cursor = connection.cursor()
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id SERIAL PRIMARY KEY,
                content TEXT,
                is_image BOOLEAN,
                embedding VECTOR(128)
            );
        """)
        connection.commit()
    except Exception as e:
        print(f"Error creating table: {e}")
    finally:
        if connection:
            cursor.close()
            connection.close()

# Function to insert text embeddings into the database
def load_text_embeddings(db_config, text_chunks):
    try:
        embeddings = generate_text_embeddings(text_chunks)
        connection = psycopg2.connect(**db_config)
        cursor = connection.cursor()
        for text, embedding in zip(text_chunks, embeddings):
            cursor.execute(
                "INSERT INTO embeddings (content, is_image, embedding) VALUES (%s, %s, %s);",
                (text, False, embedding)
            )
        connection.commit()
    except Exception as e:
        print(f"Error inserting text embeddings: {e}")
    finally:
        if connection:
            cursor.close()
            connection.close()

# Function to insert image embeddings into the database
def load_image_embeddings(db_config, image_paths):
    try:
        embeddings = generate_image_embeddings(image_paths)
        connection = psycopg2.connect(**db_config)
        cursor = connection.cursor()
        for image_path, embedding in zip(image_paths, embeddings):
            cursor.execute(
                "INSERT INTO embeddings (content, is_image, embedding) VALUES (%s, %s, %s);",
                (image_path, True, embedding)
            )
        connection.commit()
    except Exception as e:
        print(f"Error inserting image embeddings: {e}")
    finally:
        if connection:
            cursor.close()
            connection.close()

# Initialize the database connection in session state
if "db_connection" not in st.session_state:
    try:
        st.session_state.db_connection = psycopg2.connect(**db_config)
        st.session_state.db_cursor = st.session_state.db_connection.cursor()
        st.success("Database connection established.")
    except Exception as e:
        st.error(f"Error connecting to the database: {str(e)}")

# Close the database connection when the app is closed
def close_db_connection():
    if "db_connection" in st.session_state:
        try:
            st.session_state.db_cursor.close()
            st.session_state.db_connection.close()
            print("Database connection closed.")
        except Exception as e:
            print(f"Error closing the database connection: {str(e)}")

# Register the close_db_connection function to run when the app stops
atexit.register(close_db_connection)

# Function to retrieve similar content from the database
def retrieve_similar_content(query_embedding, top_k=5, include_images="False"):
    try:
        cursor = st.session_state.db_cursor
        query_embedding = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        similar_content = []
        if include_images == "False":
            cursor.execute("""
                SELECT content, is_image, embedding <-> %s::vector AS distance
                FROM embeddings
                WHERE is_image = False
                ORDER BY distance ASC
                LIMIT %s;
            """, (query_embedding, top_k))
        elif include_images == "True":
            cursor.execute("""
                SELECT content, is_image, embedding <-> %s::vector AS distance
                FROM embeddings
                WHERE is_image = True
                ORDER BY distance ASC
                LIMIT %s;
            """, (query_embedding, top_k))
        elif include_images == "Both":
            cursor.execute("""
                SELECT content, is_image, embedding <-> %s::vector AS distance
                FROM embeddings
                ORDER BY distance ASC
                LIMIT %s;
            """, (query_embedding, top_k))
        results = cursor.fetchall()
        for content, is_image, distance in results:
            similar_content.append({"content": content, "is_image": is_image, "distance": round(distance, 4)})
        return similar_content
    except Exception as e:
        st.error(f"Error retrieving similar content: {str(e)}")
        return []

# Function to create a prompt for the LLM
def create_prompt(query, db_config, top_k=5):
    query_embedding = generate_text_embeddings([query])[0]
    retrieved_content = retrieve_similar_content(query_embedding, top_k=top_k, include_images="Both")
    text_matches = [item for item in retrieved_content if not item["is_image"]]
    image_matches = [item for item in retrieved_content if item["is_image"]]

    prompt = f"User query: {query}\n\n"
    prompt += "Here are the most relevant text results:\n"
    for text in text_matches:
        prompt += f"- {str(text['content'])} (Similarity Score: {text['distance']})\n"
    prompt += "\nHere are the most relevant images:\n"
    for image in image_matches:
        prompt += f"- {str(image['content'])} (Similarity Score: {image['distance']})\n"
    prompt += "\nGenerate a response based on the query and the above context."
    return prompt

# Function to generate a response using the LLM
def generate_llm_response(query, db_config, model="gemma3:4b", top_k=5):
    try:
        prompt = create_prompt(query, db_config, top_k=top_k)
        response: ChatResponse = chat(model=model, messages=[{'role': 'user', 'content': prompt}])
        return response.message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Streamlit UI
st.title("Multimodal RAG: Text and Image Retrieval")

# Tab for uploading PDF
uploaded_pdf = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

if uploaded_pdf:
    output_dir = "output"
    extracted_content = extract_content(uploaded_pdf, output_dir)
    st.sidebar.success("PDF content extracted successfully!")

# Tab for querying
query = st.text_input("Your Query", placeholder="Type your question here...")

# Initialize a session state to store the conversation history
if "conversation" not in st.session_state:
    st.session_state.conversation = []

if st.button("Generate Results"):
    if query:
        try:
            # Generate LLM response
            response_text = generate_llm_response(query, db_config, model="gemma3:4b", top_k=3)

            # Retrieve the closest matching image
            query_embedding = generate_text_embeddings([query])[0]
            retrieved_content = retrieve_similar_content(query_embedding, top_k=1, include_images="True")
            closest_image_path = retrieved_content[0]["content"] if retrieved_content else None

            # Add the user query and LLM response to the conversation history
            st.session_state.conversation.append({"role": "user", "content": query})
            st.session_state.conversation.append({"role": "assistant", "content": response_text})

            # Display the conversation
            for message in st.session_state.conversation:
                if message["role"] == "user":
                    st.chat_message("user").markdown(f"**You:** {message['content']}")
                elif message["role"] == "assistant":
                    st.chat_message("assistant").markdown(f"**Assistant:** {message['content']}")

            # Display the closest matching image (if available)
            if closest_image_path:
                st.subheader("Closest Matching Image")
                st.image(closest_image_path, caption="Closest Matching Image", use_column_width=True)
            else:
                st.warning("No matching image found.")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a query.")