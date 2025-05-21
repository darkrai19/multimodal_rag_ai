Local MultiModal RAG: Text and Images

Here are some examples of what it can do:
-Organizing Information: Quickly sort through documents that include both written content and visuals, like reports or scanned files.

-Support Learning: Help students and educators by retrieving relevant text and images to create engaging materials or presentations.

-Creative Thinking: Assist with brainstorming ideas by sourcing text and images that inspire new concepts.

Specifications:

-Sample Document: https://ir.amd.com/financial-information/financial-results

-Document Parser: PyTesseract (text and image from pdf)

-Chunking: RecursiveCharacterTextSplitter (chunks: 400, overlap: 200)

-Embedding Model: ColQWen2, miniLM-l6-v2, CLIP

-Vector Database: pgvector

-MultiModal Model: gemma3:4b

-UI: Streamlit
