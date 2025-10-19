import streamlit as st
import openai
import time
import fitz  # PyMuPDF

st.set_page_config(page_title="LLM Text Processor", layout="wide")
st.title("🤖 LLM Text Processor")

# Initialize API client from Streamlit Secrets
openai_client = None

try:
    openai_key = st.secrets["OPENAI_API_KEY"]
    openai_client = openai.OpenAI(api_key=openai_key)
except KeyError:
    st.warning("OpenAI API key not found in secrets.")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Configuration")
    reasoning_effort = st.selectbox(
        "Reasoning Effort",
        ["low", "medium", "high"],
        help="Controls how many reasoning tokens the model generates"
    )

with col2:
    st.subheader("Model")
    st.info("Using GPT-5-Pro")

st.divider()

st.subheader("Input")
user_prompt = st.text_area(
    "Enter your prompt (instructions for the model):",
    height=100,
    placeholder="e.g., Summarize this PDF, extract key points, analyze the document..."
)

uploaded_file = st.file_uploader(
    "Upload a PDF file to process:",
    type="pdf",
    help="Select a PDF file to extract text from"
)

st.divider()

# Process button
if st.button("Process PDF", type="primary", use_container_width=True):
    if not user_prompt:
        st.error("Please provide a prompt.")
    elif not uploaded_file:
        st.error("Please upload a PDF file.")
    elif not openai_client:
        st.error("Please provide a valid OpenAI API key in secrets.")
    else:
        start_time = time.time()
        timer_placeholder = st.empty()
        
        try:
            # Extract text from PDF
            pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            pdf_text = ""
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                pdf_text += page.get_text()
            
            pdf_document.close()
            
            # Combine prompt with extracted text
            full_prompt = f"{user_prompt}\n\n---\n\n{pdf_text}"
            
            with timer_placeholder.container():
                st.info("⏱️ Processing...")
            
            response = openai_client.responses.create(
                model="gpt-5-pro",
                reasoning={
                    "effort": reasoning_effort
                },
                input=full_prompt,
                max_output_tokens=100000
            )
            
            result = response.output_text
            
            # Extract token usage info
            reasoning_tokens = response.usage.output_tokens_details.reasoning_tokens if hasattr(response.usage, 'output_tokens_details') else "N/A"
            usage_info = f"Input tokens: {response.usage.input_tokens} | Reasoning tokens: {reasoning_tokens} | Output tokens: {response.usage.output_tokens}"
            
            elapsed_time = time.time() - start_time
            timer_placeholder.success(f"✅ Completed in {elapsed_time:.2f} seconds")
            
            with st.expander("📄 Output from GPT-5-Pro", expanded=True):
                st.markdown(result)
            
            # Display usage info
            st.caption(usage_info)
        
        except openai.APIError as e:
            timer_placeholder.error(f"❌ Error after {time.time() - start_time:.2f}s")
            st.error(f"OpenAI API error: {str(e)}")
        except Exception as e:
            timer_placeholder.error(f"❌ Error after {time.time() - start_time:.2f}s")
            st.error(f"An error occurred: {str(e)}")
