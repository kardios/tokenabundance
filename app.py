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
        ["high", "medium"],
        index=0,
        help="Controls how many reasoning tokens the model generates"
    )

with col2:
    st.subheader("Model")
    st.info("Using GPT-5-Pro")

st.divider()

st.subheader("Input")
default_prompt = """Instructions: Analyze the situation (historical event, company, policy, or system) using the following structured approach. Answer each section clearly and concisely.

1. Coordination Stack Analysis
Break down the system into four dimensions:
- Information: How knowledge, signals, and attention flow. Are there bottlenecks, feedback loops, or superior intelligence?
- Economic: How value, incentives, and resources are distributed. Who benefits? Are incentives aligned or misaligned?
- Institutional: How rules, hierarchies, and power structures shape action. Are institutions flexible or rigid?
- Cognitive: How beliefs, legitimacy, and perception evolve. How does psychology affect behavior and outcomes?
Output Example: 2–4 sentences per dimension, highlighting the key structural drivers.

2. Zero to One Insights
Identify 2–3 counterintuitive, underappreciated, or hidden truths implied by the situation. These should:
- Challenge conventional assumptions
- Reveal drivers of change that are not obvious
- Highlight leverage opportunities
Output Example: 1–2 sentences per insight.

3. Leverage Points
For each insight, identify specific points of intervention in the system where small changes could produce outsized effects. Consider:
- Rules and constraints
- Information flows
- Incentives
- Feedback loops
- Perceptions and beliefs
Output Example: 1–3 concrete interventions per insight, phrased as actionable moves.

4. Strategic Implications
Synthesize the above into clear, actionable strategy recommendations for a leader, policymaker, or creator. Include:
- Opportunities for competitive advantage or system improvement
- How to align incentives, information, and beliefs for maximum effect
- High-impact moves rather than general observations
Output Example: 3–5 bullet points of concise, actionable advice.

Tips for Use: Keep language simple and clear; explain complex ideas plainly. Focus on causality, not just description. Always highlight where small interventions have outsized impact."""

user_prompt = st.text_area(
    "Enter your prompt (instructions for the model):",
    height=150,
    value=default_prompt
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
            
            response = openai_client.beta.responses.create(
                model="gpt-5-pro",
                reasoning={
                    "effort": reasoning_effort
                },
                input=[
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
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
