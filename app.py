import streamlit as st
import anthropic
import openai
import time

st.set_page_config(page_title="LLM Text Processor", layout="wide")
st.title("🤖 LLM Text Processor")

# Initialize API clients from Streamlit Secrets
anthropic_client = None
openai_client = None

try:
    anthropic_key = st.secrets["ANTHROPIC_API_KEY"]
    anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
except KeyError:
    st.warning("Anthropic API key not found in secrets.")

try:
    openai_key = st.secrets["OPENAI_API_KEY"]
    openai_client = openai.OpenAI(api_key=openai_key)
except KeyError:
    st.warning("OpenAI API key not found in secrets.")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Model Selection")
    model_choice = st.radio(
        "Choose a model:",
        ["Claude Opus 4.1", "GPT-5-Pro"],
        help="Select which LLM to use for processing"
    )

with col2:
    st.subheader("Configuration")
    if model_choice == "Claude Opus 4.1":
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, step=0.1)
        max_tokens = st.slider("Max Tokens", 100, 100000, 100000, step=1000)
    else:  # GPT-5-Pro
        reasoning_effort = st.selectbox(
            "Reasoning Effort",
            ["low", "medium", "high"],
            help="Controls how many reasoning tokens the model generates"
        )

st.divider()

st.subheader("Input")
user_prompt = st.text_area(
    "Enter your prompt (instructions for the model):",
    height=100,
    placeholder="e.g., Summarize the following text in 3 bullet points..."
)

text_to_process = st.text_area(
    "Enter the text to process:",
    height=200,
    placeholder="Paste your text here..."
)

st.divider()

# Process button
if st.button("Process Text", type="primary", use_container_width=True):
    if not user_prompt or not text_to_process:
        st.error("Please provide both a prompt and text to process.")
    elif model_choice == "Claude Opus 4.1" and not anthropic_client:
        st.error("Please provide a valid Anthropic API key in secrets.")
    elif model_choice == "GPT-5" and not openai_client:
        st.error("Please provide a valid OpenAI API key in secrets.")
    else:
        try:
            with st.spinner(f"Processing with {model_choice}..."):
                full_prompt = f"{user_prompt}\n\n---\n\n{text_to_process}"
                
                if model_choice == "Claude Opus 4.1":
                    response = anthropic_client.messages.create(
                        model="claude-opus-4-1-20250805",
                        max_tokens=max_tokens,
                        temperature=temperature,
                        messages=[
                            {"role": "user", "content": full_prompt}
                        ]
                    )
                    result = response.content[0].text
                    model_display = "Claude Opus 4.1"
                    usage_info = f"Input tokens: {response.usage.input_tokens} | Output tokens: {response.usage.output_tokens}"
                
                else:  # GPT-5-Pro
                    response = openai_client.responses.create(
                        model="gpt-5-pro",
                        reasoning={
                            "effort": reasoning_effort
                        },
                        input=full_prompt,
                        max_output_tokens=100000
                    )
                    result = response.output_text
                    model_display = "GPT-5-Pro"
                    
                    # Extract token usage info
                    reasoning_tokens = response.usage.output_tokens_details.reasoning_tokens if hasattr(response.usage, 'output_tokens_details') else "N/A"
                    usage_info = f"Input tokens: {response.usage.input_tokens} | Reasoning tokens: {reasoning_tokens} | Output tokens: {response.usage.output_tokens}"
            
            st.success("Processing complete!")
            
            with st.expander(f"📄 Output from {model_display}", expanded=True):
                st.markdown(result)
            
            # Display usage info
            st.caption(usage_info)
        
        except anthropic.APIError as e:
            st.error(f"Anthropic API error: {str(e)}")
        except openai.APIError as e:
            st.error(f"OpenAI API error: {str(e)}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
