# app.py

import streamlit as st
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------
st.set_page_config(page_title="🧮 Math Problem Solver", page_icon="🧮", layout="wide")
st.title("🧮 Math Problem Solver")

# ---------------------------------------
# 2. LOAD MODEL AND TOKENIZER (with caching)
# ---------------------------------------
@st.cache_resource
def load_model_and_tokenizer():
    MODEL_PATH = "./deepseek-math-1.3b-final-3"
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="cpu",
            torch_dtype=torch.float32  # safe for CPU
        )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

# ---------------------------------------
# 3. SOLVE MATH FUNCTION
# ---------------------------------------
def solve_math(question, model, tokenizer):
    prompt = f"### Instruction:\n{question}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.1,
        do_sample=False,
        num_beams=1,
        early_stopping=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_response.split("### Response:")[-1].strip()

    try:
        calculations = [line.split('<<')[-1].split('>>')[0] 
                        for line in response.split('\n') if '<<' in line and '>>' in line]
        for calc in calculations:
            if '=' in calc:
                expr, expected = calc.split('=')
                actual = eval(expr)
                if str(actual) != expected:
                    response = response.replace(f"<<{calc}>>", f"<<{expr}={actual}>>")
    except:
        pass

    clean_lines = []
    final_answer = None
    for line in response.split('\n'):
        line = line.strip()
        if not line:
            continue
        if not line.startswith('####'):
            clean_lines.append(line)
        elif line.startswith('####') and final_answer is None:
            answer = line.replace('####', '').strip()
            if answer.isdigit() or (answer.startswith('-') and answer[1:].isdigit()):
                final_answer = answer

    if clean_lines:
        response = '\n'.join(clean_lines)
        if final_answer is not None:
            response += f'\n#### {final_answer}'

    response = response.split('final one')[0].strip()
    return response

# ---------------------------------------
# 4. CLEAN FINAL ANSWER FROM RESPONSE
# ---------------------------------------
def clean_response(response):
    numbers = re.findall(r'\d+\.?\d*', response)
    return numbers[-1] if numbers else "No valid answer found"

# ---------------------------------------
# 5. MAIN STREAMLIT APP
# ---------------------------------------
def main():
    # Load model
    if 'model_loaded' not in st.session_state:
        with st.spinner('Loading model... (Please wait a few minutes)'):
            try:
                model, tokenizer = load_model_and_tokenizer()
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
                st.session_state.model_loaded = True
                st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Failed to load model: {str(e)}")
                st.stop()

    # Initialize chat history
    if 'history' not in st.session_state:
        st.session_state.history = []

    # --- Sidebar for history ---
    with st.sidebar:
        st.header("📚 History")
        if st.session_state.history:
            for i, item in enumerate(reversed(st.session_state.history)):
                st.markdown(f"**Q{i+1}:** {item['question']}")
                if st.button(f"View Solution {i+1}", key=f"view_{i}"):
                    st.session_state.current_response = item['response']
                    st.session_state.current_final = item['final_answer']
        else:
            st.info("No previous questions yet.")

        if st.button("🧹 Clear History"):
            st.session_state.history.clear()
            st.experimental_rerun()

    # --- Main Area ---
    st.subheader("📝 Enter your math problem:")

    with st.form(key="math_form"):
        question = st.text_area(
            label="Problem Statement:",
            value="If a train travels 300 miles in 5 hours, what is its speed?",
            height=120,
            help="Write a math problem you want to solve."
        )
        submitted = st.form_submit_button(label="🔍 Solve Problem")

    if submitted and question.strip():
        with st.spinner('Solving your problem...'):
            try:
                response = solve_math(question, st.session_state.model, st.session_state.tokenizer)
                final_answer = clean_response(response)

                # Save to history
                st.session_state.history.append({
                    "question": question,
                    "response": response,
                    "final_answer": final_answer
                })

                st.session_state.current_response = response
                st.session_state.current_final = final_answer

            except Exception as e:
                st.error(f"An error occurred while solving: {str(e)}")

    # Display current response
    if 'current_response' in st.session_state:
        st.success("✅ Problem Solved!")
        st.subheader("📋 Full Step-by-Step Solution:")
        st.code(st.session_state.current_response, language="markdown")

        st.subheader("🎯 Final Answer:")
        st.info(f"**{st.session_state.current_final}**")

if __name__ == "__main__":
    main()
