import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_PATH = "/home/llm-dev/models/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

st.title("Instruction-tuned FLAN-T5 Inference")

instruction = st.text_area("Enter your instruction:")
if st.button("Generate Response"):
    if instruction.strip():
        inputs = tokenizer(instruction, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=128)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.success(response)
    else:
        st.warning("Please enter an instruction.")