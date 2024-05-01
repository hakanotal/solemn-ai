import google.generativeai as genai
import streamlit as st


GOOGLE_API_KEY = "AIzaSyD7GeHoyZxCmvCkDZLtUQ6QaLL8ZKjbu6s"
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-pro')


def generate_content(prompt):
    response = model.generate_content(prompt)
    return response.text

if __name__ == '__main__':
    st.title('Gemini AI Text Generator')
    prompt = st.text_input('Enter a prompt:')
    if st.button('Generate'):
        response = generate_content(prompt)
        st.write(response)
   