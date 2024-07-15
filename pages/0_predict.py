import streamlit as st
import pandas as pd
from Backend import prediction, preprocessing, scrape
import matplotlib.pyplot as plt
def run():
    st.title('Text Classification App')

    input_text = st.text_input("Enter text for classification:")

    if st.button('Predict'):
        if input_text:
            prediction = preprocessing.predict_text(input_text)
            st.write(f"Prediction: {prediction[0][0]}")
            if prediction[0][0] == 0:
                st.write("Negative")
            else:
                st.write("Positive")
    else:
        st.write("Please enter some text to classify.")
        
run()