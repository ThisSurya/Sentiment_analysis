import streamlit as st
import keras
import pandas as pd
from Backend import prediction, preprocessing, scrape
import matplotlib.pyplot as plt
import seaborn as sns
def run():
    
    #     Jika input hanya 1

    # example_text = st.text_input('Input text')
    
    # clean_text = preprocessing.precleantext(example_text)
    # tokenized_text = preprocessing.padding_sequence(clean_text)
    
    # result = prediction.predict_text(tokenized_text)
    # st.write(result)
    
    # Jika input dataframe
    path = 'cleaned_text.txt'
    
    with open(path, 'r', encoding="utf8") as file:
        raw_text = file.read()
    
    data = []
    for row in raw_text.splitlines():
        if(row != ''):
            data.append(row)
        continue
    dataframe = pd.Series(data)
    
    test_sequence = preprocessing.padding_sequence(dataframe)
    
    model = keras.models.load_model('lstm_model.h5')
    y_result = model.predict(test_sequence).round()
    
    y_result = pd.DataFrame(y_result, columns=['label'])
    fig, ax = plt.subplots()
    sns.countplot(y_result, x='label')
    
    st.pyplot(fig)
    negative = y_result.loc[y_result['label'] < 0.5]
    positive = y_result.loc[y_result['label'] > 0.5]
    
    length_total = len(negative) + len(positive)
    
    positive_percent = (len(positive) / length_total) * 100
    negative_percent = (len(negative) / length_total) * 100
    
    if(positive_percent < negative_percent):
        st.write(f"Hasil data menunjukkan bahwa video tersebut lebih banyak negative komentar karena memiliki nilai sebesar: {negative_percent}")
    else:
        st.write(f"Hasil data menunjukkan bahwa video tersebut lebih banyak positive komentar karena memiliki nilai sebesar: {positive_percent}")
run()