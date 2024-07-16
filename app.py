import streamlit as st
import keras
import pandas as pd
from Backend import prediction, preprocessing, scrape, data_modification
import matplotlib.pyplot as plt
import seaborn as sns

def run():
    st.write("# Sentiment analysis Apple Vision Pro")
    path = 'cleaned_text.txt'
    
    with open(path, 'r', encoding="utf8") as file:
        raw_text = file.read()
    
    data = st.file_uploader('Harap input data .txt dahulu!')
    if data is not None:
        raw_text = data.getvalue().decode('utf-8')
        clean_text = preprocessing.precleantext(raw_text)
        raw_dataframe = pd.DataFrame(raw_text.splitlines(), columns=['Tweet'])
        data = []
        
        for row in clean_text.splitlines():
            if(row != ''):
                data.append(row)
            continue
        dataframe = pd.Series(data)
        
        test_sequence = preprocessing.padding_sequence(dataframe)
        
        model = keras.models.load_model('lstm_model.h5')
        y_result = model.predict(test_sequence).round()
        
        y_result = pd.DataFrame(y_result, columns=['label'])
        dataframe = pd.DataFrame(dataframe, columns=['tweet'])
        
        fig, ax = plt.subplots()
        sns.countplot(y_result, x='label')
        st.pyplot(fig)
        
        # Mempersiapkan untuk menggunakan wordcloud
        new_dataframe = data_modification.concatenation(dataframe, y_result)
        dataframe_negative, dataframe_positive = data_modification.split_neg_post(new_dataframe)
        wordcloud_positive = data_modification.count_word(dataframe_positive)
        wordcloud_negative = data_modification.count_word(dataframe_negative)
        
        # Display proses
        st.write("Dataset yang masih kotor")
        st.write(raw_dataframe)
        st.write("Dataset yang telah dibersihkan")
        st.write(dataframe)
        st.write("Dataset yang belum dilabel")
        st.write(new_dataframe['tweet'].head())
        st.write("Dataset yang diubah kedalam format vektor")
        st.write(test_sequence)
        st.write("Dataset yang telah diprediksi dengan model: ")
        st.write(new_dataframe.head())
        
        # Membandingkan keseluruhan apakah dataset tersebut positive atau negative
        negative = y_result.loc[y_result['label'] < 0.5]
        positive = y_result.loc[y_result['label'] > 0.5]
        
        length_total = len(negative) + len(positive)
        
        positive_percent = (len(positive) / length_total) * 100
        negative_percent = (len(negative) / length_total) * 100
        
        if(positive_percent < negative_percent):
            st.write(f"Video tersebut memiliki konten yang: ")
            st.write("# negative")
            fig_wordcloud_neg, ax_wordcloud_neg = plt.subplots()
            plt.imshow(wordcloud_negative)
            st.pyplot(fig_wordcloud_neg)
        else:
            st.write(f"Video tersebut memiliki konten yang: ")
            st.write("# positive")
            fig_wordcloud_pos, ax_wordcloud_pos = plt.subplots()
            plt.imshow(wordcloud_positive)
            st.pyplot(fig_wordcloud_pos)
run()