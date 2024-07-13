import os
import re
import json
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from sklearn.model_selection import train_test_split
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
import tensorflow
import pandas as pd
from tensorflow.keras.layers import SpatialDropout1D
import keras

# Load the model
# Meliputi emotikon, duplikat data, URLS, tag
def remove_dirty(text):
    
    # Ekspresi reguler untuk mencocokkan karakter emotikon
    emoticon_pattern = re.compile("["
                                  u"\U0001F600-\U0001F64F"  # emoticon umum
                                  u"\U0001F300-\U0001F5FF"  # simbol & emoticon lainnya
                                  u"\U0001F680-\U0001F6FF"  # emoticon transportasi & ikon
                                  u"\U0001F1E0-\U0001F1FF"  # bendera (iOS)
                                  u"\U00002500-\U00002BEF"  # karakter CJK Extension A
                                  u"\U00002702-\U000027B0"
                                  u"\U00002702-\U000027B0"
                                  u"\U000024C2-\U0001F251"
                                  u"\U0001f926-\U0001f937"
                                  u"\U00010000-\U0010ffff"
                                  u"\u2640-\u2642"
                                  u"\u2600-\u2B55"
                                  u"\u200d"
                                  u"\u23cf"
                                  u"\u23e9"
                                  u"\u231a"
                                  u"\ufe0f"  # variasi penggunaan emoji
                                  u"\u3030"
                                 
                                  "]+", flags=re.UNICODE)
    return emoticon_pattern.sub(r'', text)  # Menghapus karakter emotikon dari teks

# Fungsi untuk menghapus duplikat dari teks
def remove_duplicates(text):
    # Membagi teks menjadi baris-baris unik
    unique_lines = set(text.split('\n'))
    # Menggabungkan baris-baris unik kembali menjadi teks
    clean_text = '\n'.join(unique_lines)
    return clean_text

def case_folding(text):
    # Lowercase text
    text = text.lower()
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove whitespace
    text = text.strip()
    
    return text

def remove_stopword(text):
    factory = StopWordRemoverFactory()
    stopwords = factory.get_stop_words()
    stopword = factory.create_stop_word_remover()
    
    text = stopword.remove(text)
    return text
            
# menghapus slangword
def remove_slangword(text):
    file_dict = "vocab.txt"
    with open(file_dict, 'r') as f:
            data = f.read()
            f.close()
    slangdict = json.loads(data)
    textiterable = text.splitlines()
    
    new_text = []
    for line in textiterable:
        if(line != ''):
            word = " ". join(slangdict.get(ele, ele) for ele in line.split())
            new_text.append(word)
        continue
    
    return "\n".join(new_text)
        
def stemming(text):
    # Inisialisasi stemmer dari Sastrawi
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    
#     stemmed_text = stemmer.stem(text)    
    new_text = []
    textiterable = text.splitlines()
    for line in textiterable:
        if(line != ''):
            stemmed_text = stemmer.stem(line)
            new_text.append(stemmed_text)
        continue
        
    return "\n".join(new_text)

def precleantext(raw_text):
    # Panggil fungsi case_folding untuk melakukan preprocessing pada teks
    processed_text = case_folding(raw_text)

    # Panggil fungsi remove_emoticons untuk menghapus emotikon dari teks
    cleaned_text = remove_dirty(processed_text)

    # Mengubah kata slang menjadi kata bahasa indonesia 
    cleaned_text = remove_slangword(cleaned_text)

    # Menghapus imbuhan
    cleaned_text = stemming(cleaned_text)

    # Menghapus kata sambung seperti "yang, di, ke"
    cleaned_text = remove_stopword(cleaned_text)

    return cleaned_text

def padding_sequence(dataframe):
    max_len = 150
    oov_tok = '<OOV>' # out of vocabulary token
    
    tokenizer = tensorflow.keras.preprocessing.text.Tokenizer(
        num_words = max_len,
                    char_level = False,
                      oov_token = oov_tok
    )
    tokenizer.fit_on_texts(dataframe)
    training_sequences = tokenizer.texts_to_sequences(dataframe)
    new_data = tensorflow.keras.preprocessing.sequence.pad_sequences(training_sequences,
                                    maxlen = max_len,
                                    padding = 'pre',
                                    truncating = 'pre'
    #                                 padding = padding_type,
    #                                 truncating = trunc_type
                                )
    
    return new_data

def predict_text(text, threshold=0.5):
    model = keras.models.load_model('lstm_model.h5', custom_objects={'SpatialDropout1D': SpatialDropout1D})
    
    data = pd.read_csv('sampilng.csv')  # Replace with actual file path
    X_train = data['tweet']
    
    max_len = 150
    oov_tok = "<OOV>"
    vocab_size = 450
    
    tokenizer = Tokenizer(num_words=vocab_size, char_level=False, oov_token=oov_tok)
    tokenizer.fit_on_texts(X_train)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='pre', truncating='pre')
    pred = model.predict(padded)
    class_pred = (pred >= threshold).astype(int)
    return class_pred

def padding_sequence_single(dataframe):
    max_len = 150
    oov_tok = '<OOV>' # out of vocabulary token
    
    tokenizer = tensorflow.keras.preprocessing.text.Tokenizer(
        num_words = max_len,
                    char_level = False,
                      oov_token = oov_tok
    )
    tokenizer.fit_on_texts(dataframe)
    training_sequences = tokenizer.texts_to_sequences([dataframe])
    new_data = tensorflow.keras.preprocessing.sequence.pad_sequences(training_sequences,
                                    maxlen = max_len,
                                    padding = 'pre',
                                    truncating = 'pre'
    #                                 padding = padding_type,
    #                                 truncating = trunc_type
                                )
    
    return new_data