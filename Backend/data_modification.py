import pandas as pd;
from wordcloud import WordCloud, STOPWORDS
def concatenation(series_data, numpyarray_result):
    result_dataframe = pd.DataFrame(numpyarray_result, columns=['label'])
    new_data = pd.concat([series_data, result_dataframe], axis=1)
    # print(numpyarray_result['label'])
    
    # df = pd.DataFrame({'tweet' : series_data, 'label' : numpyarray_result.flatten()})
    
    return new_data

def split_neg_post(dataframe_result):
    negative = dataframe_result.loc[dataframe_result['label'] < 0.5]
    positive = dataframe_result.loc[dataframe_result['label'] > 0.5]
    
    return negative, positive

def count_word(dataframe_result):
    count_word = ''
    for index,val in dataframe_result.iterrows():
        word = str(val['tweet'])
        tokens = word.split()
        
        count_word = " ".join(tokens)+" "
    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          min_font_size=10,
                          stopwords=STOPWORDS
                          ).generate(word)
    
    return wordcloud
    
    
    