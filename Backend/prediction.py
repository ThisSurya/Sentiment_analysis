import keras;
def predict_text(clean_paddsequence):
    model = keras.models.load_model('lstm_model.h5')
    y_predict = model.predict(clean_paddsequence)

    return y_predict
    if y_predict > 0.5:
        return 'positif'
    else:
        return 'negatif'