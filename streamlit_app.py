import streamlit as st
import PIL.Image
from src.tester import Tester
import os
import json


CONFIG_PATH = os.path.join('config', 'test_config.json')
params = json.load(open(CONFIG_PATH, 'r'))
predict = Tester(**params)


def main():
    st.title("Tool to predict ant and bee image")

    image_uploaded = st.file_uploader("Choose an bee or ant image... ", type=['png', 'jpg', 'jpeg'],
                                      accept_multiple_files=False)

    if image_uploaded is not None:
        image = PIL.Image.open(image_uploaded).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        (W, H) = image.size
        st.write("Size of image: Weight is {} - Height is {}".format(W, H))
        st.write("")

    col1, col2 = st.beta_columns(2)
    with col1:
        pred_button = st.button("Predict")

    with col2:
        if pred_button:
            predicted = predict.predict_image(image_uploaded)
            st.write('Output: This is ', predicted)


if __name__ == '__main__':
    main()
