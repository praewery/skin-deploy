# Include PIL, load_image before main()
from pyrsistent import s
import streamlit as st
import os
from PIL import Image
from fastai.vision.all import (
    load_learner,
    PILImage,
)
import pathlib
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

MAIN_MODEL = pathlib.Path("dbc_resnet34_fastai.pkl")

learn_inf = load_learner(MAIN_MODEL)


def load_image(image_file):
    img = Image.open(image_file)
    return img


st.set_page_config(
    page_title=" Skin Diseases Classification",
    layout="centered",

)




st.title("Skin Diseases Classification")
st.subheader("Something's wrong on your Skin? Let's find it.")

image_file = st.file_uploader("Upload Leaf Images",
                                  type=["png", "jpg", "jpeg","webp"])


def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

if image_file is not None:

    st.image(load_image(image_file), width=None)

    # Saving upload
    with open(os.path.join("./images/upload/", image_file.name), "wb") as f:
        f.write((image_file).getbuffer())

    
        file = f"./images/upload/{image_file.name}"

        #checkleaf = good_or_bad.predict(file)
        result = learn_inf.predict(file)
        predict = f"<div style='background:#00d26a;padding:0 20px 0 20px;border-radius:10px 10px 0 0;'><h1 style='color:#fff'>{result[0]}</h1></div>"
        st.markdown(predict, unsafe_allow_html=True)

        diseases = [
                "Acne and Rosacea Photos",
                "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
                "Atopic Dermatitis Photos",
                "Bullous Disease Photos",
                "Cellulitis Impetigo and other Bacterial Infections",
                "Eczema Photos",
                "Exanthems and Drug Eruptions",
                "Hair Loss Photos Alopecia and other Hair Diseases",
                "Herpes HPV and other STDs Photos",
                "Light Diseases and Disorders of Pigmentation",
                "Lupus and other Connective Tissue diseases",
                "Melanoma Skin Cancer Nevi and Moles",
                "Nail Fungus and other Nail Disease"
            ]

        description = [
                
                ""
            ]

        index = diseases.index(result[0])

        st.markdown(description[index], unsafe_allow_html=True)

