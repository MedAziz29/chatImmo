import pandas as pd
import re
from langdetect import detect
from googletrans import Translator
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import streamlit as st

translator = Translator()


file_path = './PROPERTY_ITEM.csv'
data = pd.read_csv(file_path)

filtered_data = data[[
    "PI_TITLE", "PI_CONTENT", "PI_ATTR_BED", "PI_ATTR_BATH",
    "PI_ATTR_SURFACE", "PI_PRICE_TND", "PI_ATTR_PARKING", "PI_ALIAS"
]].rename(columns={
    "PI_TITLE": "Title",
    "PI_CONTENT": "Description",
    "PI_ATTR_BED": "Bedrooms",
    "PI_ATTR_BATH": "Bathrooms",
    "PI_ATTR_SURFACE": "Surface",
    "PI_PRICE_TND": "Price (TND)",
    "PI_ATTR_PARKING": "Parking",
    "PI_ALIAS": "Link"
})

filtered_data["Location"] = filtered_data["Description"].apply(
    lambda x: "Lac2" if "Lac2" in x else ("Tunis" if "Tunis" in x else "Unknown")
)

def recommend_properties(bedrooms=None, max_price=None, min_surface=None, location=None):
    recommendations = filtered_data.copy()
    if bedrooms is not None:
        recommendations = recommendations[recommendations["Bedrooms"] >= bedrooms]
    if max_price is not None:
        recommendations = recommendations[recommendations["Price (TND)"] <= max_price]
    if min_surface is not None:
        recommendations = recommendations[recommendations["Surface"] >= min_surface]
    if location is not None:
        recommendations = recommendations[
            recommendations["Location"].str.contains(location, case=False, na=False)
        ]
    return recommendations


model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"


def translate_text(text, target_lang="en"):
    try:
        translated = translator.translate(text, dest=target_lang)
        return translated.text
    except:
        return text


def get_bot_response(user_input, user_lang):
    if user_lang == "fr":
        user_input = translate_text(user_input, "en")

    bedrooms, max_price, min_surface, location = None, None, None, None

    bedrooms_match = re.search(r"\b(\d+)\s*bedrooms?", user_input, re.IGNORECASE)
    if bedrooms_match:
        bedrooms = int(bedrooms_match.group(1))

    price_match = re.search(r"\bprice\s*under\s*(\d+)", user_input, re.IGNORECASE)
    if price_match:
        max_price = float(price_match.group(1))

    surface_match = re.search(r"\bsurface\s*at least\s*(\d+)", user_input, re.IGNORECASE)
    if surface_match:
        min_surface = float(surface_match.group(1))

    location_match = re.search(r"(in|at|near)\s+([\w\s]+)", user_input, re.IGNORECASE)
    if location_match:
        location = location_match.group(2).strip()

    if any([bedrooms, max_price, min_surface, location]):
        recommendations = recommend_properties(bedrooms=bedrooms, max_price=max_price, min_surface=min_surface, location=location)
        if not recommendations.empty:
            response = "Here are some properties matching your preferences:\n"
            for idx, row in recommendations.head(3).iterrows():
                response += f"- {row['Title']} (Price: {row['Price (TND)']} TND, Bedrooms: {row['Bedrooms']}, Surface: {row['Surface']} m², Location: {row['Location']})\n"
                response += f"  Description: {row['Description'][:100]}...\n"
        else:
            response = "Sorry, no properties match your preferences."
    else:
        new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
        bot_input_ids = new_input_ids
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    if user_lang == "fr":
        response = translate_text(response, "fr")

    return response


st.set_page_config(page_title="Clé d'Or Chatbot", layout="wide")
st.title("Clé d'Or Property Finder Chatbot")


if "messages" not in st.session_state:
    st.session_state.messages = []


user_input = st.text_input("Type your message:", key="user_input")
if user_input:
    user_lang = detect_language(user_input)
    bot_response = get_bot_response(user_input, user_lang)

   
    st.session_state.messages.append({"sender": "You", "message": user_input})
    st.session_state.messages.append({"sender": "Bot", "message": bot_response})

st.write("### Chat History")
for msg in st.session_state.messages:
    if msg["sender"] == "You":
        st.markdown(f"<div style='text-align: right;'><strong>{msg['sender']}:</strong> {msg['message']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align: left;'><strong>{msg['sender']}:</strong> {msg['message']}</div>", unsafe_allow_html=True)

st.write("---")
