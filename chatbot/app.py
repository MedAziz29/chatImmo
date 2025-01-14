import pandas as pd
import re
from langdetect import detect
from googletrans import Translator
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import streamlit as st

# Initialisation du traducteur et du modèle
translator = Translator()

# Charger les données du fichier CSV
file_path = '/workspaces/chatImmo/chatbot/PROPERTY_ITEM.csv'
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

# Détection de la location dans la description
filtered_data["Location"] = filtered_data["Description"].apply(
    lambda x: "Lac2" if "Lac2" in x else ("Tunis" if "Tunis" in x else "Unknown")
)

# Fonction pour recommander des propriétés
def recommend_properties(bedrooms=None, max_price=None, min_surface=None, location=None):
    recommendations = filtered_data.copy()
    if bedrooms is not None:
        recommendations =  recommendations[recommendations["Bedrooms"] >= bedrooms]
    if max_price is not None:
        recommendations = recommendations[recommendations["Price (TND)"] <= max_price]
    if min_surface is not None:
        recommendations = recommendations[recommendations["Surface"] >= min_surface]
    if location is not None:
        recommendations = recommendations[recommendations["Location"].str.contains(location, case=False, na=False)]
    
    if recommendations.empty:
        # Si aucune propriété correspond, proposer une similaire
        if bedrooms is not None:
            recommendations = filtered_data[filtered_data["Bedrooms"] == bedrooms]
        if recommendations.empty and max_price is not None:
            recommendations = filtered_data[filtered_data["Price (TND)"] <= max_price]
        if recommendations.empty and min_surface is not None:
            recommendations = filtered_data[filtered_data["Surface"] >= min_surface]

    return recommendations

# Fonction de détection de langue
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

# Fonction de traduction
def translate_text(text, target_lang="en"):
    try:
        translated = translator.translate(text, dest=target_lang)
        return translated.text
    except:
        return text

# Fonction pour obtenir la réponse du bot
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

    response = ""
    if any([bedrooms, max_price, min_surface, location]):
        recommendations = recommend_properties(bedrooms=bedrooms, max_price=max_price, min_surface=min_surface, location=location)
        if not recommendations.empty:
            response = "Here are some suggestions before showing you the matching properties:\n"
            for idx, row in recommendations.head(5).iterrows():
                response += f"- {row['Title']} (Price: {row['Price (TND)']} TND, Bedrooms: {row['Bedrooms']}, Surface: {row['Surface']} m², Location: {row['Location']})\n"
                response += f"  Description: {row['Description'][:100]}...\n"
            # Afficher le tableau si des propriétés sont trouvées
            st.dataframe(recommendations[['Title', 'Price (TND)', 'Bedrooms', 'Surface', 'Location']])
            st.session_state.messages.append({"sender": "Bot", "message": response, "properties": recommendations.to_dict(orient="records")})
        else:
            response = "Sorry, no properties match your preferences exactly. Here are some similar options:\n"
            recommendations = recommend_properties(bedrooms=bedrooms, max_price=max_price, min_surface=min_surface, location=None)
            for idx, row in recommendations.head(3).iterrows():
                response += f"- {row['Title']} (Price: {row['Price (TND)']} TND, Bedrooms: {row['Bedrooms']}, Surface: {row['Surface']} m², Location: {row['Location']})\n"
    else:
        response = "I'm here to help you find properties!"

    if user_lang == "fr":
        response = translate_text(response, "fr")

    return response

# Configuration de Streamlit
st.set_page_config(page_title="Clé d'Or Chatbot", layout="wide")
st.title("Clé d'Or Property Finder Chatbot")

# Initialisation de l'historique des messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage de l'historique des messages
st.write("### Chat History")
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if msg["sender"] == "You":
            st.markdown(f"<div style='border-radius:15px; padding:10px; margin:5px; width:80%; text-align:right; box-shadow: 0px 2px 5px rgba(0,0,0,0.1);'>{msg['message']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='border-radius:15px; padding:10px; margin:5px; width:80%; text-align:left; box-shadow: 0px 2px 5px rgba(0,0,0,0.1);'>{msg['message']}</div>", unsafe_allow_html=True)

# Zone de saisie
user_input = st.text_input("Type your message:", key="user_input", label_visibility="collapsed")
if user_input:
    user_lang = detect_language(user_input)
    bot_response = get_bot_response(user_input, user_lang)

    # Ajout des messages dans l'historique
    st.session_state.messages.append({"sender": "You", "message": user_input})
    st.session_state.messages.append({"sender": "Bot", "message": bot_response})

st.write("---")
