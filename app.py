import streamlit as st
from transformers import pipeline, set_seed, AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, MarianMTModel, MarianTokenizer
#from setfit import AbsaModel
import pandas as pd
from langdetect import detect
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoModelForSequenceClassification
import random
from setfit import AbsaModel
from transformers import pipeline, set_seed, GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer



# Configuration pour assurer la reproductibilité lors de la génération de texte
set_seed(123)

# Titre de l'application
st.title('L\'analyse des avis des restaurants YELP')

# Chargement des données des restaurants
@st.cache_data
def load_restaurants():
    return pd.read_csv('business.csv',sep=';',encoding='utf-8')


# Chargement des données
@st.cache_data
def load_reviews():
    #data = pd.read_csv('review_final.csv')
    data = pd.read_csv('data_train_review_app.csv',sep=';',encoding='utf-8')
    return data

restaurants_df = load_restaurants()
reviews_df = load_reviews()

# Initialisation des pipelines de Transformers pour la correction et la traduction
tokenizer_correction = AutoTokenizer.from_pretrained("vennify/t5-base-grammar-correction")
model_correction = AutoModelForSeq2SeqLM.from_pretrained("vennify/t5-base-grammar-correction")
grammar_correction = pipeline("text2text-generation", model=model_correction, tokenizer=tokenizer_correction)

fix_spelling = pipeline("text2text-generation", model="oliverguhr/spelling-correction-english-base")

# Initialisation du modèle de traduction Marian
tokenizer_translation = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
model_translation = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-es-en")

# Interface utilisateur pour saisir un avis
st.write('Correction et Traduction de l\'avis d\'un restaurant')
user_input = st.text_area("Écrivez votre avis ici :")

if st.button('Corriger et Traduire l\'avis'):
    if user_input:
        # Correction de la grammaire et de l'orthographe
        corrected_output = grammar_correction(user_input, max_length=512)
        corrected_text = corrected_output[0]['generated_text']
        st.write('Avis corrigé :', corrected_text)

        # Correction de l'orthographe (optionnel, si vous souhaitez l'utiliser)
        spelled_corrected_text = fix_spelling(corrected_text, max_length=2048)
        st.write('Correction orthographique :', spelled_corrected_text[0]['generated_text'])

        # Détection de la langue
        detected_language = detect(corrected_text)
        st.write('Langue détectée :', detected_language)

        # Traduction en anglais si nécessaire
        if detected_language != 'en':
            # Préparer le texte pour la traduction
            translated = model_translation.generate(**tokenizer_translation(corrected_text, return_tensors="pt", padding=True))
            translated_text = tokenizer_translation.decode(translated[0], skip_special_tokens=True)
            st.write('Avis traduit en anglais :', translated_text)
        else:
            st.write('Avis déjà en anglais :', corrected_text)
    else:
        st.error('Veuillez saisir un avis avant de cliquer sur le bouton.')

# Chargement du modèle BERT pour l'analyse de sentiment
sentiment_analysis = pipeline("text-classification", model="mrcaelumn/yelp_restaurant_review_sentiment_analysis")

# Fonction pour convertir le label en score numérique
def convert_label(label):
    return {'LABEL_2': 'Positive', 'LABEL_1': 'Neutre', 'LABEL_0': 'Negative'}.get(label, 0)

# Création de la liste déroulante pour la sélection du restaurant
restaurant_id_to_name = dict(zip(restaurants_df['business_id'], restaurants_df['name']))
restaurant_selection = st.selectbox('Choisissez un restaurant', options=list(restaurant_id_to_name.values()))

# Lorsque l'utilisateur sélectionne un restaurant, trouvez son 'business_id'
selected_business_id = None
for business_id, name in restaurant_id_to_name.items():
    if name == restaurant_selection:
        selected_business_id = business_id
        break

# Afficher un avis au hasard pour le restaurant sélectionné et sa correction
if st.button('Afficher un avis au hasard'):
    if selected_business_id:
        # Filtrer les avis pour le restaurant sélectionné
        selected_reviews = reviews_df[reviews_df['business_id'] == selected_business_id]
        if not selected_reviews.empty:
            # Sélectionner un avis au hasard
            random_review = selected_reviews.sample(1).iloc[0]
            st.write('Avis sélectionné au hasard :')
            st.text_area("Avis original", random_review['text'], height=150)
            # Analyse de sentiment avec VADER
            if 'sentiment_vader' in random_review:
                st.write("Analyse de sentiment avec VADER :", random_review['sentiment_vader'], height=150)
            else:
                st.error("La colonne des avis corrigés n'existe pas dans le DataFrame.")

            #Analyse de sentiment avec BERT
            bert_result = sentiment_analysis(random_review['text'])
            sentiment_score = convert_label(bert_result[0]['label'])
            st.write("Analyse de sentiment avec BERT :", sentiment_score)


    else:
        st.error("Erreur : Impossible de trouver l'ID du restaurant sélectionné.")


#Chargement ABSA
absa_model = AbsaModel.from_pretrained(
    "tomaarsen/setfit-absa-paraphrase-mpnet-base-v2-restaurants-aspect",
    "tomaarsen/setfit-absa-paraphrase-mpnet-base-v2-restaurants-polarity"
)# Section ABSA avec menu déroulant pour le choix des commentaires
st.subheader('Analyse de sentiment des aspects (ABSA)')
# Liste des commentaires prédéfinis
predefined_reviews = [
    "The ambiance was lovely as well as the service, but the food was not that good.",
    "The food was excellent, but the service was too slow.",
    "Great location, but the prices are too high for the quality provided.",
    "The dessert was divine, though the main course did not meet expectations."
]

# Création du menu déroulant avec les commentaires prédéfinis
selected_review = st.selectbox("Choisir un commentaire pour l'analyse :", predefined_reviews)

if st.button('Analyser les aspects'):
    if selected_review:
        # Effectuer l'analyse ABSA
        aspects = absa_model.predict(selected_review)
        
        # Afficher les résultats
        if aspects:
            st.write('Résultats de l\'analyse des aspects :')
            for aspect in aspects:
                st.json(aspect)
        else:
            st.error('Aucun aspect détecté dans l\'avis.')
    else:
        st.error('Veuillez choisir un commentaire pour l\'analyse.')


# Fonction pour effectuer le résumé des avis
def summarize_text(text, sentences_count):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = Summarizer()
    summary = summarizer(parser.document, sentences_count)
    return ' '.join([str(sentence) for sentence in summary])



# Regrouper les avis par 'business_id' et créer un résumé pour chaque restaurant
grouped_reviews = reviews_df.groupby('business_id')['text'].apply(' '.join)

# Interface utilisateur pour la section de résumé
st.subheader("Résumé des avis des restaurants")
restaurant_id = st.selectbox("Choisissez un ID de restaurant", options=grouped_reviews.index)
selected_sentences_count = st.slider("Choisissez le nombre de phrases pour le résumé", min_value=1, max_value=10, value=2)

if st.button("Générer un résumé"):
    # Obtenir les avis pour le restaurant sélectionné
    reviews_text = grouped_reviews[restaurant_id]
    
    # Limiter le texte aux 3 premiers avis pour éviter une surcharge de texte
    reviews_text = ' '.join(reviews_text.split(' ')[:500])  # Ajuster le nombre de mots si nécessaire
    
    # Générer le résumé
    summary = summarize_text(reviews_text, selected_sentences_count)
    
    st.write("Avis pour le restaurant sélectionné :")
    st.text_area(reviews_text)
    st.write("Résumé :")
    st.text(summary)

# Chargement des modèles et initialisation de la pipeline de génération de texte
dialo_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
dialo_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
dialogue_generator = pipeline('text-generation', model=dialo_model, tokenizer=dialo_tokenizer)


# Interface utilisateur pour DialoGPT
st.subheader('DialoGPT Insights Generator')
st.write('Entrez les avis clients ou sélectionnez des avis prédéfinis pour générer des insights.')

# Sélection des avis prédéfinis
default_reviews = ["Great service but the food was bland.Loved the ambiance and the dessert, but the main course was too salty. The waiter was rude. ", "The ambiance was lovely, but the food was terrible and the service was great."]
selected_reviews = st.selectbox('Choisissez des avis prédéfinis', options=default_reviews)

# Entrée pour les avis personnalisés
user_reviews = st.text_area("Ou écrivez vos propres avis ici :", height=100, value=selected_reviews)

combined_reviews = " ".join(user_reviews[:5])  # Limiter à 5 avis pour l'analyse
prompt = (
    f"Customer Reviews: {combined_reviews}\n"
    f"AI: Based on these reviews, the key weaknesses of the restaurant are:"
)
generated_response = dialogue_generator(prompt, max_length=300)
answer=generated_response[0]['generated_text']
st.write('Insights générés :', answer)
   