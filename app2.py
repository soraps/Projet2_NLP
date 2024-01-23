import streamlit as st
import pandas as pd
from transformers import pipeline, set_seed

# Configuration pour assurer la reproductibilité lors de la génération de texte
set_seed(123)

# Titre de l'application
st.title('L\'analyse des avis des restaurants YELP')

# Chargement des données des restaurants
@st.cache_data
def load_restaurants():
    return pd.read_csv('restaurants.csv',sep=';',encoding='utf-8')


# Chargement des données
@st.cache_data
def load_reviews():
    data = pd.read_csv('review_final.csv',sep=';',encoding='utf-8')
    #data = pd.read_csv('data_train_review_app.csv',sep=';',encoding='utf-8')
    return data

restaurants_df = load_restaurants()
reviews_df = load_reviews()

# Création de la liste déroulante pour la sélection du restaurant
restaurant_id_to_name = dict(zip(restaurants_df['id'], restaurants_df['name']))
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
            st.subheader('Avis sélectionné au hasard :')
            st.subheader("Avis original:")
            st.write(random_review['text'], height=150)
            # Vérifiez si la colonne de correction existe et affichez-la
            if 'sentiment_vader' in random_review:
                st.subheader("Prediction de sentiment avec Vader: \n")
                st.write(random_review['sentiment_vader'])
            else:
                st.error("La colonne des prédiction de sentiment avec vader n'existe pas dans le DataFrame.")
            if 'sentiment_resultsBert_label' in random_review:
                st.subheader("Prediction de sentiment avec Bert:\n")
                st.write(random_review['sentiment_resultsBert_label'], height=150)
            else:
                st.error("La colonne des prédiction de sentiment avec Bert n'existe pas dans le DataFrame.")
            if 'cnn_sentiment_label' in random_review:
                st.subheader("Prediction de sentiment avec CNN:\n\n")
                st.write(random_review['cnn_sentiment_label'], height=150)
            else:
                st.error("La colonne des prédiction de sentiment avec CNN n'existe pas dans le DataFrame.")
        else:
            st.error('Aucun avis trouvé pour ce restaurant.')
    else:
        st.error("Erreur : Impossible de trouver l'ID du restaurant sélectionné.")
