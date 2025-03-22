import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import decode_predictions, preprocess_input
import os
from PIL import Image
import io
import requests
import json

# Configuration de la page
st.set_page_config(
    page_title="Assistant Races de Chiens",
    page_icon="🐶",
    layout="wide"
)

# Titre et description de l'application
st.title("🐶 Identification et Conseils sur les Races de Chiens")
st.write("Téléchargez une photo de chien pour identifier sa race et obtenir des conseils personnalisés")

# Récupération de la clé API depuis les secrets
try:
    api_key = st.secrets["api_mistral"]
    api_configured = True
except Exception:
    api_configured = False
    st.warning("Clé API Mistral non configurée dans les secrets. Certaines fonctionnalités peuvent ne pas être disponibles.")
    # Optionnel: Permettre la saisie manuelle si non configurée dans les secrets
    api_key = st.text_input("Entrez votre clé API Mistral", type="password")

# Vérifiez si la clé API est fournie
if api_key:
    # Configuration de l'API Mistral
    MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

    # Fonction pour communiquer avec l'API Mistral
    def query_mistral(prompt, system_prompt=""):
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": "mistral-large-latest",  # Vous pouvez choisir un modèle différent selon vos besoins
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(MISTRAL_API_URL, headers=headers, data=json.dumps(payload))
            response.raise_for_status()  # Lève une exception pour les codes d'erreur HTTP
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            st.error(f"Erreur lors de la communication avec Mistral AI: {str(e)}")
            return "Je ne peux pas générer d'informations sur cette race pour le moment. Veuillez réessayer plus tard."

    # Chargement du modèle de vision par ordinateur
    @st.cache_resource
    def load_model():
        return ResNet50(weights='imagenet')

    # Charger le modèle
    model = load_model()

    # Fonction pour obtenir des conseils du LLM
    def get_dog_advice(breed_name, confidence):
        system_prompt = "Tu es un expert vétérinaire spécialisé dans les races de chiens. Donne des conseils précis, factuels et concis."
        
        prompt = f"""
        En tant qu'expert canin, donne-moi des informations et conseils concis sur la race de chien '{breed_name}' (identifiée avec {confidence:.1f}% de confiance).
        
        Structure tes conseils en courtes sections sur :
        
        1) Personnalité et caractère
        2) Besoins d'exercice et d'activité 
        3) Entretien et soins
        4) Compatibilité avec les enfants
        5) Compatibilité avec les autres animaux, notamment les chats
        6) Santé et préoccupations médicales courantes
        7) 3 conseils pratiques pour les propriétaires
        
        Reste factuel, précis et concis pour chaque section.
        """
        
        return query_mistral(prompt, system_prompt)

    # Liste des mots-clés associés aux races de chiens dans ImageNet
    DOG_KEYWORDS = [
        'terrier', 'retriever', 'spaniel', 'husky', 'malamute', 'bulldog', 'poodle', 
        'shepherd', 'collie', 'hound', 'mastiff', 'beagle', 'doberman', 'boxer', 
        'labrador', 'dachshund', 'rottweiler', 'corgi', 'dalmatian', 'schnauzer',
        'setter', 'wolfhound', 'newfoundland', 'sheepdog', 'pug', 'chihuahua',
        'maltese', 'papillon', 'pekinese', 'shih-tzu', 'chow', 'pomeranian',
        'samoyed', 'spitz', 'akita', 'basenji', 'bernese', 'whippet', 'greyhound',
        'komondor', 'kuvasz', 'vizsla', 'weimaraner', 'yorkshire', 'bouvier',
        'kelpie', 'borzoi', 'saluki', 'afghan', 'basset', 'bloodhound', 'bluetick',
        'coonhound', 'dhole', 'dingo', 'leonberg', 'malinois', 'mexicanhairless',
        'otterhound', 'pembroke', 'pinscher', 'pitbull', 'ridgeback', 'sennenhund',
        'pyrenees', 'appenzeller', 'dog'
    ]

    # Fonction de prédiction
    def predict_image(img_array):
        # Prétraiter l'image
        img_array = preprocess_input(img_array)
        
        # Prédiction
        predictions = model.predict(img_array)
        
        # Décoder les prédictions pour obtenir les classes et probabilités
        decoded_predictions = decode_predictions(predictions, top=5)[0]
        
        # Vérifier si la prédiction principale est un chien
        is_dog = any(any(keyword in pred[1].lower() for keyword in DOG_KEYWORDS) for pred in decoded_predictions[:2])
        
        return decoded_predictions, is_dog

    # Interface utilisateur
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Téléchargement d'image")
        uploaded_file = st.file_uploader("Choisissez une image de chien...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Afficher l'image téléchargée
            img = Image.open(uploaded_file)
            st.image(img, caption="Image téléchargée", use_column_width=True)
            
            # Préparation de l'image pour la prédiction
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Bouton pour lancer la prédiction
            if st.button("Identifier l'image"):
                with st.spinner("Analyse en cours..."):
                    # Obtenir les prédictions
                    predictions, is_dog = predict_image(img_array)
                    
                    # Afficher les prédictions
                    st.subheader("Résultats de l'identification")
                    
                    for i, (imagenet_id, label, score) in enumerate(predictions[:5]):
                        label_formatted = label.replace('_', ' ').title()
                        st.write(f"{i + 1}: {label_formatted} ({score * 100:.1f}%)")
                    
                    if is_dog:
                        # C'est un chien, on peut stocker les prédictions
                        best_dog_pred = next((pred for pred in predictions if any(keyword in pred[1].lower() for keyword in DOG_KEYWORDS)), predictions[0])
                        
                        st.session_state.is_dog = True
                        st.session_state.current_prediction = best_dog_pred[1].replace('_', ' ').title()
                        st.session_state.confidence = best_dog_pred[2] * 100
                        
                        st.success(f"✅ Un chien a été identifié! Il s'agit probablement d'un(e) {st.session_state.current_prediction}.")
                    else:
                        # Ce n'est probablement pas un chien
                        st.session_state.is_dog = False
                        st.warning("⚠️ Cette image ne semble pas contenir un chien. Les conseils sur les races ne sont disponibles que pour les chiens.")

    with col2:
        st.subheader("Conseils sur la race identifiée")
        
        if 'is_dog' in st.session_state:
            if st.session_state.is_dog:
                breed = st.session_state.current_prediction
                confidence = st.session_state.confidence
                
                st.markdown(f"### {breed}")
                
                with st.spinner("Génération des conseils en cours..."):
                    advice = get_dog_advice(breed, confidence)
                    st.markdown(advice)
                    
                # Zone de chat pour poser des questions supplémentaires
                st.subheader("Posez vos questions sur cette race")
                user_question = st.text_input("Votre question:", placeholder="Exemple: Est-ce que cette race est adaptée aux appartements?")
                
                if user_question:
                    with st.spinner("Réponse en cours..."):
                        system_prompt = "Tu es un expert vétérinaire spécialisé dans les races de chiens. Donne des conseils précis, factuels et concis."
                        
                        chat_prompt = f"""
                        Race de chien identifiée: {breed} (confiance: {confidence:.1f}%)
                        
                        Question de l'utilisateur: {user_question}
                        
                        Réponds à cette question spécifique de manière concise, factuelle et utile, en te concentrant uniquement sur cette race de chien.
                        """
                        
                        response = query_mistral(chat_prompt, system_prompt)
                        st.markdown(response)
            else:
                st.error("Aucun chien n'a été identifié dans l'image. Veuillez télécharger une photo de chien pour obtenir des conseils spécifiques.")
                
                # Proposer une option pour identifier l'animal ou l'objet détecté
                if st.button("Que contient cette image?"):
                    top_prediction = predictions[0][1].replace('_', ' ').title()
                    confidence = predictions[0][2] * 100
                    
                    system_prompt = "Tu es un expert en identification d'images et en animaux."
                    
                    prompt = f"""
                    L'image a été identifiée comme contenant principalement un(e) {top_prediction} (confiance: {confidence:.1f}%).
                    
                    Donne une brève description de cet animal/objet en 3-4 phrases maximum. Si c'est un animal, mentionne quelques informations générales à son sujet.
                    """
                    
                    with st.spinner("Génération d'informations..."):
                        info = query_mistral(prompt, system_prompt)
                        st.markdown(f"### {top_prediction}")
                        st.markdown(info)
        else:
            st.info("�� Veuillez télécharger une image et lancer l'identification pour obtenir des conseils personnalisés.")

    # Ajouter quelques informations supplémentaires
    st.sidebar.title("À propos")
    st.sidebar.info("""
    Cette application utilise:
    - ResNet50 pour identifier l'animal ou l'objet dans l'image
    - Mistral AI pour générer des conseils personnalisés sur les races de chiens
    - Streamlit pour l'interface utilisateur

    **Note**: Pour obtenir des conseils sur les races de chiens, veuillez télécharger une image contenant un chien.
    """)

    # Ajouter une section pour les races populaires
    with st.expander("Découvrir des races populaires"):
        popular_breeds = {
            "Labrador Retriever": "Amical, sociable et parfait pour les familles",
            "Berger Allemand": "Intelligent, loyal et protecteur",
            "Golden Retriever": "Affectueux, intelligent et excellent avec les enfants",
            "Bouledogue Français": "Enjoué, alerte et s'adapte bien à la vie en appartement",
            "Beagle": "Curieux, joyeux et bon avec les enfants"
        }
        
        for breed, description in popular_breeds.items():
            st.markdown(f"**{breed}**: {description}")
else:
    st.warning("Veuillez entrer votre clé API pour continuer. (https://console.mistral.ai/api-keys) c'est gratuit !")
