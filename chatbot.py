import os
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

intents=[
    {
    "tag": "music",
    "patterns": ["Recommend a song", "What's your favorite music", "Any good music suggestions"],
    "responses": ["Sure, check out [Song Title] by [Artist]. It's a [genre] song with a great beat.", "I'm a fan of [Genre] music. You might enjoy [Song Title] by [Artist].", "If you're in the mood for [Genre], give [Song Title] a listen. It's a catchy tune by [Artist]."]
},
{
    "tag": "fitness",
    "patterns": ["Give me a workout idea", "How can I stay fit", "Fitness tips"],
    "responses": ["Try a quick home workout: 20 jumping jacks, 15 squats, 10 push-ups, and a 1-minute plank.", "Staying fit is essential. Incorporate activities like walking, jogging, or cycling into your routine.", "Fitness tip: Consistency is key. Find activities you enjoy to make staying active a habit."]
},
{
    "tag": "recipes",
    "patterns": ["Suggest a recipe", "What should I cook for dinner", "Any tasty recipes"],
    "responses": ["How about making a delicious [Cuisine] dish? I recommend trying [Recipe Name].", "If you're a fan of [Ingredient], you'll love the [Recipe Type] recipe. Give it a shot!", "Cooking tip: Experiment with flavors! Add a pinch of [Spice] to enhance the taste of your dishes."]
},
{
    "tag": "technology",
    "patterns": ["Latest tech news", "Tell me about a cool gadget", "Tech trends"],
    "responses": ["Stay updated on the latest tech trends by following reputable tech blogs and news websites.", "Have you heard about [New Technology]? It's making waves in the tech industry for its innovative features.", "Tech tip: Regularly update your devices and software to ensure security and access new features."]
},
{
    "tag": "inspiration",
    "patterns": ["I need motivation", "Inspire me", "Quotes for inspiration"],
    "responses": ["Remember, every expert was once a beginner. Don't be afraid to take the first step.", "Success is not final, failure is not fatal: It's the courage to continue that counts.", "Inspiration tip: Set small, achievable goals to build confidence on your journey."]
}

]

vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)


x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)


def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

counter = 0

def main():
    global counter
    st.title("Chatbot")
    st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")

    counter += 1
    user_input = st.text_input("You:", key=f"user_input_{counter}")

    if user_input:
        response = chatbot(user_input)
        st.text_area("Chatbot:", value=response, height=100, max_chars=None, key=f"chatbot_response_{counter}")

        if response.lower() in ['goodbye', 'bye']:
            st.write("Thank you for chatting with me. Have a great day!")
            st.stop()

if __name__ == '__main__':
    main()            


