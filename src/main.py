from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
import numpy as np
import pickle
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template_string
import xml.etree.ElementTree as ET
import re
from keras.utils import to_categorical

app = Flask(__name__)

HTML_TEMPLATE = """
<html>
<head>
    <title>Medical Chatbot</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f0f0f0;
        }
        .chat-container {
            max-width: 800px;
            margin: 40px auto;
            padding: 20px;
            background-color: #fff;
            border: 1px solid #ddd;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        .chat-header {
            background-color: #333;
            color: #fff;
            padding: 10px;
            border-bottom: 1px solid #333;
        }
        .chat-header h1 {
            margin: 0;
        }
        .chat-input {
            padding: 10px;
        }
        .chat-input input[type="text"] {
            width: 100%;
            padding: 10px;
            font-size: 16px;
            border: 1px solid #ccc;
        }
        .chat-input input[type="submit"] {
            background-color: #4CAF50;
            color: #fff;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .chat-input input[type="submit"]:hover {
            background-color: #3e8e41;
        }
        .chat-response {
            padding: 10px;
        }
        .chat-response p {
            margin: 10px 0;
        }
        .sidebar {
            position: fixed;
            top: 0;
            left: 0;
            width: 200px;
            height: 100%;
            background-color: #f0f0f0;
            padding: 20px;
            border-right: 1px solid #ddd;
            overflow: auto;
        }
        .sidebar ul {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        .sidebar li {
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }
        .sidebar li:hover {
            background-color: #eee;
        }
    </style>
</head>
<body>
    <div class="sidebar">
        <h2>Questions</h2>
        <ul>
            {% for question in questions %}
            <li>{{ question }}</li>
            {% endfor %}
        </ul>
    </div>
    <div class="chat-container">
        <div class="chat-header">
            <h1>QA Chatbot</h1>
        </div>
        <form action="/chat" method="post">
            <div class="chat-input">
                <label for="message">Message:</label>
                <input type="text" id="message" name="message">
                <input type="submit" value="Send">
            </div>
        </form>
        {% if response %}
        <div class="chat-response">
            <h2>Response:</h2>
            <p>{{ response }}</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

tree = ET.parse('ClinicalInquiries.xml')
root = tree.getroot()

train_data = []
train_labels = []

for record in root.findall('record'):
    question = record.find('question').text
    answer = record.find('answer').find('snip').find('sniptext').text
    train_data.append(question)
    train_labels.append(answer)

label_encoder = LabelEncoder()

encoded_labels = label_encoder.fit_transform(train_labels)
encoded_labels = to_categorical(encoded_labels)

try:
    with open('trained_data.pkl', 'rb') as f:
        print('\n\nREADING TRAINED DATA...')
        print('\n\n')
        tokenizer, label_encoder, model, train_sequences = pickle.load(f)
except FileNotFoundError:
    encoded_labels = label_encoder.fit_transform(train_labels)
    encoded_labels = to_categorical(encoded_labels)

    tokenizer = keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(train_data)
    train_sequences = tokenizer.texts_to_sequences(train_data)
    train_sequences = keras.preprocessing.sequence.pad_sequences(train_sequences)

    model = keras.models.Sequential()

    model.add(keras.layers.Embedding(len(tokenizer.word_index) + 1, 100, input_length=train_sequences.shape[1]))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(len(set(train_labels)), activation='softmax')) 


    model.compile(optimizer='adam', 
                loss='categorical_crossentropy', 
                metrics=['accuracy', 'mae', 'mse', 'categorical_crossentropy'])

    history = model.fit(train_sequences, encoded_labels, epochs=50, verbose=1) # Print the performance metrics for each epoch

    loss, accuracy, mae, mse, categorical_crossentropy = model.evaluate(train_sequences, encoded_labels)

    print(f'Loss: {loss:.3f}, Accuracy: {accuracy:.3f}, MAE: {mae:.3f}, MSE: {mse:.3f}, Categorical Cross-Entropy: {categorical_crossentropy:.3f}')

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['mae'])
    plt.plot(history.history['mse'])
    plt.plot(history.history['categorical_crossentropy'])
    plt.legend(['Accuracy', 'Loss', 'MAE', 'MSE', 'Categorical Cross-Entropy'])
    plt.show()

    with open('trained_data.pkl', 'wb') as f:
        pickle.dump((tokenizer, label_encoder, model, train_sequences), f)

def generate_response(text, tokenizer, model, train_sequences, label_encoder):
    sequence = tokenizer.texts_to_sequences([text])
    sequence = keras.preprocessing.sequence.pad_sequences(sequence, maxlen=train_sequences.shape[1])
    prediction = model.predict(sequence)
    predicted_label = np.argmax(prediction)
    response = label_encoder.inverse_transform([predicted_label])[0]
    return response

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE, questions=train_data)

@app.route('/chat', methods=['POST'])
def chat_post():
    user_input = request.form['message']
    response = generate_response(user_input, tokenizer, model, train_sequences, label_encoder)

    if response not in train_labels:
        response = "I'm not sure I understand. Can you please rephrase?"

    print('\n\n')
    print('Doctor_Bot:', response)
    print('\n\n')

    return render_template_string(HTML_TEMPLATE, response=response, questions=train_data)

if __name__ == '__main__':
    app.run(debug=True)
