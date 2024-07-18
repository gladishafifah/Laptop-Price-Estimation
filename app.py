from flask import Flask, render_template, request, url_for
import pickle
import numpy as np
import pandas as pd

# Load model
model = pickle.load(open('model/linear.pkl', 'rb'))

app = Flask(__name__, template_folder="templates")

# Mapping dictionaries (example, adjust according to your mappings)
processor_mapping = {'Core i3': 0, 'M1': 1, 'Core i7': 2, 'Core i5': 3, 'Ryzen 5 Hexa Core': 4, 'Celeron Dual Core': 5, 'Ryzen 7 Octa Core': 6, 'Ryzen 5 Quad Core': 7, 'Ryzen 3 Dual Core': 8,
                     'Ryzen 3 Quad Core': 9, 'M2': 10, 'Celeron Quad Core': 11, 'Athlon Dual Core': 12, 'MediaTek Kompanio 1200': 13, 'Ryzen 9 Octa Core': 14, 'MediaTek MT8788': 15, 'Ryzen Z1 HexaCore': 16,
                     'MediaTek Kompanio 500': 17, 'Core i9': 18, 'MediaTek Kompanio 520': 19, 'Ryzen Z1 Octa Core': 20, 'Pentium Silver': 21, 'Ryzen 5': 22, 'M1 Max': 23, 'M2 Max': 24, 'M3 Pro': 25,
                     'M1 Pro': 26, 'Ryzen 7 Quad Core': 27, 'Ryzen 5 Dual Core': 28, 'Ryzen 9 16 Core': 29}

os_mapping = {'Windows 11 Home': 0,'Mac OS Big Sur': 1, 'DOS': 2, 'Mac OS Monterey': 3, 'Chrome': 4, 'Windows 10': 5, 'Windows 10 Home': 6, 'Prime OS': 7, 'Windows 11 Pro': 8, 'Ubuntu': 9, 'Windows 10 Pro': 10,
              'macOS Ventura': 11, 'macOS Sonoma': 12, 'Mac OS Mojave': 13}

# Function to preprocess input and make prediction
def preprocess_input(Processor, Operating_System, Touch_Screen, Storage, RAM, Screen_Size):
    processor_encode = processor_mapping.get(Processor, -1)
    os_encode = os_mapping.get(Operating_System, -1)

    # Return processed input as numpy array
    return np.array([processor_encode, os_encode, Touch_Screen, Storage, RAM, Screen_Size]).reshape(1, -1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # Get form values
    features = [x for x in request.form.values()]

    # Extract individual features
    Processor = features[0]
    Operating_System = features[1]
    Touch_Screen = int(features[2])
    Storage = int(features[3])
    RAM = int(features[4])
    Screen_Size = float(features[5])

     # Preprocess input
    final_features = preprocess_input(Processor, Operating_System, Touch_Screen, Storage, RAM, Screen_Size)

    # Make prediction
    prediction = model.predict(final_features)

    # Format output
    output = f'{prediction[0]:.3f}'

    return render_template('index.html', prediction_text=f'Prediksi Harga Laptop yaitu sebesar <b>{output} Rupee</b>')

if __name__ == '__main__':
    app.run(debug=True)
