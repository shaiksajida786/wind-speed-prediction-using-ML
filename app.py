from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

app = Flask(__name__)

# Load the dataset
wind_df = pd.read_csv('wind_dataset.csv')

# Drop the date column if it's not relevant for prediction
wind_df = wind_df.drop('DATE', axis=1)

# Assuming 'WIND' is the target variable and other columns are features
X = wind_df.drop('WIND', axis=1)  # Features
y = wind_df['WIND']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest regressor
rforest = RandomForestRegressor()
rforest.fit(X_train, y_train)
def categorize_weather(wind_speed):
    if wind_speed < 5:
        return "Calm", "Calm wind. Smoke rises vertically with little if any drift.", "No significant damage."
    elif 5 <= wind_speed < 10:
        return "Light Breeze", "Wind felt on face. Leaves rustle and small twigs move.", "Minor damage to loose outdoor objects."
    elif 10 <= wind_speed < 20:
        return "Moderate Wind", "Small trees sway. Loose objects are blown about.", "Damage to trees, branches, and power lines."
    elif 20 <= wind_speed < 30:
        return "Strong Wind", "Large branches move. Whistling heard in overhead lines.", "Structural damage possible, especially to roofs."
    elif 30 <= wind_speed < 40:
        return "Gale", "Whole trees sway. Difficulty walking against the wind.", "Widespread structural damage, power outages likely."
    else:
        return "Severe Gale", "Widespread damage. Large trees may be uprooted.", "Significant damage to buildings and infrastructure."
# Prediction route
@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get input values from the form
        
        ind = float(request.form['ind'])
        rain = float(request.form['rain'])
        ind_1 = float(request.form['ind_1'])
        t_max = float(request.form['t_max'])
        ind_2 = float(request.form['ind_2'])
        t_min = float(request.form['t_min'])
        t_min_g = float(request.form['t_min_g'])
        
        # Make prediction using the loaded model
        features = np.array([[ ind, rain, ind_1, t_max, ind_2, t_min, t_min_g]])
        prediction = rforest.predict(features)[0]
        chart_data = features.tolist() 
        # Categorize weather based on wind speed
        weather_condition, description, damage_effects = categorize_weather(prediction)
        
        return render_template('result.html', prediction=prediction, chart_data=chart_data, weather_condition=weather_condition, description=description, damage_effects=damage_effects)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
