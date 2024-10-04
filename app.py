from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import os
from caloriesLogic import get_daily_calories, validate_user_data

app = Flask(__name__)

# Load the trained models (update with your actual model file path)
model_filename = 'mealPredictingModel_2024-09-21_08-01-49.pkl'
with open(model_filename, 'rb') as model_file:
    models = pickle.load(model_file)

# Define the unique allergens from the trained model
unique_allergens = list(models.keys())

# Load the allergen mapping from 'finalAllergens.csv'
def load_allergen_mapping(allergen_csv_path):
    allergen_df = pd.read_csv(allergen_csv_path, encoding='ISO-8859-1')  # Specify encoding to avoid UnicodeDecodeError
    allergen_df.columns = allergen_df.columns.str.strip().str.lower()
    allergen_df['food'] = allergen_df['food'].str.strip().str.lower().fillna('none')
    allergen_df['allergen'] = allergen_df['allergen'].str.strip().str.lower().fillna('none')
    allergen_mapping = allergen_df.groupby('food')['allergen'].apply(list).to_dict()
    return allergen_mapping

allergen_csv_path = 'finalAllergens.csv'
allergen_mapping = load_allergen_mapping(allergen_csv_path)

# Load the meals data from 'finalMeals_with_diet_classification.csv'
def load_meals(meals_csv_path):
    meals_df = pd.read_csv(meals_csv_path, encoding='ISO-8859-1')  # Load the meals CSV
    # Ensure that all values in the 'ingredients' column are strings and handle missing values
    meals_df['ingredients'] = meals_df['ingredients'].fillna('').astype(str)
    meals_df['ingredients'] = meals_df['ingredients'].apply(lambda x: [i.strip().lower() for i in x.split(',')])
    return meals_df

meals_csv_path = 'finalMeals_with_diet_classification.csv'
meals_df = load_meals(meals_csv_path)

# Function to create features for ingredients
def create_features_vectorized(ingredients_list):
    features_df = pd.DataFrame(0, index=ingredients_list.index, columns=unique_allergens)
    for allergen, allergens in allergen_mapping.items():
        ingredients_mask = ingredients_list.apply(lambda ingredients: allergen in ingredients)
        for allergy in allergens:
            if allergy in features_df.columns:
                features_df.loc[ingredients_mask, allergy] = 1
    return features_df

# Updated Function to predict meal safety based on allergens and diet preferences
def predict_meal_safety_vectorized(ingredients_list, user_allergies, diet_preference):
    # Generate the features for all meals at once
    features_df = create_features_vectorized(ingredients_list)

    # Initialize a DataFrame to store predictions for each allergen
    predictions_df = pd.DataFrame(0, index=features_df.index, columns=unique_allergens)

    # Make batch predictions for each allergen
    for allergy in unique_allergens:
        if allergy in models:
            predictions_df[allergy] = models[allergy].predict(features_df)

    # Filter out meals that are not safe based on allergies (model predictions)
    unsafe_mask = predictions_df[user_allergies].max(axis=1) == 1
    safe_meals_from_model = meals_df.loc[~unsafe_mask]  # Meals predicted safe by model

    # Filter based on diet preferences (only show meals that match the user's preference)
    if diet_preference in safe_meals_from_model.columns:
        safe_meals_from_model = safe_meals_from_model[safe_meals_from_model[diet_preference] == 1]

    # Return the safe meals predicted by the model and then filtered by diet preferences
    return safe_meals_from_model['recipeName'].tolist()

@app.route('/')
def index():
    # Render HTML page with input field for allergies and user details
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the request (allergies + personal details)
    user_allergies = request.json['allergies']
    weight = request.json['weight']
    height = request.json['height']
    age = request.json['age']
    gender = request.json['gender']
    activity_level = request.json['activity_level']
    goal = request.json['goal']
    diet_preference = request.json['diet_preference']  # Example: 'vegan', 'keto', etc.

    # Perform meal safety prediction using vectorized operations
    safe_meals = predict_meal_safety_vectorized(meals_df['ingredients'], user_allergies, diet_preference)

    # Calculate daily caloric needs
    weight, height = validate_user_data(weight, height)  # Validate weight and height
    daily_calories = get_daily_calories(weight, height, age, gender, activity_level, goal)

    # Return the safe meals and calorie count as JSON response
    return jsonify({'safe_meals': safe_meals, 'daily_calories': daily_calories})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
