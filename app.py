import os
from flask import Flask, request, jsonify
import pandas as pd
import pickle
from caloriesLogic import get_daily_calories
from firebaseHandler import initialize_firestore, get_user_data_from_firestore

# Import functions from caloriesLogic.py
from caloriesLogic import get_daily_calories, validate_user_data

app = Flask(__name__)

# Load the trained models (update with your actual model file path)
model_filename = 'mealPredictingModel_2024-09-21_08-01-49.pkl'
with open(model_filename, 'rb') as model_file:
    models = pickle.load(model_file)

# Define the unique allergens from the trained model
unique_allergens = list(models.keys())

# Load the allergen mapping from 'Allergens.csv'
def load_allergen_mapping(allergen_csv_path):
    allergen_df = pd.read_csv(allergen_csv_path, encoding='ISO-8859-1')  # Specify encoding to avoid UnicodeDecodeError
    allergen_df.columns = allergen_df.columns.str.strip().str.lower()
    allergen_df['food'] = allergen_df['food'].str.strip().str.lower().fillna('none')
    allergen_df['allergen'] = allergen_df['allergen'].str.strip().str.lower().fillna('none')
    allergen_mapping = allergen_df.groupby('food')['allergen'].apply(list).to_dict()
    return allergen_mapping

allergen_csv_path = 'finalAllergens.csv'
allergen_mapping = load_allergen_mapping(allergen_csv_path)

# Add root route to the app
@app.route('/')
def index():
    return "Welcome to the Diet Recommendation Microservice! Use the '/predict' endpoint to get meal recommendations."

# Rest of your code remains the same...

def classify_meals(meals_df):
    dietary_keywords = {
        # Add your diet preference keywords...
    }

    # Your classification logic...

    return meals_df

# Load meals and apply dietary classification
meals_csv_path = 'finalMeals.csv'
meals_df = load_meals(meals_csv_path)

def create_features_vectorized(ingredients_list):
    features_df = pd.DataFrame(0, index=ingredients_list.index, columns=unique_allergens)
    for allergen, allergens in allergen_mapping.items():
        ingredients_mask = ingredients_list.apply(lambda ingredients: allergen in ingredients)
        for allergy in allergens:
            if allergy in features_df.columns:
                features_df.loc[ingredients_mask, allergy] = 1
    return features_df

def predict_meal_safety_with_diet(ingredients_list, user_allergies, diet_preference):
    features_df = create_features_vectorized(ingredients_list)
    predictions_df = pd.DataFrame(0, index=features_df.index, columns=unique_allergens)

    for allergy in unique_allergens:
        if allergy in models:
            predictions_df[allergy] = models[allergy].predict(features_df)

    # Filter out unsafe meals
    unsafe_mask = predictions_df[user_allergies].max(axis=1) == 1
    safe_meals = meals_df.loc[~unsafe_mask]

    # Apply dietary preference filter
    if diet_preference in meals_df.columns:
        safe_meals = safe_meals[safe_meals[diet_preference] == 1]

    return safe_meals['recipeName'].tolist()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_id = data.get('user_id')
    
    # Initialize Firestore and fetch user data
    db = initialize_firestore()
    user_data = get_user_data_from_firestore(db, user_id)
    
    if not user_data:
        return jsonify({"error": "User data not found"}), 404
    
    # Extract NutriInfo fields
    user_allergies = user_data.get('allergies', [])
    diet_preference = user_data.get('diet', 'none')
    weight = user_data.get('weight')
    height = user_data.get('height')
    age = user_data.get('age')
    gender = user_data.get('gender')
    activity_level = user_data.get('activity')
    goal = user_data.get('goal')

    # Perform meal safety prediction
    safe_meals = predict_meal_safety_with_diet(meals_df['ingredients'], user_allergies, diet_preference)

    # Calculate daily caloric needs
    daily_calories = get_daily_calories(weight, height, age, gender, activity_level, goal)

    return jsonify({'safe_meals': safe_meals, 'daily_calories': daily_calories})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
