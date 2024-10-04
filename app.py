from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
from caloriesLogic import get_daily_calories, initialize_firestore, get_user_data_from_firestore

app = Flask(__name__)

# Load the trained models (update with your actual model file path)
model_filename = 'mealPredictingModel_2024-09-21_08-01-49.pkl'
with open(model_filename, 'rb') as model_file:
    models = pickle.load(model_file)

# Define the unique allergens from the trained model
unique_allergens = list(models.keys())

# Load the allergen mapping from 'Allergens.csv'
def load_allergen_mapping(allergen_csv_path):
    allergen_df = pd.read_csv(allergen_csv_path, encoding='ISO-8859-1')
    allergen_df.columns = allergen_df.columns.str.strip().str.lower()
    allergen_df['food'] = allergen_df['food'].str.strip().str.lower().fillna('none')
    allergen_df['allergen'] = allergen_df['allergen'].str.strip().str.lower().fillna('none')
    allergen_mapping = allergen_df.groupby('food')['allergen'].apply(list).to_dict()
    return allergen_mapping

allergen_csv_path = 'finalAllergens.csv'
allergen_mapping = load_allergen_mapping(allergen_csv_path)

# Load the meals data from 'Meals.csv'
def load_meals(meals_csv_path):
    meals_df = pd.read_csv(meals_csv_path, encoding='ISO-8859-1')
    # Ensure all values in the 'ingredients' column are treated as strings, even if they are missing or invalid
    meals_df['ingredients'] = meals_df['ingredients'].apply(lambda x: str(x) if pd.notna(x) else '')
    meals_df['ingredients'] = meals_df['ingredients'].apply(lambda x: [i.strip().lower() for i in x.split(',')])
    return meals_df


meals_csv_path = 'finalMeals.csv'
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

# Function to predict meal safety
def predict_meal_safety_vectorized(ingredients_list, user_allergies):
    # Generate the features for all meals at once
    features_df = create_features_vectorized(ingredients_list)

    # Initialize a DataFrame to store predictions for each allergen
    predictions_df = pd.DataFrame(0, index=features_df.index, columns=unique_allergens)

    # Make batch predictions for each allergen
    for allergy in unique_allergens:
        if allergy in models:
            predictions_df[allergy] = models[allergy].predict(features_df)

    # Filter out meals that are not safe
    unsafe_mask = predictions_df[user_allergies].max(axis=1) == 1
    safe_meals = meals_df.loc[~unsafe_mask, 'recipeName']

    return safe_meals.tolist()

@app.route('/')
def index():
    # Render HTML page with input field for allergies
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_allergies = data.get('allergies', [])
    user_id = data.get('user_id', '')

    # Fetch user data from Firestore
    try:
        db = initialize_firestore()
        user_data = get_user_data_from_firestore(db, user_id)
    except Exception as e:
        return jsonify({'error': f'Error connecting to Firestore: {e}'}), 500

    if user_data:
        # Get user parameters from Firestore
        weight = user_data.get('weight', 70)  
        height = user_data.get('height', 170)  
        age = user_data.get('age', 25)  
        gender = user_data.get('gender', 'male')  
        activity_level = user_data.get('activity', 'sedentary')  
        goal = user_data.get('goal', 'maintain')  

        # Calculate daily caloric needs
        try:
            daily_calories = get_daily_calories(weight, height, age, gender, activity_level, goal)
        except ValueError as ve:
            return jsonify({'error': f'Error calculating calories: {ve}'}), 400

        # Perform the meal safety prediction using vectorized operations
        safe_meals = predict_meal_safety_vectorized(meals_df['ingredients'], user_allergies)

        # Combine the calorie count and safe meals into the response
        return jsonify({
            'daily_calories': daily_calories,
            'safe_meals': safe_meals
        })
    else:
        return jsonify({'error': 'User data not found'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
