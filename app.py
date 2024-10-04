from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Mock functions for testing (replace with actual implementations)
def get_daily_calories(weight, height, age, gender, activity_level, goal):
    # Simple BMR calculation (Mifflin-St Jeor Equation)
    if gender.lower() == 'male':
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    # Adjust for activity level (simplified)
    activity_factors = {'sedentary': 1.2, 'light': 1.375, 'moderate': 1.55, 'active': 1.725}
    bmr *= activity_factors.get(activity_level.lower(), 1.2)
    # Adjust for goal
    if goal.lower() == 'lose':
        bmr -= 500
    elif goal.lower() == 'gain':
        bmr += 500
    return int(bmr)

def initialize_firestore():
    # Mock Firestore initialization
    return None

def get_user_data_from_firestore(db, user_id):
    # Mock user data retrieval
    return {
        'weight': 70,
        'height': 170,
        'age': 25,
        'gender': 'male',
        'activity': 'moderate',
        'goal': 'maintain'
    }

app = Flask(__name__)

# Load the trained models
model_filename = 'mealPredictingModel_2024-09-21_08-01-49.pkl'
try:
    with open(model_filename, 'rb') as model_file:
        models = pickle.load(model_file)
except Exception as e:
    logging.error(f"Error loading model: {e}")
    models = {}

# Define the unique allergens from the trained model
unique_allergens = list(models.keys())

# Load the allergen mapping
def load_allergen_mapping(allergen_csv_path):
    try:
        allergen_df = pd.read_csv(allergen_csv_path, encoding='ISO-8859-1')
        allergen_df.columns = allergen_df.columns.str.strip().str.lower()
        allergen_df['food'] = allergen_df['food'].str.strip().str.lower().fillna('none')
        allergen_df['allergen'] = allergen_df['allergen'].str.strip().str.lower().fillna('none')
        allergen_mapping = allergen_df.groupby('food')['allergen'].apply(list).to_dict()
        return allergen_mapping
    except Exception as e:
        logging.error(f"Error loading allergen mapping: {e}")
        return {}

allergen_csv_path = 'finalAllergens.csv'
allergen_mapping = load_allergen_mapping(allergen_csv_path)

# Load meals
def load_meals(csv_path):
    try:
        meals_df = pd.read_csv(csv_path)
        meals_df['ingredients'] = meals_df['ingredients'].fillna("").astype(str)
        meals_df['ingredients'] = meals_df['ingredients'].apply(lambda x: [i.strip().lower() for i in x.split(',') if i.strip()])
        return meals_df
    except Exception as e:
        logging.error(f"Error loading meals: {e}")
        return pd.DataFrame()

meals_csv_path = 'finalMeals.csv'
meals_df = load_meals(meals_csv_path)

# Function to create features for ingredients
def create_features_vectorized(ingredients_list):
    features_df = pd.DataFrame(0, index=ingredients_list.index, columns=unique_allergens)
    for ingredient, allergens in allergen_mapping.items():
        ingredients_mask = ingredients_list.apply(lambda ingredients: ingredient in ingredients)
        for allergen in allergens:
            if allergen in features_df.columns:
                features_df.loc[ingredients_mask, allergen] = 1
    return features_df

# Function to predict meal safety
def predict_meal_safety_vectorized(ingredients_list, user_allergies):
    features_df = create_features_vectorized(ingredients_list)
    predictions_df = pd.DataFrame(0, index=features_df.index, columns=unique_allergens)
    for allergen in unique_allergens:
        if allergen in models:
            try:
                predictions = models[allergen].predict(features_df)
                predictions_df[allergen] = predictions
            except Exception as e:
                logging.error(f"Error predicting for allergen '{allergen}': {e}")
    unsafe_mask = predictions_df[user_allergies].max(axis=1) == 1
    safe_meals = meals_df.loc[~unsafe_mask, 'recipeName']
    return safe_meals.tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_allergies = data.get('allergies', [])
    user_id = data.get('user_id', '')
    try:
        db = initialize_firestore()
        user_data = get_user_data_from_firestore(db, user_id)
    except Exception as e:
        logging.error(f'Error connecting to Firestore: {e}')
        return jsonify({'error': f'Error connecting to Firestore: {e}'}), 500
    if user_data:
        weight = user_data.get('weight', 70)
        height = user_data.get('height', 170)
        age = user_data.get('age', 25)
        gender = user_data.get('gender', 'male')
        activity_level = user_data.get('activity', 'sedentary')
        goal = user_data.get('goal', 'maintain')
        try:
            daily_calories = get_daily_calories(weight, height, age, gender, activity_level, goal)
        except ValueError as ve:
            logging.error(f'Error calculating calories: {ve}')
            return jsonify({'error': f'Error calculating calories: {ve}'}), 400
        safe_meals = predict_meal_safety_vectorized(meals_df['ingredients'], user_allergies)
        return jsonify({
            'daily_calories': daily_calories,
            'safe_meals': safe_meals
        })
    else:
        logging.error('User data not found')
        return jsonify({'error': 'User data not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
