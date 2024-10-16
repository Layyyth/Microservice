import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
from caloriesLogic import get_daily_calories
from firebaseHandler import initialize_firestore, get_user_data_from_firestore

app = Flask(__name__)

# Specify the allowed origins for CORS
origins = [
    'https://nutri-wise.vercel.app',
    'https://nutri-wise-lq7zew6rf-layyyths-projects.vercel.app'
    'https://a0ab-94-73-21-32.ngrok-free.app'
]

# Configure CORS for the /predict endpoint
CORS(app, resources={r"/predict": {"origins": origins, "methods": ["GET", "POST", "OPTIONS"]}})

# Load the trained models
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

# Add a root route to the app
@app.route('/')
def index():
    return "Welcome to the Diet Recommendation Microservice! Use the '/predict' endpoint to get meal recommendations."

# Define dietary keywords for classification
dietary_keywords = {
    'vegan': [
        'meat', 'chicken', 'beef', 'pork', 'fish', 'lamb', 'eggs', 'milk', 'cheese', 'butter', 'honey',
        'bacon', 'sausage', 'gelatin', 'shrimp', 'tuna', 'salmon', 'sardines', 'anchovies', 'caviar',
        'yogurt', 'cream', 'mayo', 'whey', 'casein', 'lard', 'tallow', 'duck', 'goose', 'shellfish',
        'mozzarella cheese', 'parmesan cheese', 'cheddar cheese', 'brie cheese', 'blue cheese',
        'gouda cheese', 'camembert cheese', 'feta cheese', 'goat cheese', 'cream cheese', 'ricotta cheese',
        'chicken breast', 'chicken thigh', 'chicken wings', 'chicken legs', 'turkey', 'beef steak',
        'minced beef', 'ground beef', 'pork loin', 'pork belly', 'ham', 'prosciutto', 'lamb chops',
        'duck breast', 'goose liver', 'pâté', 'mackerel', 'trout', 'cod', 'haddock', 'halibut',
        'sea bass', 'snapper', 'tilapia', 'flounder', 'swordfish', 'catfish', 'lobster', 'crab',
        'mussels', 'scallops', 'oysters', 'prawns'
    ],
    'vegetarian': [
        'meat', 'chicken', 'beef', 'pork', 'fish', 'lamb', 'bacon', 'sausage', 'gelatin', 'shrimp',
        'tuna', 'salmon', 'sardines', 'anchovies', 'caviar', 'duck', 'goose', 'shellfish',
        'chicken breast', 'chicken thigh', 'chicken wings', 'chicken legs', 'turkey', 'beef steak',
        'minced beef', 'ground beef', 'pork loin', 'pork belly', 'ham', 'prosciutto', 'lamb chops',
        'duck breast', 'goose liver', 'pâté', 'mackerel', 'trout', 'cod', 'haddock', 'halibut',
        'sea bass', 'snapper', 'tilapia', 'flounder', 'swordfish', 'catfish', 'lobster', 'crab',
        'mussels', 'scallops', 'oysters', 'prawns'
    ],
    'keto': [
        'bread', 'pasta', 'rice', 'potato', 'sugar', 'beans', 'legumes', 'grains', 'honey',
        'corn', 'quinoa', 'oats', 'barley', 'carrot', 'pumpkin', 'sweet potato', 'beetroot'
    ],
    'paleo': [
        'dairy', 'grains', 'legumes', 'sugar', 'processed foods', 'corn', 'rice', 'quinoa',
        'oats', 'barley', 'peanuts', 'soy', 'tofu', 'tempeh', 'chickpeas', 'lentils'
    ],
    'gluten_free': [
        'wheat', 'barley', 'rye', 'bread', 'pasta', 'flour', 'croutons', 'bulgur', 'semolina',
        'spelt', 'kamut', 'couscous', 'malt', 'farro', 'oats (unless certified gluten-free)'
    ],
    'dairy_free': [
        'milk', 'cheese', 'butter', 'cream', 'yogurt', 'whey', 'casein', 'ghee', 'clarified butter',
        'ice cream', 'buttermilk', 'milk powder', 'custard', 'evaporated milk', 'condensed milk',
        'mozzarella cheese', 'parmesan cheese', 'cheddar cheese', 'brie cheese', 'blue cheese',
        'gouda cheese', 'camembert cheese', 'feta cheese', 'goat cheese', 'cream cheese', 'ricotta cheese'
    ]
}

# Define meal time keywords for classification
meal_time_keywords = {
    'breakfast': [
        'egg', 'pancake', 'waffle', 'toast', 'cereal', 'smoothie', 'oatmeal', 'yogurt',
        'bagel', 'granola', 'omelette', 'muffin', 'bacon', 'sausage', 'frittata', 'scramble',
        'hash brown', 'quiche', 'crepe', 'french toast', 'porridge', 'breakfast', 'brunch'
    ],
    'lunch': [
        'sandwich', 'salad', 'wrap', 'burger', 'soup', 'panini', 'burrito', 'taco', 'pita',
        'noodle', 'rice bowl', 'quesadilla', 'sub', 'hoagie', 'gyro', 'lunch', 'midday'
    ],
    'dinner': [
        'pasta', 'steak', 'curry', 'stir fry', 'roast', 'casserole', 'lasagna', 'pizza',
        'risotto', 'grill', 'barbecue', 'bbq', 'shepherd\'s pie', 'meatloaf', 'dinner',
        'evening', 'supper', 'entree', 'main course'
    ]
}

# Function to classify meals based on dietary preferences
def classify_meals(meals_df):
    # Classify functions for each diet preference
    def classify_diet(ingredients, diet):
        return 1 if not any(kw in ingredients for kw in dietary_keywords[diet]) else 0

    # Apply classifications to meals DataFrame
    for diet in dietary_keywords.keys():
        meals_df[diet] = meals_df['ingredients'].apply(lambda ingredients: classify_diet(ingredients, diet))

    return meals_df

# Function to classify meals based on meal times
def classify_meal_times(meals_df):
    # Initialize columns for meal times
    for meal_time in meal_time_keywords.keys():
        meals_df[meal_time] = 0

    # Function to classify a single meal
    def classify_meal(row):
        # Combine recipe name and ingredients into one text
        text = row['recipeName'].lower() + ' ' + ' '.join(row['ingredients'])
        for meal_time, keywords in meal_time_keywords.items():
            if any(keyword in text for keyword in keywords):
                row[meal_time] = 1
        return row

    # Apply the classification to each meal
    meals_df = meals_df.apply(classify_meal, axis=1)
    return meals_df

# Load the meals data from 'finalMeals.csv'
def load_meals(meals_csv_path):
    meals_df = pd.read_csv(meals_csv_path, encoding='ISO-8859-1')
    meals_df['ingredients'] = meals_df['ingredients'].fillna('').astype(str)
    meals_df['ingredients'] = meals_df['ingredients'].apply(lambda x: [i.strip().lower() for i in x.split(',')])

    # Automatically classify meals based on the ingredients
    meals_df = classify_meals(meals_df)

    # Classify meals based on meal times
    meals_df = classify_meal_times(meals_df)

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

# Function to predict meal safety based on allergies and dietary preferences
def predict_meal_safety_with_diet(ingredients_list, user_allergies, diet_preference, meal_time=None):
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

    # Apply meal time filter
    if meal_time and meal_time in meal_time_keywords:
        safe_meals = safe_meals[safe_meals[meal_time] == 1]

    # Prepare the response data
    safe_meals_list = safe_meals['recipeName'].tolist()

    return safe_meals_list

# Define the /predict endpoint
@app.route('/predict', methods=['GET', 'POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 200  # Simply return a 200 OK response for preflight requests

    if request.method == 'GET':
        # For GET requests, extract 'user_id' and 'meal_time' from query parameters
        user_id = request.args.get('user_id', default='S7Hehcqz6qhhy38ZemmEg2tKPki2')  # Default user ID for testing
        meal_time = request.args.get('meal_time', default=None)
        print(f"GET request received with user_id: {user_id}, meal_time: {meal_time}")

    elif request.method == 'POST':
        # For POST requests, extract 'user_id' and 'meal_time' from JSON body
        data = request.json
        user_id = data.get('user_id')
        meal_time = data.get('meal_time')
        print(f"POST request received with user_id: {user_id}, meal_time: {meal_time}")

    # Normalize and validate meal_time
    if meal_time:
        meal_time = meal_time.lower()
        if meal_time not in meal_time_keywords:
            return jsonify({"error": f"Invalid meal_time '{meal_time}'. Valid options are 'breakfast', 'lunch', or 'dinner'."}), 400

    # Initialize Firestore and fetch user data
    db = initialize_firestore()
    user_data = get_user_data_from_firestore(db, user_id)

    # Debugging: print fetched user data to ensure it exists
    print(f"Fetched user data: {user_data}")

    # Check if user_data is None (i.e., if the user was not found in Firestore)
    if not user_data:
        return jsonify({"error": "User data not found"}), 404

    # Extract relevant NutriInfo fields
    user_allergies = user_data.get('allergies', [])
    diet_preference = user_data.get('diet', 'none')
    weight = user_data.get('weight')
    height = user_data.get('height')
    age = user_data.get('age')
    gender = user_data.get('gender')
    activity_level = user_data.get('activity')
    goal = user_data.get('goal')

    # Perform meal safety prediction with meal_time
    safe_meals = predict_meal_safety_with_diet(
        meals_df['ingredients'],
        user_allergies,
        diet_preference,
        meal_time
    )

    # Calculate daily caloric needs
    daily_calories = get_daily_calories(weight, height, age, gender, activity_level, goal)

    # Return the actual response with predicted meals and calorie needs
    return jsonify({'safe_meals': safe_meals, 'daily_calories': daily_calories})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5001)), debug=True)
