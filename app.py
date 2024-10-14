import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
from caloriesLogic import get_daily_calories
from firebaseHandler import initialize_firestore, get_user_data_from_firestore

# Import functions from caloriesLogic.py
from caloriesLogic import get_daily_calories, validate_user_data

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "https://nutri-wise.vercel.app"}}, methods=["GET", "POST", "OPTIONS"])

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

# Add a root route to the app
@app.route('/')
def index():
    return "Welcome to the Diet Recommendation Microservice! Use the '/predict' endpoint to get meal recommendations."

def classify_meals(meals_df):
    # Updated dictionary of keywords for each diet preference
    dietary_keywords = {
    'vegan': [
        'meat', 'chicken', 'beef', 'pork', 'fish', 'lamb', 'eggs', 'milk', 'cheese', 'butter', 'honey',
        'bacon', 'sausage', 'gelatin', 'shrimp', 'tuna', 'salmon', 'sardines', 'anchovies', 'caviar',
        'yogurt', 'cream', 'mayo', 'whey', 'casein', 'lard', 'tallow', 'duck', 'goose', 'shellfish',
        'mozzarella cheese', 'parmesan cheese', 'cheddar cheese', 'brie cheese', 'blue cheese', 
        'gouda cheese', 'camembert cheese', 'feta cheese', 'goat cheese', 'cream cheese', 'ricotta cheese',
        'chicken breast', 'chicken thigh', 'chicken wings', 'chicken legs', 'turkey', 'beef steak', 
        'minced beef', 'ground beef', 'pork loin', 'pork belly', 'ham', 'prosciutto', 'lamb chops',
        'duck breast', 'goose liver', 'pâté', 'salmon', 'tuna', 'sardines', 'mackerel', 'trout',
        'cod', 'haddock', 'anchovies', 'halibut', 'sea bass', 'snapper', 'tilapia', 'flounder',
        'swordfish', 'catfish', 'lobster', 'crab', 'mussels', 'scallops', 'oysters', 'prawns',
        'raw', 'cooked', 'canned', 'fried', 'grilled', 'baked', 'roasted', 'boiled', 'steamed'
    ],
    'vegetarian': [
        'meat', 'chicken', 'beef', 'pork', 'fish', 'lamb', 'bacon', 'sausage', 'gelatin', 'shrimp', 
        'tuna', 'salmon', 'sardines', 'anchovies', 'caviar', 'duck', 'goose', 'shellfish',
        'chicken breast', 'chicken thigh', 'chicken wings', 'chicken legs', 'turkey', 'beef steak', 
        'minced beef', 'ground beef', 'pork loin', 'pork belly', 'ham', 'prosciutto', 'lamb chops',
        'duck breast', 'goose liver', 'pâté', 'salmon', 'tuna', 'sardines', 'mackerel', 'trout',
        'cod', 'haddock', 'anchovies', 'halibut', 'sea bass', 'snapper', 'tilapia', 'flounder',
        'swordfish', 'catfish', 'lobster', 'crab', 'mussels', 'scallops', 'oysters', 'prawns',
        'raw', 'cooked', 'canned', 'fried', 'grilled', 'baked', 'roasted', 'boiled', 'steamed'
    ],
    'keto': [
        'bread', 'pasta', 'rice', 'potato', 'sugar', 'beans', 'legumes', 'grains', 'honey', 
        'corn', 'quinoa', 'oats', 'barley', 'carrot', 'pumpkin', 'sweet potato', 'beetroot'
    ],
    'paleo': [
        'dairy', 'grains', 'legumes', 'sugar', 'processed foods', 'corn', 'rice', 'quinoa', 
        'oats', 'barley', 'peanuts', 'soy', 'tofu', 'tempeh', 'chickpeas', 'lentils'
    ],
    'gluten-free': [
        'wheat', 'barley', 'rye', 'bread', 'pasta', 'flour', 'croutons', 'bulgur', 'semolina', 
        'spelt', 'kamut', 'couscous', 'malt', 'farro', 'oats (unless certified gluten-free)'
    ],
    'dairy-free': [
        'milk', 'cheese', 'butter', 'cream', 'yogurt', 'whey', 'casein', 'ghee', 'clarified butter',
        'ice cream', 'buttermilk', 'milk powder', 'custard', 'evaporated milk', 'condensed milk',
        'mozzarella cheese', 'parmesan cheese', 'cheddar cheese', 'brie cheese', 'blue cheese', 
        'gouda cheese', 'camembert cheese', 'feta cheese', 'goat cheese', 'cream cheese', 'ricotta cheese'
    ]
}

 # Classify functions for each diet preference
    def classify_vegan(ingredients):
        return 1 if not any(kw in ingredients for kw in dietary_keywords['vegan']) else 0
    def classify_vegetarian(ingredients):
        return 1 if not any(kw in ingredients for kw in dietary_keywords['vegetarian']) else 0
    def classify_keto(ingredients):
        return 1 if not any(kw in ingredients for kw in dietary_keywords['keto']) else 0
    def classify_paleo(ingredients):
        return 1 if not any(kw in ingredients for kw in dietary_keywords['paleo']) else 0
    def classify_gluten_free(ingredients):
        return 1 if not any(kw in ingredients for kw in dietary_keywords['gluten-free']) else 0
    def classify_dairy_free(ingredients):
        return 1 if not any(kw in ingredients for kw in dietary_keywords['dairy-free']) else 0

    # Apply classifications to meals DataFrame
    meals_df['vegan'] = meals_df['ingredients'].apply(lambda ingredients: classify_vegan(ingredients))
    meals_df['vegetarian'] = meals_df['ingredients'].apply(lambda ingredients: classify_vegetarian(ingredients))
    meals_df['keto'] = meals_df['ingredients'].apply(lambda ingredients: classify_keto(ingredients))
    meals_df['paleo'] = meals_df['ingredients'].apply(lambda ingredients: classify_paleo(ingredients))
    meals_df['gluten_free'] = meals_df['ingredients'].apply(lambda ingredients: classify_gluten_free(ingredients))
    meals_df['dairy_free'] = meals_df['ingredients'].apply(lambda ingredients: classify_dairy_free(ingredients))

    return meals_df

# Load the meals data from 'Meals.csv'
def load_meals(meals_csv_path):
    meals_df = pd.read_csv(meals_csv_path, encoding='ISO-8859-1')  # Load the meals CSV
    meals_df['ingredients'] = meals_df['ingredients'].fillna('').astype(str)
    meals_df['ingredients'] = meals_df['ingredients'].apply(lambda x: [i.strip().lower() for i in x.split(',')])

    # Automatically classify meals based on the ingredients
    meals_df = classify_meals(meals_df)

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

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    # Log the request method to track incoming requests
    print(f"Request method: {request.method}")

    if request.method == 'GET':
        # For GET requests, extract 'user_id' from query parameters
        user_id = request.args.get('user_id', default='S7Hehcqz6qhhy38ZemmEg2tKPki2')  # Default user ID for testing
        print(f"GET request received with user_id: {user_id}")
    elif request.method == 'POST':
        data = request.json
        user_id = data.get('user_id')
        print(f"POST request received with user_id: {user_id}")

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

    # Perform meal safety prediction
    safe_meals = predict_meal_safety_with_diet(meals_df['ingredients'], user_allergies, diet_preference)

    # Calculate daily caloric needs
    daily_calories = get_daily_calories(weight, height, age, gender, activity_level, goal)

    return jsonify({'safe_meals': safe_meals, 'daily_calories': daily_calories})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)


'''
def classify_meals(meals_df):
    # Updated dictionary of keywords for each diet preference
    dietary_keywords = {
    'vegan': [
        'meat', 'chicken', 'beef', 'pork', 'fish', 'lamb', 'eggs', 'milk', 'cheese', 'butter', 'honey',
        'bacon', 'sausage', 'gelatin', 'shrimp', 'tuna', 'salmon', 'sardines', 'anchovies', 'caviar',
        'yogurt', 'cream', 'mayo', 'whey', 'casein', 'lard', 'tallow', 'duck', 'goose', 'shellfish',
        'mozzarella cheese', 'parmesan cheese', 'cheddar cheese', 'brie cheese', 'blue cheese', 
        'gouda cheese', 'camembert cheese', 'feta cheese', 'goat cheese', 'cream cheese', 'ricotta cheese',
        'chicken breast', 'chicken thigh', 'chicken wings', 'chicken legs', 'turkey', 'beef steak', 
        'minced beef', 'ground beef', 'pork loin', 'pork belly', 'ham', 'prosciutto', 'lamb chops',
        'duck breast', 'goose liver', 'pâté', 'salmon', 'tuna', 'sardines', 'mackerel', 'trout',
        'cod', 'haddock', 'anchovies', 'halibut', 'sea bass', 'snapper', 'tilapia', 'flounder',
        'swordfish', 'catfish', 'lobster', 'crab', 'mussels', 'scallops', 'oysters', 'prawns',
        'raw', 'cooked', 'canned', 'fried', 'grilled', 'baked', 'roasted', 'boiled', 'steamed'
    ],
    'vegetarian': [
        'meat', 'chicken', 'beef', 'pork', 'fish', 'lamb', 'bacon', 'sausage', 'gelatin', 'shrimp', 
        'tuna', 'salmon', 'sardines', 'anchovies', 'caviar', 'duck', 'goose', 'shellfish',
        'chicken breast', 'chicken thigh', 'chicken wings', 'chicken legs', 'turkey', 'beef steak', 
        'minced beef', 'ground beef', 'pork loin', 'pork belly', 'ham', 'prosciutto', 'lamb chops',
        'duck breast', 'goose liver', 'pâté', 'salmon', 'tuna', 'sardines', 'mackerel', 'trout',
        'cod', 'haddock', 'anchovies', 'halibut', 'sea bass', 'snapper', 'tilapia', 'flounder',
        'swordfish', 'catfish', 'lobster', 'crab', 'mussels', 'scallops', 'oysters', 'prawns',
        'raw', 'cooked', 'canned', 'fried', 'grilled', 'baked', 'roasted', 'boiled', 'steamed'
    ],
    'keto': [
        'bread', 'pasta', 'rice', 'potato', 'sugar', 'beans', 'legumes', 'grains', 'honey', 
        'corn', 'quinoa', 'oats', 'barley', 'carrot', 'pumpkin', 'sweet potato', 'beetroot'
    ],
    'paleo': [
        'dairy', 'grains', 'legumes', 'sugar', 'processed foods', 'corn', 'rice', 'quinoa', 
        'oats', 'barley', 'peanuts', 'soy', 'tofu', 'tempeh', 'chickpeas', 'lentils'
    ],
    'gluten-free': [
        'wheat', 'barley', 'rye', 'bread', 'pasta', 'flour', 'croutons', 'bulgur', 'semolina', 
        'spelt', 'kamut', 'couscous', 'malt', 'farro', 'oats (unless certified gluten-free)'
    ],
    'dairy-free': [
        'milk', 'cheese', 'butter', 'cream', 'yogurt', 'whey', 'casein', 'ghee', 'clarified butter',
        'ice cream', 'buttermilk', 'milk powder', 'custard', 'evaporated milk', 'condensed milk',
        'mozzarella cheese', 'parmesan cheese', 'cheddar cheese', 'brie cheese', 'blue cheese', 
        'gouda cheese', 'camembert cheese', 'feta cheese', 'goat cheese', 'cream cheese', 'ricotta cheese'
    ]
}

 # Classify functions for each diet preference
    def classify_vegan(ingredients):
        return 1 if not any(kw in ingredients for kw in dietary_keywords['vegan']) else 0
    def classify_vegetarian(ingredients):
        return 1 if not any(kw in ingredients for kw in dietary_keywords['vegetarian']) else 0
    def classify_keto(ingredients):
        return 1 if not any(kw in ingredients for kw in dietary_keywords['keto']) else 0
    def classify_paleo(ingredients):
        return 1 if not any(kw in ingredients for kw in dietary_keywords['paleo']) else 0
    def classify_gluten_free(ingredients):
        return 1 if not any(kw in ingredients for kw in dietary_keywords['gluten-free']) else 0
    def classify_dairy_free(ingredients):
        return 1 if not any(kw in ingredients for kw in dietary_keywords['dairy-free']) else 0
'''