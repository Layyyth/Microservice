from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained models (update with your actual model file path)
model_filename = '/Users/layth/Documents/Developer/Diet-Recommendation-Prototype/trained-models/mealPredictingModel_2024-09-13_10-55-51.pkl'
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

allergen_csv_path = '/Users/layth/Documents/Developer/Diet-Recommendation-Prototype/Allergens.csv'
allergen_mapping = load_allergen_mapping(allergen_csv_path)

# Load the meals data from 'Meals.csv'
def load_meals(meals_csv_path):
    meals_df = pd.read_csv(meals_csv_path, encoding='ISO-8859-1')  # Specify encoding to avoid UnicodeDecodeError
    meals_df['ingredients'] = meals_df['ingredients'].apply(lambda x: [i.strip().lower() for i in x.split(',')])
    return meals_df

meals_csv_path = '/Users/layth/Documents/Developer/Diet-Recommendation-Prototype/Meals.csv'
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
    user_allergies = request.json['allergies']
    
    # Perform the meal safety prediction using vectorized operations
    safe_meals = predict_meal_safety_vectorized(meals_df['ingredients'], user_allergies)
    
    return jsonify({'safe_meals': safe_meals})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
