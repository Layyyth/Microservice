import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, classification_report

# Load the trained model
model_file = '/Users/layth/Documents/Developer/Diet-Recommendation-Prototype/trained-models/mealPredictingModel_2024-09-21_08-01-49.pkl'
with open(model_file, 'rb') as f:
    models = pickle.load(f)

# Load the new meals.csv data
new_meals_file = '/Users/layth/Documents/Developer/Diet-Recommendation-Prototype/testmeals.csv'
new_meals_df = pd.read_csv(new_meals_file)

# Clean the columns by stripping whitespace and making them lowercase
new_meals_df.columns = new_meals_df.columns.str.strip().str.lower()

# Cleanse the 'recipename' and 'ingredients' columns in new_meals_df
new_meals_df['recipename'] = new_meals_df['recipename'].str.strip().str.lower().fillna('none')
new_meals_df['ingredients'] = new_meals_df['ingredients'].str.strip().str.lower().fillna('none')

# Split ingredients in meals to create a list for each recipe
new_meals_df['ingredients'] = new_meals_df['ingredients'].apply(lambda x: [ingredient.strip() for ingredient in x.split(',')])

# Load allergens from the CSV file provided by the user
allergens_file_path = 'finalAllergens.csv'
allergens_df = pd.read_csv(allergens_file_path)

# Create the allergen mapping dictionary dynamically from the CSV file
allergen_mapping = allergens_df.groupby('food')['allergen'].apply(list).to_dict()

# Synonym Mapping (Extend this list for more ingredient synonyms)
synonym_mapping = {
    "groundnuts": "peanuts",
    "peanut butter": "peanuts",
    "dairy": "milk",
    "seafood": "fish",
    "prawns": "shrimp",
}

# Cross-reactivity mapping for specific allergens (e.g., latex-fruit syndrome)
cross_reactivity_mapping = {
    "bananas": ["latex allergy"],
    "avocados": ["latex allergy"],
    "shellfish": ["fish allergy"],
}

# Enhance Allergen Mapping
def enhance_allergen_mapping(ingredient):
    if ingredient in synonym_mapping:
        ingredient = synonym_mapping[ingredient]
    allergens = allergen_mapping.get(ingredient, [])
    
    if ingredient in cross_reactivity_mapping:
        allergens.extend(cross_reactivity_mapping[ingredient])
    
    return allergens

# Unique allergens (assuming this was used during model training)
unique_allergens = set(allergen for allergens in allergen_mapping.values() for allergen in allergens)

# Verify unique_allergens
print(f"Unique allergens: {unique_allergens}")

# Prepare features based on the new meals data
def create_features(ingredients):
    features = {allergen: 0 for allergen in unique_allergens}
    for ingredient in ingredients:
        enhanced_allergens = enhance_allergen_mapping(ingredient)
        for allergen in enhanced_allergens:
            if allergen in features:
                features[allergen] = 1
    return features

# Create feature matrix for the new data
X_new = pd.DataFrame(new_meals_df['ingredients'].apply(create_features).tolist())

# Ensure all feature columns that the model expects are present in X_new
# Fill missing columns with 0s to match the features seen at fit time
for column in models[list(models.keys())[0]].feature_names_in_:
    if column not in X_new.columns:
        X_new[column] = 0

# Reorder X_new columns to match the model's feature order
X_new = X_new[models[list(models.keys())[0]].feature_names_in_]

# True labels for the new data (based on the 'allergen' column)
y_true = pd.DataFrame({allergen: (new_meals_df['allergen'].str.contains(allergen)).astype(int) for allergen in unique_allergens})

# Ensure y_true has all columns, even if some allergens are not present in the test data
for allergen in unique_allergens:
    if allergen not in y_true.columns:
        y_true[allergen] = 0

# Initialize variables to store predictions and accuracies
y_pred = {}
accuracy = {}

# Test the trained model on each allergen
for allergen in unique_allergens:
    if allergen in models:
        # Get predictions from the trained model for each allergen
        y_pred[allergen] = models[allergen].predict(X_new)
        
        # Calculate the accuracy for each allergen
        accuracy[allergen] = accuracy_score(y_true[allergen], y_pred[allergen])
        print(f"Accuracy for {allergen}: {accuracy[allergen]:.2f}")
    else:
        print(f"No model available for {allergen}")

# Calculate the overall accuracy across all allergens
if accuracy:
    overall_accuracy = sum(accuracy.values()) / len(accuracy)
    print(f"\nOverall model accuracy: {overall_accuracy:.2f}")

# Optionally, print a detailed classification report for each allergen
for allergen in unique_allergens:
    if allergen in models:
        print(f"\nClassification report for {allergen}:")
        print(classification_report(y_true[allergen], y_pred[allergen], zero_division=1))
