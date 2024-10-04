import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import pickle
import datetime

# Load the datasets from CSV files
food_allergens_df = pd.read_csv('finalAllergens.csv')
meals_df = pd.read_csv('finalMeals.csv')

# Clean column names by stripping whitespace and making them lowercase
food_allergens_df.columns = food_allergens_df.columns.str.strip().str.lower()
meals_df.columns = meals_df.columns.str.strip().str.lower()

# Cleanse the 'food' and 'allergen' columns in food_allergens_df
food_allergens_df['food'] = food_allergens_df['food'].str.strip().str.lower().fillna('none')
food_allergens_df['allergen'] = food_allergens_df['allergen'].str.strip().str.lower().fillna('none')

# Cleanse the 'recipename' and 'ingredients' columns in meals_df
meals_df['recipename'] = meals_df['recipename'].str.strip().str.lower().fillna('none')
meals_df['ingredients'] = meals_df['ingredients'].str.strip().str.lower().fillna('none')

# Split ingredients in meals to create a list for each recipe
meals_df['ingredients'] = meals_df['ingredients'].apply(lambda x: [ingredient.strip() for ingredient in x.split(',')])

# Synonym Mapping (Extend this list for more ingredient synonyms)
synonym_mapping = {
    "groundnuts": "peanuts",
    "peanut butter": "peanuts",
    "dairy": "milk",
    "seafood": "fish",
    "prawns": "shrimp",
    # Add more synonyms here
}

# Add cross-reactivity mapping for specific allergens (e.g., latex-fruit syndrome)
cross_reactivity_mapping = {
    "bananas": ["latex allergy"],
    "avocados": ["latex allergy"],
    "shellfish": ["fish allergy"],
    # Add more cross-reactive foods and allergens here
}

# Update Allergen Mapping Function to include synonyms and cross-reactivity
def enhance_allergen_mapping(ingredient):
    # Check if the ingredient has a synonym
    if ingredient in synonym_mapping:
        ingredient = synonym_mapping[ingredient]
    
    allergens = allergen_mapping.get(ingredient, [])
    
    # Add cross-reactivity allergens
    if ingredient in cross_reactivity_mapping:
        allergens.extend(cross_reactivity_mapping[ingredient])
    
    return allergens

# Create a dictionary for allergen mapping (with enhancements)
allergen_mapping = food_allergens_df.groupby('food')['allergen'].apply(list).to_dict()

# Create a unique list of allergens
unique_allergens = set(allergen for allergens in allergen_mapping.values() for allergen in allergens)

# Prepare the data for Random Forest
def create_features(ingredients):
    features = {allergen: 0 for allergen in unique_allergens}
    for ingredient in ingredients:
        # Apply synonym and cross-reactivity enhancements
        enhanced_allergens = enhance_allergen_mapping(ingredient)
        for allergen in enhanced_allergens:
            if allergen in features:
                features[allergen] = 1
    return features

# Create feature and target matrices
X = pd.DataFrame(meals_df['ingredients'].apply(create_features).tolist())

# Now, create the true label columns in meals_df based on allergens
for allergen in unique_allergens:
    meals_df[allergen] = meals_df['ingredients'].apply(lambda ingredients: any([allergen in enhance_allergen_mapping(ingredient) for ingredient in ingredients]))

# Set y as the allergen columns for training
y = meals_df[list(unique_allergens)]

# Perform k-fold cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define hyperparameter grid for tuning
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [10, 20, 30, None],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split
    'min_samples_leaf': [1, 2, 4],    # Minimum number of samples required to be a leaf node
    'bootstrap': [True, False],       # Whether to use bootstrap samples when building trees
}

# Train Random Forest models for each allergen using GridSearchCV
models = {}
best_params = {}
accuracy = {}
y_pred = {}

# Train-Test Split for Accuracy Evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Loop over each allergen to apply SMOTE and train models
for allergen in unique_allergens:
    print(f"Training for allergen: {allergen}")
    
    # Check the number of samples for the current allergen
    num_samples = y_train[allergen].sum()
    
    if num_samples > 1:
        # Adjust SMOTE n_neighbors dynamically based on the number of samples
        n_neighbors = min(5, num_samples - 1)  # Ensure n_neighbors is not greater than the number of samples - 1
        smote = SMOTE(random_state=42, k_neighbors=n_neighbors)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train[allergen])
    else:
        # If too few samples, skip SMOTE and use original data
        X_train_res, y_train_res = X_train, y_train[allergen]

    # Train the RandomForestClassifier
    clf = RandomForestClassifier(random_state=42, class_weight='balanced')
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=kfold, n_jobs=-1, verbose=2)
    grid_search.fit(X_train_res, y_train_res)
    
    best_clf = grid_search.best_estimator_
    models[allergen] = best_clf
    best_params[allergen] = grid_search.best_params_

    # Predict on the test set with adjusted threshold for sensitivity
    y_pred_prob = best_clf.predict_proba(X_test)
    
    if y_pred_prob.shape[1] == 2:
        y_pred[allergen] = (y_pred_prob[:, 1] >= 0.4).astype(int)  # Probability of class 1
    else:
        y_pred[allergen] = (y_pred_prob[:, 0] < 0.6).astype(int)  # Adjust threshold for binary classification

    # Evaluate model accuracy, precision, recall, and F1-score
    accuracy[allergen] = accuracy_score(y_test[allergen], y_pred[allergen])
    precision = precision_score(y_test[allergen], y_pred[allergen], zero_division=1)
    recall = recall_score(y_test[allergen], y_pred[allergen], zero_division=1)
    f1 = f1_score(y_test[allergen], y_pred[allergen], zero_division=1)
    
    print(f"Model accuracy for {allergen}: {accuracy[allergen]:.2f}")
    print(f"Precision for {allergen}: {precision:.2f}")
    print(f"Recall for {allergen}: {recall:.2f}")
    print(f"F1-score for {allergen}: {f1:.2f}")
    print(f"Best parameters for {allergen}: {best_params[allergen]}")

# Calculate overall accuracy for all allergens
overall_accuracy = sum(accuracy.values()) / len(accuracy)
print(f"\nOverall model accuracy: {overall_accuracy * 100:.2f}%")

# Get the current date and time for naming the model file
current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Save the trained models with a dynamic name based on the current date and time
model_filename = f"/Users/layth/Documents/Developer/Diet-Recommendation-Prototype/trained-models/mealPredictingModel_{current_date}.pkl"
with open(model_filename, 'wb') as model_file:
    pickle.dump(models, model_file)
print(f"Model saved as '{model_filename}'")
