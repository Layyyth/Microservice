import pandas as pd

# Load the meals data from 'Meals.csv'
meals_csv_path = './Diet-Recommendation-Prototype/finalMeals.csv'  # Make sure to use the correct path in your project
meals_df = pd.read_csv(meals_csv_path, encoding='ISO-8859-1')

# Ensure that all values in the 'ingredients' column are strings and handle missing values
meals_df['ingredients'] = meals_df['ingredients'].fillna('').astype(str)
meals_df['ingredients'] = meals_df['ingredients'].apply(lambda x: [i.strip().lower() for i in x.split(',')])

# Define detailed rules for each diet based on ingredients, now including broths, raw/cooked versions, canned, and more

# Non-Vegan Ingredients (Expanded to include raw, cooked, canned versions)
non_vegan_ingredients = [
    # Meat & Poultry (Raw/Cooked/Canned)
    'beef', 'raw beef', 'cooked beef', 'canned beef', 'pork', 'raw pork', 'cooked pork', 'bacon', 'ham', 'lamb', 
    'veal', 'duck', 'venison', 'goat', 'rabbit', 'chicken', 'raw chicken','cooked chicken', 'canned chicken', 
    'turkey', 'raw turkey', 'cooked turkey', 'quail', 'pheasant', 'sausages', 'salami', 'hot dog', 'bologna',

    # Fish & Seafood (Including raw/cooked/canned and broths)
    'fish', 'raw fish', 'cooked fish', 'fish broth', 'canned fish', 'salmon', 'raw salmon', 'cooked salmon', 
    'canned salmon', 'tuna', 'raw tuna', 'cooked tuna', 'canned tuna', 'swordfish', 'cod', 'haddock', 'mackerel', 
    'sardines', 'anchovies', 'trout', 'halibut', 'bass', 'tilapia', 'flounder', 'catfish', 'snapper', 'mahimahi', 
    'grouper', 'shrimp', 'raw shrimp', 'cooked shrimp', 'canned shrimp', 'prawns', 'lobster', 'crab', 'scallops', 
    'mussels', 'oysters', 'clams', 'octopus', 'squid', 'caviar', 'fish sauce', 'dashi',

    # Eggs
    'egg', 'eggs', 'egg yolk', 'egg white', 'quail eggs', 'omelette', 'frittata', 'scrambled eggs',

    # Dairy (Milk, Cheese, Butter, Yogurt)
    'milk', 'whole milk', 'skim milk', 'buttermilk', 'cream', 'heavy cream', 'half and half', 'evaporated milk', 
    'condensed milk', 'cheese', 'parmesan cheese', 'cheddar cheese', 'mozzarella cheese', 'brie', 'feta', 
    'goat cheese', 'blue cheese', 'gorgonzola', 'ricotta', 'cream cheese', 'colby jack', 'camembert', 'pecorino', 
    'halloumi', 'paneer', 'swiss cheese', 'monterey jack', 'provolone', 'asiago', 'stilton', 'butter', 'ghee', 
    'yogurt', 'greek yogurt', 'sour cream', 'clotted cream',

    # Animal-based Sauces & Broths
    'chicken broth', 'beef broth', 'pork broth', 'bone broth', 'veal broth', 'lamb broth', 'fish broth', 
    'anchovy paste', 'mayonnaise', 'honey'
]

# Non-Keto Ingredients (Expanded with more variations of sugars, starchy foods, grains, and canned/processed versions)
non_keto_ingredients = [
    # Sugars & Sweeteners
    'sugar', 'brown sugar', 'cane sugar', 'powdered sugar', 'honey', 'maple syrup', 'agave syrup', 'corn syrup', 
    'molasses', 'date syrup', 'coconut sugar', 'glucose', 'fructose', 'dextrose', 'maltose',
    
    # Grains & Grain Products (Bread, Pasta, Rice, Canned/Processed)
    'bread', 'white bread', 'whole wheat bread', 'bagels', 'croissants', 'pita', 'naan', 'ciabatta', 'focaccia', 
    'brioche', 'sourdough', 'rolls', 'biscuits', 'crackers', 'cornbread',
    'pasta', 'spaghetti', 'macaroni', 'fettuccine', 'lasagna', 'penne', 'ravioli', 'tortellini', 'noodles', 
    'udon', 'soba', 'ramen', 'vermicelli', 'rigatoni', 'lo mein', 'rice noodles', 'glass noodles',
    'rice', 'white rice', 'brown rice', 'wild rice', 'jasmine rice', 'basmati rice', 'sushi rice', 'sticky rice',
    
    # Starches & High-Carb Vegetables (Raw/Cooked/Canned)
    'potato', 'raw potato', 'cooked potato', 'canned potato', 'sweet potato', 'yam', 'french fries', 'hash browns', 
    'tater tots', 'mashed potatoes', 'corn', 'raw corn', 'cooked corn', 'canned corn', 'cornmeal', 'polenta', 
    'popcorn', 'tortillas', 'quinoa', 'millet', 'barley', 'farro', 'oats', 'oatmeal', 'bulgur wheat', 'couscous', 
    'buckwheat',

    # Legumes & Beans (Raw/Cooked/Canned)
    'lentils', 'chickpeas', 'beans', 'black beans', 'kidney beans', 'pinto beans', 'navy beans', 'white beans', 
    'lima beans', 'split peas', 'green peas', 'canned beans'
]

# Non-Gluten-Free Ingredients (Detailed with specific grains, flours, broths, canned/processed variations)
non_glutenfree_ingredients = [
    # Grains & Flours (Wheat, Barley, Rye)
    'wheat', 'whole wheat', 'bread flour', 'all-purpose flour', 'self-raising flour', 'semolina', 'durum wheat', 
    'barley', 'rye', 'spelt', 'farro', 'triticale', 'bulgur', 'kamut', 'einkorn',
    
    # Gluten-containing products (Bread, Pasta, Baked Goods)
    'bread', 'white bread', 'whole wheat bread', 'bagels', 'croissants', 'pita', 'naan', 'ciabatta', 
    'pasta', 'spaghetti', 'macaroni', 'fettuccine', 'lasagna', 'penne', 'ravioli', 'tortellini',
    'pizza dough', 'baguette', 'rolls', 'muffins', 'cakes', 'cookies', 'pastries', 'waffles', 'pancakes',

    # Broths & Sauces (Gluten-based)
    'wheat flour broth', 'soy sauce', 'miso broth', 'barley broth', 'canned barley soup'
]

# Non-Dairy-Free Ingredients (Expanded with raw/cooked/canned versions of dairy products)
non_dairyfree_ingredients = [
    # Milk & Cream (Raw/Cooked/Canned)
    'milk', 'whole milk', 'skim milk', 'buttermilk', 'evaporated milk', 'condensed milk', 'cream', 'heavy cream', 
    'half and half', 'whipping cream', 'clotted cream', 'raw milk', 'pasteurized milk', 'yogurt', 'greek yogurt', 
    'kefir', 'sour cream', 'ice cream', 'custard', 'whipped cream',

    # Cheeses (Raw/Cooked/Canned)
    'cheese', 'parmesan cheese', 'cheddar cheese', 'mozzarella cheese', 'brie', 'feta', 'goat cheese', 
    'blue cheese', 'gorgonzola', 'ricotta', 'cream cheese', 'colby jack', 'camembert', 'pecorino', 
    'halloumi', 'paneer', 'swiss cheese', 'monterey jack', 'provolone', 'asiago', 'stilton', 'canned cheese',

    # Butter & Dairy-based Spreads
    'butter', 'ghee', 'margarine', 'cream cheese', 'buttermilk'
]
# Non-Vegetarian Ingredients (Meat, Poultry, Fish, Broths)
non_vegetarian_ingredients = [
    # Meat & Poultry (Raw/Cooked/Canned)
    'beef', 'raw beef', 'cooked beef', 'canned beef', 'pork', 'raw pork', 'cooked pork', 'bacon', 'ham', 'lamb', 
    'veal', 'duck', 'venison', 'goat', 'rabbit', 'chicken', 'raw chicken', 'cooked chicken', 'canned chicken', 
    'turkey', 'raw turkey', 'cooked turkey', 'quail', 'pheasant', 'sausages', 'salami', 'hot dog', 'bologna',
    
    # Fish & Seafood (Including raw/cooked/canned and broths)
    'fish', 'raw fish', 'cooked fish', 'fish broth', 'canned fish', 'salmon', 'raw salmon', 'cooked salmon', 
    'canned salmon', 'tuna', 'raw tuna', 'cooked tuna', 'canned tuna', 'swordfish', 'cod', 'haddock', 'mackerel', 
    'sardines', 'anchovies', 'trout', 'halibut', 'bass', 'tilapia', 'flounder', 'catfish', 'snapper', 'mahimahi', 
    'grouper', 'shrimp', 'raw shrimp', 'cooked shrimp', 'canned shrimp', 'prawns', 'lobster', 'crab', 'scallops', 
    'mussels', 'oysters', 'clams', 'octopus', 'squid', 'caviar', 'fish sauce', 'dashi',

    # Animal-based broths
    'chicken broth', 'beef broth', 'pork broth', 'bone broth', 'veal broth', 'lamb broth', 'fish broth'
]
# Non-Paleo Ingredients (Grains, Dairy, Processed Foods, Legumes, Sugars)
non_paleo_ingredients = [
    # Grains & Grain Products (Bread, Pasta, Rice)
    'bread', 'white bread', 'whole wheat bread', 'bagels', 'croissants', 'pita', 'naan', 'ciabatta', 'focaccia', 
    'brioche', 'sourdough', 'rolls', 'biscuits', 'crackers', 'cornbread', 'pasta', 'spaghetti', 'macaroni', 
    'fettuccine', 'lasagna', 'penne', 'ravioli', 'tortellini', 'noodles', 'udon', 'soba', 'ramen', 'vermicelli', 
    'rigatoni', 'lo mein', 'rice noodles', 'glass noodles', 'rice', 'white rice', 'brown rice', 'wild rice', 
    'jasmine rice', 'basmati rice', 'sushi rice', 'sticky rice', 'quinoa', 'millet', 'barley', 'farro', 'oats',
    
    # Dairy Products
    'milk', 'whole milk', 'skim milk', 'buttermilk', 'evaporated milk', 'condensed milk', 'cream', 'heavy cream', 
    'half and half', 'yogurt', 'greek yogurt', 'cheese', 'parmesan cheese', 'cheddar cheese', 'mozzarella cheese', 
    'butter', 'ghee', 'ice cream', 'custard', 'whipped cream',

    # Legumes & Beans
    'lentils', 'chickpeas', 'black beans', 'kidney beans', 'pinto beans', 'navy beans', 'white beans', 
    'lima beans', 'split peas', 'green peas', 'soybeans', 'tofu', 'edamame',

    # Sugars & Processed Sweeteners
    'sugar', 'brown sugar', 'cane sugar', 'powdered sugar', 'honey', 'maple syrup', 'agave syrup', 'corn syrup', 
    'molasses', 'date syrup', 'coconut sugar', 'glucose', 'fructose', 'dextrose', 'maltose', 'processed foods'
]


# Add columns for various diet preferences
diet_columns = ['vegan', 'keto', 'gluten-free', 'dairy-free','paleo','vegetarian']
for column in diet_columns:
    meals_df[column] = 0

# Function to classify meals based on ingredients
def classify_meals_by_diet(ingredients):
    ingredients = [i.lower().strip() for i in ingredients]
    
    # Classify Vegan
    vegan = all(ingredient not in non_vegan_ingredients for ingredient in ingredients)
    
    # Classify Keto
    keto = all(ingredient not in non_keto_ingredients for ingredient in ingredients)
    
    # Classify Gluten-Free
    gluten_free = all(ingredient not in non_glutenfree_ingredients for ingredient in ingredients)
    
    # Classify Dairy-Free
    dairy_free = all(ingredient not in non_dairyfree_ingredients for ingredient in ingredients)

    paleo = all(ingredient not in non_paleo_ingredients for ingredient in ingredients)

    vegetarian = all(ingredient not in non_vegetarian_ingredients for ingredient in ingredients)
    
    # Return the classification
    return {
        'vegan': int(vegan),
        'keto': int(keto),
        'gluten-free': int(gluten_free),
        'dairy-free': int(dairy_free),
        'paleo':int(paleo),
        'vegetarian':int(vegetarian)
    }

# Apply the classification to each row in the dataframe
for index, row in meals_df.iterrows():
    diet_classifications = classify_meals_by_diet(row['ingredients'])
    for diet, value in diet_classifications.items():
        meals_df.at[index, diet] = value

# Save the updated dataframe
updated_meals_file_path = 'finalMeals_with_diet_classification.csv'
meals_df.to_csv(updated_meals_file_path, index=False, encoding='ISO-8859-1')

print(f"Meals classified and saved to {updated_meals_file_path}")
