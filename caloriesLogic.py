import os
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Constants for min/max values
MAX_HEIGHT = 300  # cm
MIN_HEIGHT = 70  # cm
MAX_WEIGHT = 400  # kg
MIN_WEIGHT = 30  # kg

# Initialize Firebase Firestore using service account credentials
def initialize_firestore():
    # Get the path to the credentials file from an environment variable
    firebase_cred_path = os.getenv('FIREBASE_CRED_PATH')
    
    if not firebase_cred_path:
        raise Exception("Firebase credentials path is not set in the environment variables")

    cred = credentials.Certificate(firebase_cred_path)
    firebase_admin.initialize_app(cred)
    return firestore.client()

# Function to fetch user data from Firestore
def get_user_data_from_firestore(db, user_id):
    try:
        user_ref = db.collection('accounts').document(user_id)
        user_data = user_ref.get().to_dict()
        if user_data:
            return user_data.get('NutriInfo')  # Fetch the NutriInfo subfield
        else:
            return None
    except Exception as e:
        print(f"Error fetching user data: {e}")
        return None

# Function to validate and correct height and weight
def validate_user_data(weight, height):
    # Validate and correct weight
    if weight < MIN_WEIGHT:
        print(f"Correcting weight: {weight} is too low, setting to {MIN_WEIGHT}")
        weight = MIN_WEIGHT
    elif weight > MAX_WEIGHT:
        print(f"Correcting weight: {weight} is too high, setting to {MAX_WEIGHT}")
        weight = MAX_WEIGHT

    # Validate and correct height
    if height < MIN_HEIGHT:
        print(f"Correcting height: {height} is too low, setting to {MIN_HEIGHT}")
        height = MIN_HEIGHT
    elif height > MAX_HEIGHT:
        print(f"Correcting height: {height} is too high, setting to {MAX_HEIGHT}")
        height = MAX_HEIGHT

    return weight, height

# Function to calculate Basal Metabolic Rate (BMR)
def calculate_bmr(weight, height, age, gender):
    if gender == 'male':
        return 10 * weight + 6.25 * height - 5 * age + 5
    elif gender == 'female':
        return 10 * weight + 6.25 * height - 5 * age - 161
    else:
        raise ValueError("Gender must be 'male' or 'female'")

# Function to calculate Total Daily Energy Expenditure (TDEE) based on activity level
def calculate_tdee(bmr, activity_level):
    activity_factors = {
        'sedentary': 1.2,
        'lightly_active': 1.375,
        'moderately_active': 1.55,
        'very_active': 1.725,
        'super_active': 1.9
    }
    if activity_level not in activity_factors:
        raise ValueError("Invalid activity level. Choose from: 'sedentary', 'lightly_active', 'moderately_active', 'very_active', 'super_active'")
    return bmr * activity_factors[activity_level]

# Function to adjust caloric needs based on user goal
def calculate_calories(tdee, goal):
    if goal == 'lose':
        return tdee - 500
    elif goal == 'maintain':
        return tdee
    elif goal == 'gain':
        return tdee + 500
    else:
        raise ValueError("Goal must be 'lose', 'maintain', or 'gain'")

# Main function to compute daily caloric needs
def get_daily_calories(weight, height, age, gender, activity_level, goal):
    weight, height = validate_user_data(weight, height)  # Validate weight and height
    bmr = calculate_bmr(weight, height, age, gender)
    tdee = calculate_tdee(bmr, activity_level)
    return calculate_calories(tdee, goal)


