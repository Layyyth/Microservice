# Constants for min/max values
MAX_HEIGHT = 300  # cm
MIN_HEIGHT = 70  # cm
MAX_WEIGHT = 400  # kg
MIN_WEIGHT = 30  # kg

def validate_user_data(weight, height):
    if weight < MIN_WEIGHT:
        weight = MIN_WEIGHT
    elif weight > MAX_WEIGHT:
        weight = MAX_WEIGHT

    if height < MIN_HEIGHT:
        height = MIN_HEIGHT
    elif height > MAX_HEIGHT:
        height = MAX_HEIGHT

    return weight, height

def calculate_bmr(weight, height, age, gender):
    if gender == 'male':
        return 10 * weight + 6.25 * height - 5 * age + 5
    elif gender == 'female':
        return 10 * weight + 6.25 * height - 5 * age - 161
    else:
        raise ValueError("Gender must be 'male' or 'female'")

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

def calculate_calories(tdee, goal):
    if goal == 'lose':
        return tdee - 500
    elif goal == 'maintain':
        return tdee
    elif goal == 'gain':
        return tdee + 500
    else:
        raise ValueError("Goal must be 'lose', 'maintain', or 'gain'")

def get_daily_calories(weight, height, age, gender, activity_level, goal):
    weight, height = validate_user_data(weight, height)
    bmr = calculate_bmr(weight, height, age, gender)
    tdee = calculate_tdee(bmr, activity_level)
    return calculate_calories(tdee, goal)
