import firebase_admin
from firebase_admin import credentials, firestore
import os
import json

# Initialize Firebase Firestore using service account credentials from environment variable
def initialize_firestore():
    # Check if Firebase is already initialized
    if not firebase_admin._apps:
        # Get Firebase credentials from environment variable
        firebase_creds = os.environ.get('FIREBASE_CREDENTIALS')

        if firebase_creds:
            # Parse the JSON string from the environment variable
            cred_dict = json.loads(firebase_creds)
            
            # Initialize Firebase app with credentials
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
        else:
            raise ValueError("Firebase credentials not found in environment variables")
    
    return firestore.client()

# Function to fetch user data from Firestore
def get_user_data_from_firestore(db, user_id):
    try:
        # Fetch user document from Firestore
        user_ref = db.collection('accounts').document(user_id)
        user_data = user_ref.get().to_dict()
        if user_data:
            return user_data.get('NutriInfo')  # Fetch NutriInfo subfield
        else:
            return None
    except Exception as e:
        print(f"Error fetching user data: {e}")
        return None
