import joblib
import os
import sys
import numpy as np

# --- Configuration ---
MODEL_FILE = 'trained_house_pricing_LR_model.pkl'
INPUT_FEATURES = ['Bedrooms', 'Bathrooms']
OUTPUT_PREDICTION_NAME = 'Predicted House Price'

def load_model():
    """Loads the trained machine learning model."""
    try:
        if not os.path.exists(MODEL_FILE):
            print(f"Error: Model file '{MODEL_FILE}' not found.")
            print("Please ensure it is in the same directory as this script.")
            sys.exit(1)

        # Load the trained scikit-learn Linear Regression model
        pipeline = joblib.load(MODEL_FILE)
        print(f"Model '{MODEL_FILE}' loaded successfully.")
        return pipeline

    except Exception as e:
        print("-" * 50)
        print(f"CRITICAL ERROR: Failed to load model. Details: {e}")
        print("ACTION: This usually means a scikit-learn version mismatch.")
        print("-" * 50)
        sys.exit(1)

def collect_inputs():
    """Collects numerical inputs from the user for all required features."""
    input_data = []
    print("\n--- Enter House Features ---")
    
    for feature in INPUT_FEATURES:
        while True:
            try:
                # Prompt the user for the feature value
                value = input(f"Enter number of {feature}: ")
                
                # Convert input to float (essential for machine learning models)
                float_value = float(value)
                input_data.append(float_value)
                break
            except ValueError:
                print("Invalid input. Please enter a valid number.")
    
    # Return data shaped as [[input1, input2]]
    return np.array([input_data])

def predict_result(pipeline, data_to_predict):
    """Makes a prediction and prints the formatted result."""
    try:
        # Make prediction
        prediction = pipeline.predict(data_to_predict)[0]
        
        # Format the output as a currency, rounded to 2 decimal places
        formatted_prediction = f"${prediction:,.2f}"
        
        print("\n" + "=" * 50)
        print(f"{OUTPUT_PREDICTION_NAME}: \033[92m{formatted_prediction}\033[0m") # Green color
        print("=" * 50 + "\n")

    except Exception as e:
        print(f"\n[Prediction Error] Could not classify data. Details: {e}")

def main():
    """Main function to run the CLI tool."""
    pipeline = load_model()

    while True:
        try:
            # Collect data from user
            data = collect_inputs()
            
            # Predict and display result
            predict_result(pipeline, data)
            
            # Ask if the user wants to run another prediction
            another = input("Run another prediction? (yes/no): ")
            if another.lower() not in ('yes', 'y'):
                print("Exiting tool. Goodbye!")
                break
            
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\nExiting tool. Goodbye!")
            break
        except EOFError:
            print("\nExiting tool. Goodbye!")
            break

if __name__ == "__main__":
    main()
