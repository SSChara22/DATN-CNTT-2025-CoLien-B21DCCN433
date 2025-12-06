"""
Script to save trained models and data for the web application
Run this after training models in the notebook
"""

import pandas as pd
import pickle
import os

def save_models_and_data():
    """
    Save all necessary files for the web application
    This should be run from the notebook after training
    """
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('processed_data', exist_ok=True)
    
    print("="*50)
    print("Saving Models and Data for Web Application")
    print("="*50)
    
    # Instructions
    print("\nPlease run the following code in your Jupyter notebook:")
    print("\n" + "-"*50)
    print("""
# Save models
encm_model.save('models/encm_model.h5')
lncm_model.save('models/lncm_model.h5')
neumf_model.save('models/neumf_model.h5')
bmf_model.save('models/bmf_model.h5')
print("✓ Models saved")

# Save encoded data
cars_df.to_csv('processed_data/cars_encoded.csv', index=False)
users_df.to_csv('processed_data/users_encoded.csv', index=False)
print("✓ Data saved")

# Save encoders
import pickle
with open('processed_data/user_encoder.pkl', 'wb') as f:
    pickle.dump(user_encoder, f)
with open('processed_data/item_encoder.pkl', 'wb') as f:
    pickle.dump(item_encoder, f)
print("✓ Encoders saved")

# Save context features list
with open('processed_data/context_features.pkl', 'wb') as f:
    pickle.dump(context_features_all, f)
print("✓ Context features saved")

print("\\n✓ All files saved successfully!")
print("You can now run: python app.py")
    """)
    print("-"*50)

if __name__ == '__main__':
    save_models_and_data()
