import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib

def train_model(X, y):
    try:
        # Create and fit the imputer
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        
        # Create and fit the scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        # Create and fit the label encoder
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Create and fit the model
        model = SVC(kernel='rbf', C=1.0, random_state=42)
        model.fit(X_scaled, y_encoded)

        # Save the trained model, fitted scaler, and fitted label encoder
        joblib.dump({
            'model': model,
            'scaler': scaler,
            'label_encoder': label_encoder
        }, 'trained_model.pkl')

        print("Model, scaler, and label encoder saved successfully")
        
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    # Load your data
    data = pd.read_csv('Training.csv')  # Make sure this file is in the same directory
    
    print("Columns in the dataset:", data.columns.tolist())
    
    # Assuming the last column is the target (disease) and all others are symptoms
    X = data.iloc[:, :-1]  # All columns except the last one
    y = data.iloc[:, -1]   # The last column
    print(X)
    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)
    
    # Train the model
    train_model(X, y)