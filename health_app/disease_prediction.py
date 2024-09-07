import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def print_data_info(X, y, label=''):
    print(f"\n{label} Data Info:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"X dtypes:\n{X.dtypes}")
    print(f"y dtype: {y.dtype}")
    print(f"Unique values in y: {y.unique().tolist()}")
    print(f"NaN values in y: {y.isna().sum()}")

def train_model(X, y):
    try:
        print_data_info(X, y, "Original")

        # Remove the 'Unnamed: 133' column if it exists
        if 'Unnamed: 133' in X.columns:
            X = X.drop('Unnamed: 133', axis=1)
            print("Removed 'Unnamed: 133' column from training data")

        # Handle NaN values in the target variable
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]

        print_data_info(X, y, "After NaN removal")

        if X.empty or y.empty:
            print("Error: After removing NaN values, the dataset is empty. Please check your data.")
            return

        # Convert y to numeric labels
        le = LabelEncoder()
        y = le.fit_transform(y)

        print_data_info(pd.DataFrame(X), pd.Series(y), "After LabelEncoder")

        # Identify numeric and categorical columns
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns

        print(f"Numeric features: {numeric_features.tolist()}")
        print(f"Categorical features: {categorical_features.tolist()}")

        # Create preprocessing steps for numeric and categorical data
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Calculate class weights
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = dict(zip(np.unique(y), class_weights))

        # Create a preprocessing and modeling pipeline
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', SVC(kernel='rbf', C=1.0, random_state=42, class_weight=class_weight_dict, probability=True))
        ])

        # Split the data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fit the pipeline
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_val)
        print("\nModel Performance:")
        print(classification_report(y_val, y_pred, target_names=le.classes_))

        # Save the trained model and label encoder
        joblib.dump({
            'model': model,
            'label_encoder': le,
            'feature_names': X.columns.tolist()
        }, 'trained_model.pkl')

        print("Model, label encoder, and feature names saved successfully")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

def load_model():
    try:
        # Load the saved model, label encoder, and feature names
        saved_data = joblib.load('trained_model.pkl')
        
        # Ensure all required components are present
        required_keys = ['model', 'label_encoder', 'feature_names']
        if not all(key in saved_data for key in required_keys):
            raise KeyError("Missing required components in the saved model file")
        
        return saved_data['model'], saved_data['label_encoder'], saved_data['feature_names']
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

def predict(model, label_encoder, feature_names, X):
    try:
        # Ensure X has the same columns as the training data
        X = X.reindex(columns=feature_names, fill_value=0)
        
        # Make prediction
        y_pred = model.predict(X)
        
        # Get prediction probabilities
        y_proba = model.predict_proba(X)
        
        # Decode the prediction
        y_pred = label_encoder.inverse_transform(y_pred)
        
        return y_pred, y_proba
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    # Training phase
    print("Starting training phase...")
    data = pd.read_csv('Training.csv')
    X = data.drop('prognosis', axis=1)  # All columns except 'prognosis'
    y = data['prognosis']  # Only the 'prognosis' column
    print("Sample of training data:")
    print(data.head())
    print("\nColumns in training data:")
    print(data.columns)
    train_model(X, y)
    print("Training completed.")

    # Prediction phase
    print("\nStarting prediction phase...")
    model, label_encoder, feature_names = load_model()
    
    if model is not None and feature_names is not None:
        # Load your test data (adjust as needed)
        test_data = pd.read_csv('Testing.csv')  # Make sure this file exists
        X_test = test_data.drop('prognosis', axis=1)  # All columns except 'prognosis'
        print("\nSample of test data:")
        print(test_data.head())
        print("\nColumns in test data:")
        print(test_data.columns)
        
        # Make predictions
        predictions, probabilities = predict(model, label_encoder, feature_names, X_test)
        
        if predictions is not None:
            print("\nPredictions:", predictions)
            print("\nPrediction distribution:")
            print(pd.Series(predictions).value_counts(normalize=True))
            
            # Print top 3 probable diseases for each prediction
            print("\nTop 3 probable diseases for each prediction:")
            for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
                top_3 = sorted(zip(label_encoder.classes_, proba), key=lambda x: x[1], reverse=True)[:3]
                print(f"Prediction {i+1}: {pred}")
                for disease, prob in top_3:
                    print(f"  - {disease}: {prob:.2f}")
                print()
        else:
            print("Failed to make predictions")
    else:
        print("Failed to load the model or feature names")