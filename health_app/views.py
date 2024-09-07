import joblib
import numpy as np
import pandas as pd
import os
from .forms import CustomUserCreationForm
from django.contrib.auth.decorators import login_required
from .models import Symptom, Disease, UserSymptomReport
from django.contrib.auth import authenticate, login
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import logout
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from django.shortcuts import render, redirect, get_object_or_404
import traceback
from .util import load_model
from sklearn.svm import SVC
from .models import Article

model = None
scaler = None
label_encoder = None



def load_model():
    global model, scaler, label_encoder
    try:
        loaded_objects = joblib.load('trained_model.pkl')
        model = loaded_objects['model']
        scaler = loaded_objects['scaler']
        label_encoder = loaded_objects['label_encoder']
        
        print("Model type:", type(model))
        print("Scaler type:", type(scaler))
        print("Label encoder type:", type(label_encoder))
        
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure that 'trained_model.pkl' exists and contains 'model', 'scaler', and 'label_encoder'")
        
def create_symptom_vector(selected_symptom_ids):
    file_location = os.getcwd() + "\health_app\Training.csv"
    all_symptoms = [col for col in pd.read_csv(file_location).columns if col != 'prognosis']
    symptom_vector = [0] * 132  # Ensure exactly 132 elements
    
    for symptom_id in selected_symptom_ids:
        try:
            symptom = Symptom.objects.get(id=int(symptom_id))
            if symptom.name in all_symptoms:
                index = all_symptoms.index(symptom.name)
                if index < 132:  # Ensure we don't go out of bounds
                    symptom_vector[index] = 1
        except Symptom.DoesNotExist:
            print(f"Symptom with ID {symptom_id} does not exist.")
    
    return symptom_vector
def make_prediction(symptom_vector):
    global model, scaler, label_encoder
    
    if model is None:
        load_model()  # Ensure the model is loaded

    if len(symptom_vector) != 132:
        print(f"Error: Symptom vector has {len(symptom_vector)} features instead of 132.")
        return None

    try:
        # Ensure the symptom vector is a 2D array
        symptom_vector = np.array(symptom_vector).reshape(1, -1)
        
        # Scale the input features
        symptoms_scaled = scaler.transform(symptom_vector)
        
        # Make prediction
        prediction = model.predict(symptoms_scaled)
        
        # Decode the prediction
        decoded_prediction = label_encoder.inverse_transform(prediction)
        
        print(f"Prediction made: {decoded_prediction[0]}")  # Debugging output
        
        return decoded_prediction[0] if decoded_prediction else None
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None
    
def home(request):
    return render(request, 'home.html')


from .forms import CustomUserCreationForm

def register(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('home')
    else:
        form = CustomUserCreationForm()
    return render(request, 'register.html', {'form': form})

@login_required
def report_symptoms(request):
    if request.method == 'POST':
        selected_symptoms_ids = request.POST.getlist('symptoms[]')
        selected_symptoms = Symptom.objects.filter(id__in=selected_symptoms_ids)
        report = UserSymptomReport.objects.create(user=request.user)
        report.symptoms.set(selected_symptoms)
        return redirect('predict_disease', report_id=report.id)
    
    symptoms = Symptom.objects.all()
    return render(request, 'report_symptom.html', {'symptoms': symptoms})

@login_required
def predict_disease(request, report_id):
    report = get_object_or_404(UserSymptomReport, id=report_id)

    if request.method == 'POST':
        selected_symptom_ids = request.POST.getlist('symptoms')
        
        # Check if enough symptoms are selected
        if len(selected_symptom_ids) < 3:
            return render(request, 'insufficient_symptoms.html')

        try:
            # Create the symptom vector
            symptom_vector = create_symptom_vector(selected_symptom_ids)
            if symptom_vector is None:
                return render(request, 'error.html', {'message': "Invalid symptom vector."})

            # Call the prediction function
            predicted_disease = make_prediction(symptom_vector)

            # Render the prediction result
            return render(request, 'prediction_result.html', {
                'predicted_disease': predicted_disease,
                'selected_symptoms': Symptom.objects.filter(id__in=selected_symptom_ids)
            })
        except Exception as e:
            print(f"Error in disease prediction: {e}")
            print(traceback.format_exc())
            return render(request, 'prediction_error.html', {'error': str(e)})
    else:
        # If GET request, display the symptoms associated with the report
        symptoms = report.symptoms.all()
        return render(request, 'predict_disease.html', {'report': report, 'symptoms': symptoms})

@login_required
def view_report(request, report_id):
    report = get_object_or_404(UserSymptomReport, id=report_id)
    symptoms = report.symptoms.all()
    symptom_names = [symptom.name.strip() for symptom in symptoms]
    
    try:
        # Create symptom vector from names (assuming this matches your model's expected input)
        symptom_vector = [1 if name in symptom_names else 0 for name in symptom_names]  # Adjust this line as necessary
        predicted_disease = make_prediction(symptom_vector)
    except Exception as e:
        return render(request, 'error.html', {'message': str(e)})
    
    return render(request, 'view_report.html', {
        'report': report,
        'predicted_disease': predicted_disease
    })

@login_required
def symptom_list(request):
    user_symptoms = UserSymptomReport.objects.filter(user=request.user).order_by('-timestamp')
    
    # Debug print to check the type of user_symptoms
    print(f"Type of user_symptoms: {type(user_symptoms)}")
    
    if user_symptoms:
        # Access the user attribute on the first instance
        user = user_symptoms[0].user
        
        # Debug print to check the user attribute
        print(f"User: {user}")
    else:
        user = None
    
    return render(request, 'symptom_list.html', {'user_symptoms': user_symptoms, 'user': user})

def home(request):
    context = {
        'is_homepage': True,
    }
    return render(request, 'home.html', context)


def other_view(request):
    return render(request, 'other_template.html', {'is_homepage': False})

def user_login(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('welcome')  # Redirect to the welcome page
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})

def welcome_page(request):
    return render(request, 'welcome.html')

def terms_and_conditions(request):
    return render(request, 'terms_and_conditions.html')

def logout_view(request):
    logout(request)
    return redirect('home')

def about(request):
    return render(request, 'about.html')
