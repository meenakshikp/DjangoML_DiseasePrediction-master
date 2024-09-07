import csv
from django.core.management.base import BaseCommand
from health_app.models import Disease, Symptom

class Command(BaseCommand):
    help = 'Import diseases and symptoms from a new structured CSV file'

    def add_arguments(self, parser):
        parser.add_argument('csv_file', type=str, help='Path to the CSV file')

    def handle(self, *args, **options):
        csv_file_path = options['csv_file']
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                # Get the prognosis (disease name)
                disease_name = row['prognosis']
                
                # Create or get the disease
                disease, created = Disease.objects.get_or_create(name=disease_name)

                # Iterate through each symptom column
                for symptom_name, presence in row.items():
                    if symptom_name != 'prognosis' and presence == '1':
                        # Create or get the symptom
                        symptom, _ = Symptom.objects.get_or_create(name=symptom_name.strip())
                        # Add the symptom to the disease
                        disease.symptoms.add(symptom)

            self.stdout.write(self.style.SUCCESS('Successfully imported diseases and symptoms'))