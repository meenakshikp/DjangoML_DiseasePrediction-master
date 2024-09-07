from django.core.management.base import BaseCommand
from health_app.models import Disease, FollowUpQuestion
import openpyxl
from django.db import transaction
from django.db.utils import IntegrityError

class Command(BaseCommand):
    help = 'Import questions from Excel file'

    def add_arguments(self, parser):
        parser.add_argument('excel_file', type=str, help='Path to the Excel file')

    def handle(self, *args, **options):
        excel_file = options['excel_file']
        
        try:
            wb = openpyxl.load_workbook(excel_file)
            sheet = wb.active
            header = [cell.value for cell in sheet[1]]
            
            required_columns = ['Disease', 'Questions?', 'Yes_Probability', 'No_Probability']
            if not all(col in header for col in required_columns):
                self.stdout.write(self.style.ERROR('Error: Excel file does not contain all required columns.'))
                return

            disease_index = header.index('Disease')
            question_index = header.index('Questions?')
            yes_prob_index = header.index('Yes_Probability')
            no_prob_index = header.index('No_Probability')

            questions_imported = 0
            last_disease_name = None  # Variable to store the last disease name

            for row in sheet.iter_rows(min_row=2, values_only=True):
                disease_name = row[disease_index]
                question_text = row[question_index]
                yes_prob = row[yes_prob_index]
                no_prob = row[no_prob_index]

                # If the disease name is empty, use the last known disease name
                if not disease_name:
                    disease_name = last_disease_name
                else:
                    last_disease_name = disease_name  # Update last disease name

                if not disease_name or not question_text:
                    self.stdout.write(self.style.WARNING(f'Skipping row: Empty disease name or question'))
                    continue

                try:
                    with transaction.atomic():
                        disease, created = Disease.objects.get_or_create(name=disease_name)
                        yes_prob = self.parse_probability(yes_prob)
                        no_prob = self.parse_probability(no_prob)

                        FollowUpQuestion.objects.create(
                            disease=disease,
                            question_text=question_text,
                            yes_probability=yes_prob,
                            no_probability=no_prob
                        )

                        questions_imported += 1
                        self.stdout.write(self.style.SUCCESS(f'Imported: Disease: {disease_name}, Question: {question_text[:50]}...'))

                except IntegrityError as e:
                    self.stdout.write(self.style.ERROR(f'Error importing question for disease {disease_name}: {str(e)}'))
                except Exception as e:
                    self.stdout.write(self.style.ERROR(f'Unexpected error importing question for disease {disease_name}: {str(e)}'))

            self.stdout.write(self.style.SUCCESS(f'Successfully imported {questions_imported} questions'))

        except FileNotFoundError:
            self.stdout.write(self.style.ERROR(f'Error: File not found - {excel_file}'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error: An unexpected error occurred - {str(e)}'))

    def parse_probability(self, prob_string):
        if isinstance(prob_string, (int, float)):
            return prob_string
        elif isinstance(prob_string, str):
            cleaned_string = ''.join(filter(str.isdigit, prob_string))
            return float(cleaned_string) / 100 if cleaned_string else 0.0
        else:
            return 0.0
