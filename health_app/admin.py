from django.contrib import admin
from .models import Disease, Symptom

admin.site.register(Disease)
admin.site.register(Symptom)
#admin.site.register(FollowUpQuestion