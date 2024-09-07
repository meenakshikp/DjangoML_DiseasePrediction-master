from django.contrib.auth.models import AbstractUser
from django.db import models
from django.conf import settings

class CustomUser(AbstractUser):
	email = models.EmailField(unique=True)
	
class Symptom(models.Model):
    name = models.CharField(max_length=100, unique=True)
    

    def __str__(self):
        return self.name

class UserSymptomReport(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    symptoms = models.ManyToManyField(Symptom)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Report for {self.user.username} at {self.timestamp}"

class Disease(models.Model):
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(default='')
    symptoms = models.ManyToManyField(Symptom)

    def __str__(self):
        return self.name
    

    from django.db import models

class Article(models.Model):
    title = models.CharField(max_length=200)
    url = models.URLField()  # URL of the article
    description = models.TextField()
    image = models.ImageField(upload_to='images/')  # Path to the image file

    def __str__(self):
        return self.title


