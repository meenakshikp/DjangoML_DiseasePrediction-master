from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('health_app.urls')),  # This line includes your health app's URLs
]