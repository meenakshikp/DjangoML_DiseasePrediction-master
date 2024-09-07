from django.apps import AppConfig

class HealthAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'health_app'

    def ready(self):
        # Import and initialize your model here
        from .views import load_model
        load_model()
    pass
        
