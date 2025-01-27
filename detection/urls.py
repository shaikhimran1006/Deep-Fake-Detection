from django.urls import path
from .views import detect_deepfake,home

urlpatterns = [
    path('',home,name="home"),
    path('detect/', detect_deepfake, name='detect_deepfake'),
]
