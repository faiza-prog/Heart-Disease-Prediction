from django.urls import path
from predictor import views


urlpatterns = [
    path('train/', views.train_model_view, name='train_model'),
   #path('predict/', views.predict_heart_disease, name='predict_heart_disease'),  # API endpoint
   path('form/', views.show_form, name='show_form'),
    path('predict/', views.show_form, name='predict_form'),  # This should be the form handler
     
]
