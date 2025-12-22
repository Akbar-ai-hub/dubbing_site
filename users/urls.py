from django.urls import path
from .views import RegisterView, GoogleLoginView, LoginView, PasswordResetRequestView, PasswordResetVerifyView, PasswordResetCompleteView

urlpatterns = [
    path('register/', RegisterView.as_view(), name='register'),
    path('login/', LoginView.as_view(), name='login'),
    path('google/', GoogleLoginView.as_view(), name='google-login'),
    path('password-reset/request/', PasswordResetRequestView.as_view()),
    path('password-reset/verify/', PasswordResetVerifyView.as_view()),
    path('password-reset/complete/', PasswordResetCompleteView.as_view()),
]
