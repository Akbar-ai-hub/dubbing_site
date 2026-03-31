from django.urls import path
from .views import (
    RegisterView,
    GoogleLoginView,
    CurrentUserView,
    ProfileUpdateView,
    LoginView,
    PasswordResetRequestView,
    PasswordResetVerifyView,
    PasswordResetCompleteView,
    LogoutView,
    WalletBalanceView,
    WalletTopUpView,
    BillingHistoryView,
)

urlpatterns = [
    path('register/', RegisterView.as_view(), name='register'),
    path('login/', LoginView.as_view(), name='login'),
    path('google/', GoogleLoginView.as_view(), name='google-login'),
    path("me/", CurrentUserView.as_view(), name="current-user"),
    path("profile/", ProfileUpdateView.as_view(), name="profile-update"),
    path('logout/', LogoutView.as_view(), name='logout'),
    path('password-reset/request/', PasswordResetRequestView.as_view()),
    path('password-reset/verify/', PasswordResetVerifyView.as_view()),
    path('password-reset/complete/', PasswordResetCompleteView.as_view()),
    path('wallet/', WalletBalanceView.as_view(), name='wallet-balance'),
    path('wallet/top-up/', WalletTopUpView.as_view(), name='wallet-top-up'),
    path('wallet/transactions/', BillingHistoryView.as_view(), name='wallet-transactions'),
]
