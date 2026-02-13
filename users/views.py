import os
from datetime import timedelta

from django.contrib.auth import authenticate, get_user_model
from django.utils import timezone
from google.auth.transport import requests
from google.oauth2 import id_token
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken, TokenError

from .models import PasswordResetCode
from .serializers import RegisterSerializer
from .throttles import (
    PasswordResetCompleteThrottle,
    PasswordResetRequestThrottle,
    PasswordResetVerifyThrottle,
)
from .utils import generate_reset_code, send_reset_code

User = get_user_model()
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")


def get_tokens_for_user(user):
    refresh = RefreshToken.for_user(user)
    return {
        "refresh": str(refresh),
        "access": str(refresh.access_token),
    }


class RegisterView(APIView):
    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            tokens = get_tokens_for_user(user)

            return Response(
                {
                    "message": "Registration completed successfully",
                    "user": {
                        "username": user.username,
                        "email": user.email,
                    },
                    "tokens": tokens,
                },
                status=status.HTTP_201_CREATED,
            )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class LoginView(APIView):
    def post(self, request):
        email = request.data.get("email")
        password = request.data.get("password")

        if not email or not password:
            return Response(
                {"error": "Email and password are required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        user = authenticate(request, email=email, password=password)
        if user is None:
            return Response(
                {"error": "Invalid email or password"},
                status=status.HTTP_401_UNAUTHORIZED,
            )

        tokens = get_tokens_for_user(user)
        return Response(
            {
                "message": "Login successful",
                "user": {
                    "username": user.username,
                    "email": user.email,
                },
                "tokens": tokens,
            },
            status=status.HTTP_200_OK,
        )


class GoogleLoginView(APIView):
    def post(self, request):
        token = request.data.get("token")
        if not GOOGLE_CLIENT_ID:
            return Response(
                {"error": "GOOGLE_CLIENT_ID is not configured"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        try:
            google_user = id_token.verify_oauth2_token(
                token, requests.Request(), GOOGLE_CLIENT_ID
            )
        except Exception:
            return Response({"error": "Invalid Google token"}, status=status.HTTP_400_BAD_REQUEST)

        email = google_user.get("email")
        username = google_user.get("name") or email.split("@")[0]

        user, _ = User.objects.get_or_create(
            email=email,
            defaults={"username": username},
        )

        tokens = get_tokens_for_user(user)
        return Response(
            {
                "message": "Google login successful",
                "user": {
                    "username": user.username,
                    "email": user.email,
                },
                "tokens": tokens,
            },
            status=status.HTTP_200_OK,
        )


class PasswordResetRequestView(APIView):
    throttle_classes = [PasswordResetRequestThrottle]

    def post(self, request):
        email = request.data.get("email")
        if not email:
            return Response(
                {"error": "Email is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Always return the same response to prevent email enumeration.
        try:
            user = User.objects.get(email=email)
            code = generate_reset_code()
            PasswordResetCode.objects.create(user=user, code=code)
            send_reset_code(email, code)
        except User.DoesNotExist:
            pass

        return Response(
            {"message": "If this email exists, a reset code was sent"},
            status=status.HTTP_200_OK,
        )


class PasswordResetVerifyView(APIView):
    throttle_classes = [PasswordResetVerifyThrottle]

    def post(self, request):
        email = request.data.get("email")
        code = request.data.get("code")

        if not email or not code:
            return Response(
                {"error": "Email and code are required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            return Response({"error": "Email not found"}, status=status.HTTP_404_NOT_FOUND)

        try:
            reset_code = PasswordResetCode.objects.filter(user=user, code=code).latest("created_at")
        except PasswordResetCode.DoesNotExist:
            return Response({"error": "Invalid code"}, status=status.HTTP_400_BAD_REQUEST)

        if reset_code.created_at < timezone.now() - timedelta(minutes=10):
            return Response({"error": "Code has expired"}, status=status.HTTP_400_BAD_REQUEST)

        return Response({"message": "Code is valid"}, status=status.HTTP_200_OK)


class PasswordResetCompleteView(APIView):
    throttle_classes = [PasswordResetCompleteThrottle]

    def post(self, request):
        email = request.data.get("email")
        code = request.data.get("code")
        new_password = request.data.get("new_password")

        if not email or not code or not new_password:
            return Response(
                {"error": "Email, code and new_password are required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            return Response({"error": "Email not found"}, status=status.HTTP_404_NOT_FOUND)

        try:
            reset_code = PasswordResetCode.objects.filter(user=user, code=code).latest("created_at")
        except PasswordResetCode.DoesNotExist:
            return Response({"error": "Invalid code"}, status=status.HTTP_400_BAD_REQUEST)

        if reset_code.created_at < timezone.now() - timedelta(minutes=10):
            return Response({"error": "Code has expired"}, status=status.HTTP_400_BAD_REQUEST)

        user.set_password(new_password)
        user.save()
        return Response({"message": "Password updated successfully"}, status=status.HTTP_200_OK)


class LogoutView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        refresh_token = request.data.get("refresh")
        if not refresh_token:
            return Response(
                {"error": "Refresh token is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            token = RefreshToken(refresh_token)
            token.blacklist()
        except TokenError:
            return Response(
                {"error": "Invalid or expired refresh token"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        return Response({"message": "Logged out successfully"}, status=status.HTTP_200_OK)
