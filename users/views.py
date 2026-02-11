import os

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.contrib.auth import authenticate
from rest_framework_simplejwt.tokens import RefreshToken

from google.oauth2 import id_token
from google.auth.transport import requests

from .models import User
from .serializers import RegisterSerializer
from django.contrib.auth import get_user_model
from django.utils import timezone
from datetime import timedelta

from .models import PasswordResetCode
from .utils import generate_reset_code, send_reset_code

User = get_user_model()

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")


# JWT функциясы: юзерге токен генерациялайды
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

            return Response({
                "message": "Тіркелу сәтті аяқталды!",
                "user": {
                    "username": user.username,
                    "email": user.email
                },
                "tokens": tokens
            }, status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class LoginView(APIView):
    def post(self, request):
        email = request.data.get("email")
        password = request.data.get("password")

        if not email or not password:
            return Response({"error": "Email және пароль керек"}, status=status.HTTP_400_BAD_REQUEST)

        user = authenticate(request, email=email, password=password)

        if user is None:
            return Response({"error": "Email немесе пароль қате"}, status=status.HTTP_401_UNAUTHORIZED)

        tokens = get_tokens_for_user(user)

        return Response({
            "message": "Сәтті кірдіңіз!",
            "user": {
                "username": user.username,
                "email": user.email,
            },
            "tokens": tokens
        }, status=status.HTTP_200_OK)

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
        except:
            return Response({"error": "Google token дұрыс емес"}, status=status.HTTP_400_BAD_REQUEST)

        email = google_user.get("email")
        username = google_user.get("name")

        # Егер юзер бұрын жасалмаса, автоматты түрде жасаймыз
        user, created = User.objects.get_or_create(
            email=email,
            defaults={"username": username}
        )

        tokens = get_tokens_for_user(user)

        return Response({
            "message": "Google арқылы сәтті кірдіңіз!",
            "user": {
                "username": user.username,
                "email": user.email
            },
            "tokens": tokens
        })



# ------------------------------
# 1) Email-ге код жіберу
# ------------------------------

class PasswordResetRequestView(APIView):
    def post(self, request):
        email = request.data.get("email")

        if not email:
            return Response({"error": "Email енгізіңіз"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            return Response({"error": "Мұндай email табылмады"}, status=status.HTTP_404_NOT_FOUND)

        # код генерациялау
        code = generate_reset_code()

        # базаға сақтау
        PasswordResetCode.objects.create(user=user, code=code)

        # email-ге жіберу
        send_reset_code(email, code)

        return Response({"message": "Код email-ге жіберілді"}, status=status.HTTP_200_OK)


# ------------------------------
# 2) Кодты тексеру
# ------------------------------

class PasswordResetVerifyView(APIView):
    def post(self, request):
        email = request.data.get("email")
        code = request.data.get("code")

        if not email or not code:
            return Response({"error": "Email және код керек"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            return Response({"error": "Email табылмады"}, status=status.HTTP_404_NOT_FOUND)

        try:
            reset_code = PasswordResetCode.objects.filter(user=user, code=code).latest("created_at")
        except PasswordResetCode.DoesNotExist:
            return Response({"error": "Код дұрыс емес"}, status=status.HTTP_400_BAD_REQUEST)

        # Код 10 минутқа жарамды
        if reset_code.created_at < timezone.now() - timedelta(minutes=10):
            return Response({"error": "Кодтың уақыты біткен"}, status=status.HTTP_400_BAD_REQUEST)

        return Response({"message": "Код дұрыс"}, status=status.HTTP_200_OK)


# ------------------------------
# 3) Жаңа пароль орнату
# ------------------------------

class PasswordResetCompleteView(APIView):
    def post(self, request):
        email = request.data.get("email")
        code = request.data.get("code")
        new_password = request.data.get("new_password")

        if not email or not code or not new_password:
            return Response({"error": "Барлық өрістерді толтырыңыз"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            return Response({"error": "Email табылмады"}, status=status.HTTP_404_NOT_FOUND)

        try:
            reset_code = PasswordResetCode.objects.filter(user=user, code=code).latest("created_at")
        except PasswordResetCode.DoesNotExist:
            return Response({"error": "Код дұрыс емес"}, status=status.HTTP_400_BAD_REQUEST)

        # Кодтың жарамдылық уақыты
        if reset_code.created_at < timezone.now() - timedelta(minutes=10):
            return Response({"error": "Кодтың уақыты біткен"}, status=status.HTTP_400_BAD_REQUEST)

        # Пароль жаңарту
        user.set_password(new_password)
        user.save()

        return Response({"message": "Пароль сәтті жаңартылды!"}, status=status.HTTP_200_OK)
