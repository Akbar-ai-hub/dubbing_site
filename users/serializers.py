from rest_framework import serializers
from .models import User
from django.contrib.auth.validators import UnicodeUsernameValidator
from django.contrib.auth.password_validation import validate_password
from django.utils.html import strip_tags

class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, validators=[validate_password])
    username_validator = UnicodeUsernameValidator()

    class Meta:
        model = User
        fields = ['username', 'email', 'password']

    def validate_username(self, value):
        normalized = (value or "").strip()

        # Reject HTML/script payloads early to mitigate stored XSS vectors.
        if strip_tags(normalized) != normalized:
            raise serializers.ValidationError("Username must not contain HTML or script content.")

        self.username_validator(normalized)
        return normalized

    def validate_email(self, value):
        return (value or "").strip().lower()

    def create(self, validated_data):
        user = User.objects.create_user(
            username=validated_data["username"],
            email=validated_data["email"],
            password=validated_data["password"]
        )
        return user
