import unicodedata
from decimal import Decimal

from django.conf import settings
from rest_framework import serializers
from .models import (
    BillingTransaction,
    NotificationPreference,
    SupportMessage,
    SupportTicket,
    User,
    UserNotification,
)
from django.contrib.auth.password_validation import validate_password
from django.utils.html import strip_tags

class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, validators=[validate_password])

    class Meta:
        model = User
        fields = ['username', 'email', 'password']

    def validate_username(self, value):
        normalized = (value or "").strip()

        # Reject HTML/script payloads early to mitigate stored XSS vectors.
        if strip_tags(normalized) != normalized:
            raise serializers.ValidationError("Username must not contain HTML or script content.")

        if not normalized:
            raise serializers.ValidationError("Username is required.")

        if len(normalized) > 150:
            raise serializers.ValidationError("Username must be 150 characters or fewer.")

        allowed_punctuation = {" ", "-", "_", ".", "'"}
        for char in normalized:
            category = unicodedata.category(char)
            if category.startswith(("L", "N")):
                continue
            if char in allowed_punctuation:
                continue
            raise serializers.ValidationError(
                "Username may contain letters, numbers, spaces, and - _ . ' characters only."
            )

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

    def update(self, instance, validated_data):
        password = validated_data.pop("password", None)
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        if password:
            instance.set_password(password)
        instance.save()
        return instance


class WalletTopUpSerializer(serializers.Serializer):
    amount = serializers.DecimalField(max_digits=12, decimal_places=2, min_value=Decimal("0.01"))


class BillingTransactionSerializer(serializers.ModelSerializer):
    currency = serializers.SerializerMethodField()

    class Meta:
        model = BillingTransaction
        fields = ["id", "txn_type", "amount", "currency", "description", "video", "created_at"]

    def get_currency(self, obj):
        return str(getattr(settings, "BILLING_CURRENCY", "KZT")).upper()


class NotificationPreferenceSerializer(serializers.ModelSerializer):
    class Meta:
        model = NotificationPreference
        fields = ["notify_email", "notify_completed", "notify_marketing"]


class UserNotificationSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserNotification
        fields = [
            "id",
            "title",
            "message",
            "notification_type",
            "is_read",
            "created_at",
        ]


class SupportMessageSerializer(serializers.ModelSerializer):
    author_name = serializers.SerializerMethodField()

    class Meta:
        model = SupportMessage
        fields = ["id", "role", "message", "author_name", "created_at"]

    def get_author_name(self, obj):
        if obj.role == SupportMessage.ROLE_ADMIN:
            return "Admin"
        if obj.author:
            return obj.author.username or obj.author.email
        return "User"


class SupportTicketSerializer(serializers.ModelSerializer):
    messages = SupportMessageSerializer(many=True, read_only=True)

    class Meta:
        model = SupportTicket
        fields = [
            "id",
            "subject",
            "category",
            "status",
            "messages",
            "created_at",
            "updated_at",
        ]


class SupportTicketCreateSerializer(serializers.Serializer):
    subject = serializers.CharField(max_length=200, allow_blank=True, required=False)
    category = serializers.ChoiceField(
        choices=SupportTicket.CATEGORY_CHOICES,
        default=SupportTicket.CATEGORY_GENERAL,
    )
    message = serializers.CharField(min_length=2, max_length=5000)

    def validate_subject(self, value):
        clean_value = (value or "").strip()
        if strip_tags(clean_value) != clean_value:
            raise serializers.ValidationError("Subject must not contain HTML or script content.")
        return clean_value

    def validate_message(self, value):
        clean_value = (value or "").strip()
        if strip_tags(clean_value) != clean_value:
            raise serializers.ValidationError("Message must not contain HTML or script content.")
        return clean_value


class SupportMessageCreateSerializer(serializers.Serializer):
    message = serializers.CharField(min_length=2, max_length=5000)

    def validate_message(self, value):
        clean_value = (value or "").strip()
        if strip_tags(clean_value) != clean_value:
            raise serializers.ValidationError("Message must not contain HTML or script content.")
        return clean_value
