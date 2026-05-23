from django.contrib.auth.models import AbstractBaseUser, PermissionsMixin, BaseUserManager
from django.db import models
from django.conf import settings
from django.utils import timezone

class PasswordResetCode(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    code = models.CharField(max_length=6)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.email} - {self.code}"

class UserManager(BaseUserManager):
    def create_user(self, username, email, password=None):
        if not email:
            raise ValueError("Email міндетті")

        email = self.normalize_email(email)
        user = self.model(username=username, email=email)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, username, email, password=None):
        user = self.create_user(username, email, password)
        user.is_staff = True
        user.is_superuser = True
        user.save(using=self._db)
        return user


class User(AbstractBaseUser, PermissionsMixin):
    username = models.CharField(max_length=150, unique=True)
    email = models.EmailField(unique=True)
    balance = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)

    objects = UserManager()

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = ["username"]

    def __str__(self):
        return self.email


class BillingTransaction(models.Model):
    TYPE_TOP_UP = "top_up"
    TYPE_DUBBING_CHARGE = "dubbing_charge"
    TYPE_REFUND = "refund"

    TYPE_CHOICES = [
        (TYPE_TOP_UP, "Top Up"),
        (TYPE_DUBBING_CHARGE, "Dubbing Charge"),
        (TYPE_REFUND, "Refund"),
    ]

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="billing_transactions")
    video = models.ForeignKey("videos.Video", on_delete=models.SET_NULL, null=True, blank=True, related_name="billing_transactions")
    txn_type = models.CharField(max_length=32, choices=TYPE_CHOICES)
    amount = models.DecimalField(max_digits=12, decimal_places=2)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.user_id} {self.txn_type} {self.amount}"


class NotificationPreference(models.Model):
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="notification_preferences",
    )
    notify_email = models.BooleanField(default=True)
    notify_completed = models.BooleanField(default=True)
    notify_marketing = models.BooleanField(default=False)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"NotificationPreference(user={self.user_id})"


class UserNotification(models.Model):
    TYPE_DUBBING_COMPLETED = "dubbing_completed"
    TYPE_DUBBING_FAILED = "dubbing_failed"
    TYPE_BILLING = "billing"
    TYPE_SYSTEM = "system"
    TYPE_MARKETING = "marketing"

    TYPE_CHOICES = [
        (TYPE_DUBBING_COMPLETED, "Dubbing Completed"),
        (TYPE_DUBBING_FAILED, "Dubbing Failed"),
        (TYPE_BILLING, "Billing"),
        (TYPE_SYSTEM, "System"),
        (TYPE_MARKETING, "Marketing"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="notifications",
    )
    video = models.ForeignKey(
        "videos.Video",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="notifications",
    )
    notification_type = models.CharField(max_length=32, choices=TYPE_CHOICES, default=TYPE_SYSTEM)
    title = models.CharField(max_length=200)
    message = models.TextField(blank=True)
    is_read = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    read_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"UserNotification(user={self.user_id}, type={self.notification_type}, read={self.is_read})"


class MarketingMessage(models.Model):
    subject = models.CharField(max_length=200)
    message = models.TextField()
    is_active = models.BooleanField(default=True)
    sent_at = models.DateTimeField(blank=True, null=True)
    recipients_count = models.PositiveIntegerField(default=0)
    last_error = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def mark_sent(self, recipients_count):
        self.sent_at = timezone.now()
        self.recipients_count = int(recipients_count)
        self.last_error = ""
        self.save(update_fields=["sent_at", "recipients_count", "last_error", "updated_at"])

    def mark_failed(self, error):
        self.last_error = str(error)
        self.save(update_fields=["last_error", "updated_at"])

    def __str__(self):
        return self.subject


# Create your models here.
