from django.contrib import admin
from django.contrib import messages

from .models import (
    BillingTransaction,
    MarketingMessage,
    NotificationPreference,
    PasswordResetCode,
    User,
    UserNotification,
)
from .services import send_marketing_message


@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    list_display = ("id", "email", "username", "balance", "debt", "is_active", "is_staff", "created_at")
    search_fields = ("email", "username")
    list_filter = ("is_active", "is_staff")
    ordering = ("-created_at",)


@admin.register(NotificationPreference)
class NotificationPreferenceAdmin(admin.ModelAdmin):
    list_display = ("user", "notify_email", "notify_completed", "notify_marketing", "updated_at")
    list_filter = ("notify_email", "notify_completed", "notify_marketing")
    search_fields = ("user__email", "user__username")


@admin.register(UserNotification)
class UserNotificationAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "notification_type", "title", "is_read", "created_at")
    list_filter = ("notification_type", "is_read")
    search_fields = ("user__email", "title", "message")
    readonly_fields = ("created_at", "read_at")


@admin.register(MarketingMessage)
class MarketingMessageAdmin(admin.ModelAdmin):
    list_display = ("id", "subject", "is_active", "sent_at", "recipients_count", "created_at")
    list_filter = ("is_active", "sent_at")
    search_fields = ("subject", "message")
    readonly_fields = ("sent_at", "recipients_count", "last_error", "created_at", "updated_at")

    def save_model(self, request, obj, form, change):
        super().save_model(request, obj, form, change)
        if not obj.is_active or obj.sent_at:
            return

        try:
            sent_count = send_marketing_message(obj)
        except Exception as exc:
            obj.mark_failed(exc)
            self.message_user(
                request,
                f"Marketing message was saved, but email sending failed: {exc}",
                level=messages.ERROR,
            )
            return

        self.message_user(
            request,
            f"Marketing message sent to {sent_count} opted-in users.",
        )


@admin.register(BillingTransaction)
class BillingTransactionAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "txn_type", "amount", "video", "created_at")
    list_filter = ("txn_type",)
    search_fields = ("user__email", "description")


@admin.register(PasswordResetCode)
class PasswordResetCodeAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "code", "created_at")
    search_fields = ("user__email", "code")
