from django.contrib import admin
from django.contrib import messages

from .models import (
    BillingTransaction,
    MarketingMessage,
    NotificationPreference,
    PasswordResetCode,
    SupportMessage,
    SupportTicket,
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


class SupportMessageInline(admin.TabularInline):
    model = SupportMessage
    extra = 1
    fields = ("role", "author", "message", "created_at")
    readonly_fields = ("created_at",)


@admin.register(SupportTicket)
class SupportTicketAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "subject", "category", "status", "updated_at")
    list_filter = ("status", "category")
    search_fields = ("user__email", "user__username", "subject", "messages__message")
    readonly_fields = ("created_at", "updated_at")
    inlines = [SupportMessageInline]

    def save_formset(self, request, form, formset, change):
        instances = formset.save(commit=False)
        for deleted in formset.deleted_objects:
            deleted.delete()
        for instance in instances:
            if isinstance(instance, SupportMessage) and instance.pk is None and request.user.is_staff:
                instance.role = SupportMessage.ROLE_ADMIN
            if isinstance(instance, SupportMessage) and instance.author_id is None:
                if instance.role == SupportMessage.ROLE_ADMIN:
                    instance.author = request.user
                else:
                    instance.author = form.instance.user
            instance.save()
            if isinstance(instance, SupportMessage):
                instance.ticket.save(update_fields=["updated_at"])
        formset.save_m2m()


@admin.register(SupportMessage)
class SupportMessageAdmin(admin.ModelAdmin):
    list_display = ("id", "ticket", "role", "author", "message_preview", "created_at")
    list_filter = ("role",)
    search_fields = ("ticket__subject", "ticket__user__email", "message")
    readonly_fields = ("created_at",)

    def message_preview(self, obj):
        message = (obj.message or "").strip()
        if len(message) <= 80:
            return message
        return f"{message[:80]}..."

    message_preview.short_description = "Message"
