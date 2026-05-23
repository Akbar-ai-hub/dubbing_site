import logging

from django.conf import settings
from django.core.mail import get_connection, send_mass_mail

from .models import MarketingMessage, NotificationPreference, UserNotification


logger = logging.getLogger(__name__)


def send_marketing_message(marketing_message):
    if not isinstance(marketing_message, MarketingMessage):
        raise TypeError("marketing_message must be a MarketingMessage instance")
    if not marketing_message.is_active:
        return 0
    if marketing_message.sent_at:
        return marketing_message.recipients_count

    from_email = (
        getattr(settings, "DEFAULT_FROM_EMAIL", "")
        or getattr(settings, "EMAIL_HOST_USER", "")
        or None
    )
    if not from_email or not getattr(settings, "EMAIL_HOST_PASSWORD", ""):
        raise RuntimeError("EMAIL_HOST_USER/EMAIL_HOST_PASSWORD is not configured")

    preferences = (
        NotificationPreference.objects
        .select_related("user")
        .filter(notify_marketing=True, user__is_active=True)
        .exclude(user__email="")
    )
    recipients = list(
        dict.fromkeys(
            preference.user.email
            for preference in preferences
            if preference.user.email
        )
    )
    if not recipients:
        marketing_message.mark_sent(0)
        logger.info("Marketing message %s has no opted-in recipients", marketing_message.id)
        return 0

    email_payload = [
        (
            marketing_message.subject,
            marketing_message.message,
            from_email,
            [recipient],
        )
        for recipient in recipients
    ]
    sent_count = send_mass_mail(
        email_payload,
        fail_silently=False,
        connection=get_connection(fail_silently=False),
    )

    UserNotification.objects.bulk_create(
        [
            UserNotification(
                user_id=preference.user_id,
                title=marketing_message.subject,
                message=marketing_message.message,
                notification_type=UserNotification.TYPE_MARKETING,
            )
            for preference in preferences
            if preference.user.email in recipients
        ],
        batch_size=500,
    )
    marketing_message.mark_sent(sent_count)
    logger.info(
        "Marketing message %s sent to %s recipients",
        marketing_message.id,
        sent_count,
    )
    return sent_count
