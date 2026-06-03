from decimal import Decimal

from django.conf import settings
from rest_framework import status
from rest_framework.response import Response


def billing_enabled():
    return str(getattr(settings, "BILLING_ENABLED", "true")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def currency_code():
    return str(getattr(settings, "BILLING_CURRENCY", "KZT")).upper()


def user_debt_amount(user):
    return Decimal(str(getattr(user, "debt", "0") or "0"))


def user_has_debt(user):
    return user_debt_amount(user) > Decimal("0.00")


def debt_error_message(user):
    return (
        "You have unpaid debt. Please top up your balance before uploading videos, "
        f"starting dubbing, or requesting this video. Debt: {user_debt_amount(user)} {currency_code()}."
    )


def debt_required_response(user):
    return Response(
        {
            "error": debt_error_message(user),
            "debt": str(user_debt_amount(user)),
            "currency": currency_code(),
        },
        status=status.HTTP_402_PAYMENT_REQUIRED,
    )


def ensure_no_debt_response(user):
    if billing_enabled() and user_has_debt(user):
        return debt_required_response(user)
    return None
