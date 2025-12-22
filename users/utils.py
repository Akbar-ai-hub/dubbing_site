import random
from django.core.mail import send_mail

def generate_reset_code():
    return str(random.randint(100000, 999999))


def send_reset_code(email, code):
    send_mail(
        subject="Password reset code",
        message=f"Your password reset code: {code}",
        from_email="ilhomulyakbar@gmail.com",
        recipient_list=[email],
        fail_silently=False,
    )
