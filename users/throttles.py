from rest_framework.throttling import AnonRateThrottle


class PasswordResetRequestThrottle(AnonRateThrottle):
    rate = "5/hour"


class PasswordResetVerifyThrottle(AnonRateThrottle):
    rate = "10/hour"


class PasswordResetCompleteThrottle(AnonRateThrottle):
    rate = "10/hour"
