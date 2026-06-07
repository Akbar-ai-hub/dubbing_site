from django.contrib.auth import get_user_model
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from .models import SupportMessage, SupportTicket


User = get_user_model()


class SupportTicketApiTests(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="support-user",
            email="support@example.com",
            password="pass12345",
        )
        self.client.force_authenticate(self.user)

    def test_create_support_ticket_with_first_message(self):
        response = self.client.post(
            reverse("support-tickets"),
            {
                "subject": "Video processing issue",
                "category": "processing",
                "message": "My video is stuck on processing.",
            },
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data["subject"], "Video processing issue")
        self.assertEqual(response.data["category"], "processing")
        self.assertEqual(response.data["status"], SupportTicket.STATUS_OPEN)
        self.assertEqual(len(response.data["messages"]), 1)
        self.assertEqual(response.data["messages"][0]["role"], SupportMessage.ROLE_USER)
        self.assertEqual(response.data["messages"][0]["author_name"], self.user.username)

    def test_append_support_message_to_owned_ticket(self):
        ticket = SupportTicket.objects.create(
            user=self.user,
            subject="Account question",
            category=SupportTicket.CATEGORY_ACCOUNT,
        )

        response = self.client.post(
            reverse("support-ticket-detail", kwargs={"ticket_id": ticket.id}),
            {"message": "Please check my account."},
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data["messages"]), 1)
        self.assertEqual(response.data["messages"][0]["message"], "Please check my account.")

    def test_support_ticket_list_is_user_scoped(self):
        other_user = User.objects.create_user(
            username="other-user",
            email="other@example.com",
            password="pass12345",
        )
        SupportTicket.objects.create(user=self.user, subject="Mine")
        SupportTicket.objects.create(user=other_user, subject="Other")

        response = self.client.get(reverse("support-tickets"))

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0]["subject"], "Mine")
