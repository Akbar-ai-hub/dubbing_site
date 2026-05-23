from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("users", "0004_notificationpreference_usernotification"),
    ]

    operations = [
        migrations.AlterField(
            model_name="usernotification",
            name="notification_type",
            field=models.CharField(
                choices=[
                    ("dubbing_completed", "Dubbing Completed"),
                    ("dubbing_failed", "Dubbing Failed"),
                    ("billing", "Billing"),
                    ("system", "System"),
                    ("marketing", "Marketing"),
                ],
                default="system",
                max_length=32,
            ),
        ),
        migrations.CreateModel(
            name="MarketingMessage",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("subject", models.CharField(max_length=200)),
                ("message", models.TextField()),
                ("is_active", models.BooleanField(default=True)),
                ("sent_at", models.DateTimeField(blank=True, null=True)),
                ("recipients_count", models.PositiveIntegerField(default=0)),
                ("last_error", models.TextField(blank=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "ordering": ["-created_at"],
            },
        ),
    ]
