from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("users", "0005_marketingmessage_notification_type"),
    ]

    operations = [
        migrations.AddField(
            model_name="user",
            name="debt",
            field=models.DecimalField(decimal_places=2, default=0, max_digits=12),
        ),
    ]
