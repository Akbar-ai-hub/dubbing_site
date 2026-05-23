from django.db import migrations, models
from django.utils import timezone


class Migration(migrations.Migration):
    dependencies = [
        ("videos", "0008_video_duration_thumbnail"),
    ]

    operations = [
        migrations.AddField(
            model_name="video",
            name="updated_at",
            field=models.DateTimeField(auto_now=True, default=timezone.now),
            preserve_default=False,
        ),
    ]
