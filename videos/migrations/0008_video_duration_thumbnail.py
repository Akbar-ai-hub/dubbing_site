from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("videos", "0007_alter_video_status"),
    ]

    operations = [
        migrations.AddField(
            model_name="video",
            name="duration",
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="video",
            name="thumbnail",
            field=models.FileField(blank=True, null=True, upload_to="video_thumbnails/"),
        ),
    ]
