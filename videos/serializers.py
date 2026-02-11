from rest_framework import serializers
from .models import Video


class VideoSerializer(serializers.ModelSerializer):
    class Meta:
        model = Video
        fields = [
            "id",
            "original_video",
            "dubbed_video",
            "status",
            "error_message",
            "created_at",
        ]
        read_only_fields = ["status", "dubbed_video", "error_message"]
