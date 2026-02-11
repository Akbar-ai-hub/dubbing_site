ALLOWED_EXTENSIONS = ["mp4", "wav"]
MAX_FILE_SIZE_MB = 100


def validate_video_file(file):
    extension = file.name.split(".")[-1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError("Файл форматы mp4 немесе wav болуы керек")

    if file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise ValueError("Файл көлемі 100MB-тан аспау керек")
