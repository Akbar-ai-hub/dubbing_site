import os
import shutil
import subprocess
import sys

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Run a Celery worker with autoscale values from settings/.env."

    def add_arguments(self, parser):
        parser.add_argument(
            "--worker-type",
            choices=["gpu", "cpu"],
            default="gpu",
            help="Worker profile to run (default: gpu).",
        )
        parser.add_argument(
            "--loglevel",
            default="info",
            help="Celery log level (default: info).",
        )
        parser.add_argument(
            "--hostname",
            default="",
            help="Optional custom Celery worker hostname.",
        )

    def handle(self, *args, **options):
        worker_type = options["worker_type"]
        loglevel = options["loglevel"]
        hostname = options["hostname"]

        if worker_type == "gpu":
            queue = settings.CELERY_DUBBING_GPU_QUEUE
            min_scale = int(settings.CELERY_GPU_AUTOSCALE_MIN)
            max_scale = int(settings.CELERY_GPU_AUTOSCALE_MAX)
            pool = (settings.CELERY_GPU_POOL or "").strip()
        else:
            queue = settings.CELERY_DUBBING_CPU_QUEUE
            min_scale = int(settings.CELERY_CPU_AUTOSCALE_MIN)
            max_scale = int(settings.CELERY_CPU_AUTOSCALE_MAX)
            pool = (settings.CELERY_CPU_POOL or "").strip()

        if max_scale < min_scale:
            raise CommandError(f"Invalid autoscale values: max={max_scale} < min={min_scale}")

        celery_bin = shutil.which("celery")
        if celery_bin:
            cmd = [celery_bin]
        else:
            cmd = [sys.executable, "-m", "celery"]

        cmd += [
            "-A",
            "dubbing_site",
            "worker",
            "-Q",
            queue,
            "-l",
            loglevel,
            "--autoscale",
            f"{max_scale},{min_scale}",
            "--prefetch-multiplier",
            str(getattr(settings, "CELERY_WORKER_PREFETCH_MULTIPLIER", 1)),
        ]
        max_tasks_per_child = int(getattr(settings, "CELERY_WORKER_MAX_TASKS_PER_CHILD", 0) or 0)
        if max_tasks_per_child > 0:
            cmd += ["--max-tasks-per-child", str(max_tasks_per_child)]
        if pool:
            cmd += ["-P", pool]
        if hostname:
            cmd += ["-n", hostname]

        self.stdout.write(
            self.style.SUCCESS(
                f"Starting Celery {worker_type} worker queue={queue} autoscale={max_scale},{min_scale} pool={pool or 'default'}"
            )
        )
        self.stdout.write("Command: " + " ".join(cmd))

        env = os.environ.copy()
        env.setdefault("DJANGO_SETTINGS_MODULE", "dubbing_site.settings")
        subprocess.run(cmd, env=env, check=True)
