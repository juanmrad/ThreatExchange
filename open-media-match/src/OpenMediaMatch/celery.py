# OpenMediaMatch/celery.py

from celery import Celery
from flask import Flask
from OpenMediaMatch.app import app

def make_celery(app: Flask)-> Celery:
    celery_app = Celery(
        app.name,
        include=['OpenMediaMatch.celery_tasks']  # Import all tasks from the celery_tasks package
    )
    celery_app.config_from_object(app.config["CELERY"])
    celery_app.set_default()
    celery_app.conf.beat_schedule = {
      'add-every-60-seconds': {
          'task': 'tasks.example_celery_task',
          'schedule': 60.0,
          'args': (16, 16)
      },
    }
    celery_app.conf.timezone = 'UTC'
    app.extensions["celery"] = celery_app
    return celery_app

celery = make_celery(app)
