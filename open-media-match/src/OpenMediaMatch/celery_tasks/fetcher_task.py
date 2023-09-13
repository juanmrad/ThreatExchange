# Copyright (c) Meta Platforms, Inc. and affiliates.

from OpenMediaMatch.extensions import celery

@celery.task
def example_celery_task(arg1, arg2):
    # Your background task code here
    # Example: print a message
    print(f'Running background task with args: {arg1}, {arg2}')