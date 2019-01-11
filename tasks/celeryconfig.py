from celery.schedules import crontab
from datetime import timedelta
from kombu import Queue,Exchange

task_default_queue='for_ml'
task_queues=(
    Queue('for_ml',Exchange('for_ml'),routing_key='for_ml'),
)
task_routes={
    'tasks.general.example.handle':{'queue':'for_ml', 'routing_key':'for_ml'},
    'tasks.general.example.word_count':{'queue':'for_ml','routing_key':'for_ml'},
}
task_default_exchange = 'for_ml'
task_default_routing_key='for_ml'

timezone = 'Asia/Shanghai'
imports = ('tasks.core',
           'tasks.beat',
           'tasks.general')

beat_schedule = {
    'example':{
        'task':'tasks.beat.beat_example',
        'schedule':crontab(minute=33,hour=3),
    }
}
