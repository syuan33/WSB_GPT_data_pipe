import sys

from config_wsb import PATH_TO_SCRAPE_PY
    # caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, PATH_TO_SCRAPE_PY)

#from scrape_data_WSB import *

from datetime import datetime, timedelta
from textwrap import dedent

# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG

# Operators; we need this to operate!
from airflow.operators.bash import BashOperator
# These args will get passed on to each operator
# You can override them on a per-task basis during operator initialization
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['sam02151015@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
    # 'wait_for_downstream': False,
    # 'dag': dag,
    # 'sla': timedelta(hours=2),
    # 'execution_timeout': timedelta(seconds=300),
    # 'on_failure_callback': some_function,
    # 'on_success_callback': some_other_function,
    # 'on_retry_callback': another_function,
    # 'sla_miss_callback': yet_another_function,
    # 'trigger_rule': 'all_success'
}
with DAG(
    dag_id='wsb_translation',
    default_args=default_args,
    description='Download data and save into sqlite db',
    schedule_interval=timedelta(hours=6),
    start_date=datetime(2021, 1, 1),
    catchup=False,
    tags=['wsb','reddit','financial'],
) as dag:

    
    dag.doc_md = """
    This is a documentation placed anywhere
    """  # otherwise, type it like this
    templated_command = dedent(
        f"""
        bash {PATH_TO_SCRAPE_PY}/cron_job.sh
    """
    )

    t3 = BashOperator(
        task_id='templated',
        depends_on_past=False,
        bash_command=templated_command,
        #params={'my_param': 'Parameter I passed in'},
    )

    t3
