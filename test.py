#coding:utf-8

from utils.helper import add_mlpm_task_func
from tasks.general.example import handle


#add_mlpm_task_func(handle)
resp=handle.delay('SVM','/srv/sites/mlpm-jobs/media/_fs/test.xlsx',{})
print(resp.id)
