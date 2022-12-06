import gpu_memory
import time

g_IsMonitor = False
g_GpuPropertiesList = []
def monitoring_gpu_task(interval):
  global g_IsMonitor
  global g_GpuPropertiesList
 
  while g_IsMonitor:
    time.sleep(interval)
    res = gpu_memory.get_gpu_properties()
    g_GpuPropertiesList.append( res )
 
def start_gpu_monitor(interval):
  global g_IsMonitor
  global g_GpuPropertiesList
 
  g_IsMonitor = True
  g_GpuPropertiesList.clear()
 
  monitor_thread = gpu_memory.threading.Thread(target=monitoring_gpu_task, args=(interval, ))
  monitor_thread.start()
 
def end_gpu_monitor():
  global g_IsMonitor
  global g_GpuPropertiesList
  g_IsMonitor = False
  return g_GpuPropertiesList

start_gpu_monitor(1)
time.sleep(15)
properties_list = end_gpu_monitor()
print(properties_list)
for dict in properties_list:
  dict[0]['timestamp'] += ('"')
import csv
with open('sample_gpu.csv', 'w', newline="") as f:
  writer = csv.DictWriter(f, ['timestamp','memory.used','memory.free', 'index', 'utilization.gpu', 'utilization.memory', 'gpu_name', 'memory.total'])
  writer.writeheader()
  for dict in properties_list:
    writer.writerow(dict[0])