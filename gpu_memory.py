import subprocess
 
def get_gpu_properties(
  # default_properies = (
  #   "timestamp",
  #   "gpu_name",
  #   #"gpu_uuid",
  #   "index",
  #   "memory.total",
  #   "memory.used",
  #   "memory.free",
  #   "utilization.gpu",
  #   "utilization.memory",
  #   )
  cmd_path="C:\Windows\System32\\nvidia-smi.exe",
  noheader=True,
  nounits=True
  ):
  target_properties = (
    "timestamp",
    "gpu_name",
    #"gpu_uuid",
    "index",
    "memory.total",
    "memory.used",
    "memory.free",
    "utilization.gpu",
    "utilization.memory",
    )
    
  """
  CUDA GPUのプロパティ情報取得
 
  Parameters
  ----------
  cmd_path : str
    コマンドラインから"nvidia-smi"を実行する際のパス
  target_properties : obj
    取得するプロパティ情報
    プロパティ情報の詳細は"nvidia-smi --help-query-gpu"で取得可能
  noheader : bool
    skip the first line with column headers
  nounits : bool
    don't print units for numerical values
 
  Returns
  -------
  gpu_properties : list
    gpuごとのproperty情報
  """
    
  # formatオプション定義
  format_option = "--format=csv"
  if noheader:
      format_option += ",noheader"
  if nounits:
      format_option += ",nounits"
 
  # コマンド生成
  cmd = '%s --query-gpu=%s %s' % (cmd_path, ','.join(target_properties), format_option)
 
  # サブプロセスでコマンド実行
  cmd_res = subprocess.check_output(cmd, shell=True)
    
  # コマンド実行結果をオブジェクトに変換
  gpu_lines = cmd_res.decode().split('\n')
  # リストの最後の要素に空行が入るため除去
  gpu_lines = [ line.strip() for line in gpu_lines if line.strip() != '' ]
 
  # ", "ごとにプロパティ情報が入っているのでdictにして格納
  gpu_properties = [ { k: v for k, v in zip(target_properties, line.split(', ')) } for line in gpu_lines ]
 
  return gpu_properties

import threading
import time
 
# グローバル変数
# g_IsMonitor = False
# g_GpuPropertiesList = []

def monitoring_gpu_task(interval):
  global g_IsMonitor
  global g_GpuPropertiesList
 
  while g_IsMonitor:
    time.sleep(interval)
    res = get_gpu_properties()
    g_GpuPropertiesList.append( res )
 
def start_gpu_monitor(interval):
  global g_IsMonitor
  global g_GpuPropertiesList
 
  g_IsMonitor = True
  g_GpuPropertiesList.clear()
 
  monitor_thread = threading.Thread(target=monitoring_gpu_task, args=(interval, ))
  monitor_thread.start()
 
def end_gpu_monitor():
  global g_IsMonitor
  global g_GpuPropertiesList
  g_IsMonitor = False
  return g_GpuPropertiesList
 

# 利用例
# 1sec間隔でGPUのProperty情報を取得
# for number in range(3):
#   start_gpu_monitor(1)
#   time.sleep(3)
#   properties_list = end_gpu_monitor()
#   print(properties_list)