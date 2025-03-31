_B='status'
_A='healthy'
import threading,time
from datetime import datetime
class Monitor:
	_instance=None;_lock=threading.Lock()
	def __new__(A):
		with A._lock:
			if A._instance is None:A._instance=super(Monitor,A).__new__(A);A._instance._initialize()
			return A._instance
	def _initialize(A):A.health_status=_A;A.last_check=time.time();A.check_interval=300;A.error_count=0;A.warning_count=0;A.status_history=[];A._start_background_check()
	def _start_background_check(A):
		def B():
			while True:
				try:A.check_health()
				except:pass
				time.sleep(A.check_interval)
		C=threading.Thread(target=B,daemon=True);C.start()
	def check_health(A):
		from vnai.beam.metrics import collector as F;from vnai.beam.quota import guardian as G;A.last_check=time.time();B=F.get_metrics_summary();C=B.get('error',0)>0;D=G.usage();E=D>80
		if C and E:A.health_status='critical';A.error_count+=1
		elif C or E:A.health_status='warning';A.warning_count+=1
		else:A.health_status=_A
		A.status_history.append({'timestamp':datetime.now().isoformat(),_B:A.health_status,'metrics':B,'resource_usage':D})
		if len(A.status_history)>10:A.status_history=A.status_history[-10:]
		return A.health_status
	def report(A):
		if time.time()-A.last_check>A.check_interval:A.check_health()
		return{_B:A.health_status,'last_check':datetime.fromtimestamp(A.last_check).isoformat(),'error_count':A.error_count,'warning_count':A.warning_count,'history':A.status_history[-3:]}
	def reset(A):A.health_status=_A;A.error_count=0;A.warning_count=0;A.status_history=[];A.last_check=time.time()
monitor=Monitor()