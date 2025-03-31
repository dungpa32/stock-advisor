_L='minimal'
_K='warnings'
_J='api_requests'
_I='last_error_time'
_H='startup_time'
_G='standard'
_F='function_calls'
_E='peak_memory'
_D='errors'
_C=True
_B=None
_A='execution_times'
import time,threading,json,os
from datetime import datetime
from pathlib import Path
class Tracker:
	_instance=_B;_lock=threading.Lock()
	def __new__(A):
		with A._lock:
			if A._instance is _B:A._instance=super(Tracker,A).__new__(A);A._instance._initialize()
			return A._instance
	def _initialize(A):A.metrics={_H:datetime.now().isoformat(),_F:0,_J:0,_D:0,_K:0};A.performance_metrics={_A:[],_I:_B,_E:0};A.privacy_level=_G;A.home_dir=Path.home();A.project_dir=A.home_dir/'.vnstock';A.project_dir.mkdir(exist_ok=_C);A.data_dir=A.project_dir/'data';A.data_dir.mkdir(exist_ok=_C);A.metrics_path=A.data_dir/'usage_metrics.json';A.privacy_config_path=A.project_dir/'config'/'privacy.json';os.makedirs(os.path.dirname(A.privacy_config_path),exist_ok=_C);A._load_metrics();A._load_privacy_settings();A._start_background_collector()
	def _load_metrics(A):
		if A.metrics_path.exists():
			try:
				with open(A.metrics_path,'r')as C:D=json.load(C)
				for(B,E)in D.items():
					if B in A.metrics:A.metrics[B]=E
			except:pass
	def _save_metrics(A):
		try:
			with open(A.metrics_path,'w')as B:json.dump(A.metrics,B)
		except:pass
	def _load_privacy_settings(A):
		if A.privacy_config_path.exists():
			try:
				with open(A.privacy_config_path,'r')as B:C=json.load(B);A.privacy_level=C.get('level',_G)
			except:pass
	def setup_privacy(B,level=_B):
		A=level;C={_L:'Essential system data only',_G:'Performance metrics and errors','enhanced':'Detailed operation analytics'}
		if A is _B:A=_G
		if A not in C:raise ValueError(f"Invalid privacy level: {A}. Choose from {', '.join(C.keys())}")
		B.privacy_level=A
		with open(B.privacy_config_path,'w')as D:json.dump({'level':A},D)
		return A
	def get_privacy_level(A):return A.privacy_level
	def _start_background_collector(A):
		def B():
			while _C:
				try:
					import psutil as C;D=C.Process();E=D.memory_info();B=E.rss/1048576
					if B>A.performance_metrics[_E]:A.performance_metrics[_E]=B
					A._save_metrics()
				except:pass
				time.sleep(300)
		C=threading.Thread(target=B,daemon=_C);C.start()
	def record(A,event_type,data=_B):
		D='execution_time';C=data;B=event_type
		if A.privacy_level==_L and B!=_D:return _C
		if B in A.metrics:A.metrics[B]+=1
		else:A.metrics[B]=1
		if B==_D:A.performance_metrics[_I]=datetime.now().isoformat()
		if B==_F and C and D in C:
			A.performance_metrics[_A].append(C[D])
			if len(A.performance_metrics[_A])>100:A.performance_metrics[_A]=A.performance_metrics[_A][-100:]
		if A.metrics[_F]%100==0 or B==_D:A._save_metrics()
		return _C
	def get_metrics(A):
		B=0
		if A.performance_metrics[_A]:B=sum(A.performance_metrics[_A])/len(A.performance_metrics[_A])
		C=A.metrics.copy();C.update({'avg_execution_time':B,'peak_memory_mb':A.performance_metrics[_E],'uptime':(datetime.now()-datetime.fromisoformat(A.metrics[_H])).total_seconds(),'privacy_level':A.privacy_level});return C
	def reset(A):A.metrics={_H:datetime.now().isoformat(),_F:0,_J:0,_D:0,_K:0};A.performance_metrics={_A:[],_I:_B,_E:0};A._save_metrics();return _C
tracker=Tracker()
def record(event_type,data=_B):return tracker.record(event_type,data)