_K='success'
_J='buffer_size'
_I='request'
_H='rate_limit'
_G='execution_time'
_F='timestamp'
_E=False
_D=True
_C='error'
_B=None
_A='function'
import sys,time,threading
from datetime import datetime
class Collector:
	_instance=_B;_lock=threading.Lock()
	def __new__(A):
		with A._lock:
			if A._instance is _B:A._instance=super(Collector,A).__new__(A);A._instance._initialize()
			return A._instance
	def _initialize(A):A.metrics={_A:[],_H:[],_I:[],_C:[]};A.thresholds={_J:50,'error_threshold':.1,'performance_threshold':5.};A.function_count=0;A.colab_auth_triggered=_E
	def record(A,metric_type,data,priority=_B):
		D='system';C=metric_type;B=data
		if not isinstance(B,dict):B={'value':str(B)}
		if _F not in B:B[_F]=datetime.now().isoformat()
		if C!='system_info'and isinstance(B,dict):
			if D in B:del B[D]
			from vnai.scope.profile import inspector as E;B['machine_id']=E.fingerprint()
		if C in A.metrics:A.metrics[C].append(B)
		else:A.metrics[_A].append(B)
		if C==_A:
			A.function_count+=1
			if A.function_count>10 and not A.colab_auth_triggered and'google.colab'in sys.modules:A.colab_auth_triggered=_D;threading.Thread(target=A._trigger_colab_auth,daemon=_D).start()
		if sum(len(A)for A in A.metrics.values())>=A.thresholds[_J]:A._send_metrics()
		if priority=='high'or C==_C:A._send_metrics()
	def _trigger_colab_auth(B):
		try:from vnai.scope.profile import inspector as A;A.get_or_create_user_id()
		except:pass
	def _send_metrics(F):
		E='vnai';D='source';C='unknown';from vnai.flow.relay import track_function_call as H,track_rate_limit as I,track_api_request as J
		for(B,G)in F.metrics.items():
			if not G:continue
			for A in G:
				try:
					if B==_A:H(function_name=A.get(_A,C),source=A.get(D,E),execution_time=A.get(_G,0),success=A.get(_K,_D),error=A.get(_C),args=A.get('args'))
					elif B==_H:I(source=A.get(D,E),limit_type=A.get('limit_type',C),limit_value=A.get('limit_value',0),current_usage=A.get('current_usage',0),is_exceeded=A.get('is_exceeded',_E))
					elif B==_I:J(endpoint=A.get('endpoint',C),source=A.get(D,E),method=A.get('method','GET'),status_code=A.get('status_code',200),execution_time=A.get(_G,0),request_size=A.get('request_size',0),response_size=A.get('response_size',0))
				except Exception as K:continue
			F.metrics[B]=[]
	def get_metrics_summary(A):return{A:len(B)for(A,B)in A.metrics.items()}
collector=Collector()
def capture(module_type=_A):
	def A(func):
		def A(*A,**D):
			E=time.time();B=_E;C=_B
			try:F=func(*A,**D);B=_D;return F
			except Exception as G:C=str(G);raise
			finally:H=time.time()-E;collector.record(module_type,{_A:func.__name__,_G:H,_K:B,_C:C,_F:datetime.now().isoformat(),'args':str(A)[:100]if A else _B})
		return A
	return A