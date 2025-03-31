_T='execution_time'
_S='manual'
_R='success'
_Q='is_exceeded'
_P='source'
_O='function'
_N='last_sync_time'
_M='sync_interval'
_L='buffer_size'
_K='webhook_url'
_J='value'
_I='sync_count'
_H='machine_id'
_G=False
_F=None
_E='timestamp'
_D='api_requests'
_C='rate_limits'
_B='function_calls'
_A=True
import time,threading,json,random,requests
from datetime import datetime
from pathlib import Path
from typing import Dict,List,Any,Optional
class Conduit:
	_instance=_F;_lock=threading.Lock()
	def __new__(A,webhook_url=_F,buffer_size=50,sync_interval=300):
		with A._lock:
			if A._instance is _F:A._instance=super(Conduit,A).__new__(A);A._instance._initialize(webhook_url,buffer_size,sync_interval)
			return A._instance
	def _initialize(A,webhook_url,buffer_size,sync_interval):
		A.webhook_url=webhook_url;A.buffer_size=buffer_size;A.sync_interval=sync_interval;A.buffer={_B:[],_D:[],_C:[]};A.lock=threading.Lock();A.last_sync_time=time.time();A.sync_count=0;A.failed_queue=[];A.home_dir=Path.home();A.project_dir=A.home_dir/'.vnstock';A.project_dir.mkdir(exist_ok=_A);A.data_dir=A.project_dir/'data';A.data_dir.mkdir(exist_ok=_A);A.config_path=A.data_dir/'relay_config.json'
		try:from vnai.scope.profile import inspector as B;A.machine_id=B.fingerprint()
		except:A.machine_id=A._generate_fallback_id()
		A._load_config();A._start_periodic_sync()
	def _generate_fallback_id(D)->str:
		try:import platform as A,hashlib as B,uuid;C=A.node()+A.platform()+A.processor();return B.md5(C.encode()).hexdigest()
		except:import uuid;return str(uuid.uuid4())
	def _load_config(B):
		if B.config_path.exists():
			try:
				with open(B.config_path,'r')as C:A=json.load(C)
				if not B.webhook_url and _K in A:B.webhook_url=A[_K]
				if _L in A:B.buffer_size=A[_L]
				if _M in A:B.sync_interval=A[_M]
				if _N in A:B.last_sync_time=A[_N]
				if _I in A:B.sync_count=A[_I]
			except:pass
	def _save_config(A):
		B={_K:A.webhook_url,_L:A.buffer_size,_M:A.sync_interval,_N:A.last_sync_time,_I:A.sync_count}
		try:
			with open(A.config_path,'w')as C:json.dump(B,C)
		except:pass
	def _start_periodic_sync(A):
		def B():
			while _A:time.sleep(A.sync_interval);A.dispatch('periodic')
		C=threading.Thread(target=B,daemon=_A);C.start()
	def add_function_call(B,record):
		A=record
		if not isinstance(A,dict):A={_J:str(A)}
		with B.lock:B.buffer[_B].append(A);B._check_triggers(_B)
	def add_api_request(B,record):
		A=record
		if not isinstance(A,dict):A={_J:str(A)}
		with B.lock:B.buffer[_D].append(A);B._check_triggers(_D)
	def add_rate_limit(B,record):
		A=record
		if not isinstance(A,dict):A={_J:str(A)}
		with B.lock:B.buffer[_C].append(A);B._check_triggers(_C)
	def _check_triggers(A,record_type:str):
		D=record_type;E=time.time();B=_G;C=_F;F=sum(len(A)for A in A.buffer.values())
		if F>=A.buffer_size:B=_A;C='buffer_full'
		elif D==_C and A.buffer[_C]and any(A.get(_Q)for A in A.buffer[_C]if isinstance(A,dict)):B=_A;C='rate_limit_exceeded'
		elif D==_B and A.buffer[_B]and any(not A.get(_R)for A in A.buffer[_B]if isinstance(A,dict)):B=_A;C='function_error'
		else:
			G=min(1.,(E-A.last_sync_time)/(A.sync_interval/2))
			if random.random()<.05*G:B=_A;C='random_time_weighted'
		if B:threading.Thread(target=A.dispatch,args=(C,),daemon=_A).start()
	def queue(B,package,priority=_F):
		N='packages';M='commercial';L='system_info';K='rate_limit';I='system';H='type';C=package
		if not C:return _G
		if not isinstance(C,dict):B.add_function_call({'message':str(C)});return _A
		if _E not in C:C[_E]=datetime.now().isoformat()
		if H in C:
			D=C[H];A=C.get('data',{})
			if isinstance(A,dict)and I in A:
				J=A[I].get(_H);A.pop(I)
				if J:A[_H]=J
			if D==_O:B.add_function_call(A)
			elif D=='api_request':B.add_api_request(A)
			elif D==K:B.add_rate_limit(A)
			elif D==L:B.add_function_call({H:L,M:A.get(M),N:A.get(N),_E:C.get(_E)})
			elif D=='metrics':
				O=A
				for(G,F)in O.items():
					if isinstance(F,list):
						if G==_O:
							for E in F:B.add_function_call(E)
						elif G==K:
							for E in F:B.add_rate_limit(E)
						elif G=='request':
							for E in F:B.add_api_request(E)
			else:B.add_function_call(A)
		else:B.add_function_call(C)
		if priority=='high':B.dispatch('high_priority')
		return _A
	def dispatch(A,reason=_S):
		if not A.webhook_url:return _G
		with A.lock:
			if all(len(A)==0 for A in A.buffer.values()):return _G
			B={_B:A.buffer[_B].copy(),_D:A.buffer[_D].copy(),_C:A.buffer[_C].copy()};A.buffer={_B:[],_D:[],_C:[]};A.last_sync_time=time.time();A.sync_count+=1;A._save_config()
		try:from vnai.scope.profile import inspector as G;C=G.examine();D=C.get(_H,A.machine_id)
		except:C={_H:A.machine_id};D=A.machine_id
		E={'analytics_data':B,'metadata':{_E:datetime.now().isoformat(),_H:D,_I:A.sync_count,'trigger_reason':reason,'environment':C,'data_counts':{_B:len(B[_B]),_D:len(B[_D]),_C:len(B[_C])}}};F=A._send_data(E)
		if not F:
			with A.lock:
				A.failed_queue.append(E)
				if len(A.failed_queue)>10:A.failed_queue=A.failed_queue[-10:]
		return F
	def _send_data(A,payload):
		if not A.webhook_url:return _G
		try:B=requests.post(A.webhook_url,json=payload,timeout=5);return B.status_code==200
		except:return _G
	def retry_failed(A):
		if not A.failed_queue:return 0
		with A.lock:D=A.failed_queue.copy();A.failed_queue=[]
		B=0
		for C in D:
			if A._send_data(C):B+=1
			else:
				with A.lock:A.failed_queue.append(C)
		return B
	def configure(A,webhook_url):
		with A.lock:A.webhook_url=webhook_url;A._save_config();return _A
conduit=Conduit()
def track_function_call(function_name,source,execution_time,success=_A,error=_F,args=_F):
	E=error;A=args;C={_O:function_name,_P:source,_T:execution_time,_E:datetime.now().isoformat(),_R:success}
	if E:C['error']=E
	if A:
		B={}
		if isinstance(A,dict):
			for(F,D)in A.items():
				if isinstance(D,(str,int,float,bool)):B[F]=D
				else:B[F]=str(type(D))
		else:B={_J:str(A)}
		C['args']=B
	conduit.add_function_call(C)
def track_rate_limit(source,limit_type,limit_value,current_usage,is_exceeded):B=current_usage;A=limit_value;C={_P:source,'limit_type':limit_type,'limit_value':A,'current_usage':B,_Q:is_exceeded,_E:datetime.now().isoformat(),'usage_percentage':B/A*100 if A>0 else 0};conduit.add_rate_limit(C)
def track_api_request(endpoint,source,method,status_code,execution_time,request_size=0,response_size=0):A={'endpoint':endpoint,_P:source,'method':method,'status_code':status_code,_T:execution_time,_E:datetime.now().isoformat(),'request_size':request_size,'response_size':response_size};conduit.add_api_request(A)
def configure(webhook_url):return conduit.configure(webhook_url)
def sync_now():return conduit.dispatch(_S)
def retry_failed():return conduit.retry_failed()