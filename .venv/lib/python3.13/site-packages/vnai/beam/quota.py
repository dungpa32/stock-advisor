_G='resource_type'
_F=False
_E=True
_D=None
_C='default'
_B='hour'
_A='min'
import time,functools,threading
from collections import defaultdict
from datetime import datetime
class RateLimitExceeded(Exception):
	def __init__(A,resource_type,limit_type=_A,current_usage=_D,limit_value=_D,retry_after=_D):
		D=resource_type;B=retry_after;A.resource_type=D;A.limit_type=limit_type;A.current_usage=current_usage;A.limit_value=limit_value;A.retry_after=B;C=f"Bạn đã gửi quá nhiều request tới {D}. "
		if B:C+=f"Vui lòng thử lại sau {round(B)} giây."
		else:C+='Vui lòng thêm thời gian chờ giữa các lần gửi request.'
		super().__init__(C)
class Guardian:
	_instance=_D;_lock=threading.Lock()
	def __new__(A):
		with A._lock:
			if A._instance is _D:A._instance=super(Guardian,A).__new__(A);A._instance._initialize()
			return A._instance
	def _initialize(A):A.resource_limits=defaultdict(lambda:defaultdict(int));A.usage_counters=defaultdict(lambda:defaultdict(list));A.resource_limits[_C]={_A:60,_B:3000};A.resource_limits['TCBS']={_A:60,_B:3000};A.resource_limits['VCI']={_A:60,_B:3000};A.resource_limits['VCI.ext']={_A:600,_B:36000};A.resource_limits['VND.ext']={_A:600,_B:36000};A.resource_limits['CAF.ext']={_A:600,_B:36000};A.resource_limits['SPL.ext']={_A:600,_B:36000};A.resource_limits['VDS.ext']={_A:600,_B:36000};A.resource_limits['FAD.ext']={_A:600,_B:36000}
	def verify(B,operation_id,resource_type=_C):
		M='is_exceeded';L='current_usage';K='limit_value';J='limit_type';I='rate_limit';A=resource_type;D=time.time();C=B.resource_limits.get(A,B.resource_limits[_C]);N=D-60;B.usage_counters[A][_A]=[A for A in B.usage_counters[A][_A]if A>N];F=len(B.usage_counters[A][_A]);O=F>=C[_A]
		if O:from vnai.beam.metrics import collector as G;G.record(I,{_G:A,J:_A,K:C[_A],L:F,M:_E},priority='high');raise RateLimitExceeded(resource_type=A,limit_type=_A,current_usage=F,limit_value=C[_A],retry_after=60-D%60)
		P=D-3600;B.usage_counters[A][_B]=[A for A in B.usage_counters[A][_B]if A>P];H=len(B.usage_counters[A][_B]);E=H>=C[_B];from vnai.beam.metrics import collector as G;G.record(I,{_G:A,J:_B if E else _A,K:C[_B]if E else C[_A],L:H if E else F,M:E})
		if E:raise RateLimitExceeded(resource_type=A,limit_type=_B,current_usage=H,limit_value=C[_B],retry_after=3600-D%3600)
		B.usage_counters[A][_A].append(D);B.usage_counters[A][_B].append(D);return _E
	def usage(A,resource_type=_C):B=resource_type;D=time.time();C=A.resource_limits.get(B,A.resource_limits[_C]);E=D-60;F=D-3600;A.usage_counters[B][_A]=[A for A in A.usage_counters[B][_A]if A>E];A.usage_counters[B][_B]=[A for A in A.usage_counters[B][_B]if A>F];G=len(A.usage_counters[B][_A]);H=len(A.usage_counters[B][_B]);I=G/C[_A]*100 if C[_A]>0 else 0;J=H/C[_B]*100 if C[_B]>0 else 0;return max(I,J)
	def get_limit_status(B,resource_type=_C):K='reset_in_seconds';J='remaining';I='percentage';H='limit';G='usage';C=resource_type;D=time.time();A=B.resource_limits.get(C,B.resource_limits[_C]);L=D-60;M=D-3600;E=len([A for A in B.usage_counters[C][_A]if A>L]);F=len([A for A in B.usage_counters[C][_B]if A>M]);return{_G:C,'minute_limit':{G:E,H:A[_A],I:E/A[_A]*100 if A[_A]>0 else 0,J:max(0,A[_A]-E),K:60-D%60},'hour_limit':{G:F,H:A[_B],I:F/A[_B]*100 if A[_B]>0 else 0,J:max(0,A[_B]-F),K:3600-D%3600}}
guardian=Guardian()
class CleanErrorContext:
	_last_message_time=0;_message_cooldown=5
	def __enter__(A):return A
	def __exit__(C,exc_type,exc_val,exc_tb):
		A=exc_val
		if exc_type is RateLimitExceeded:
			B=time.time()
			if B-CleanErrorContext._last_message_time>=CleanErrorContext._message_cooldown:print(f"\n⚠️ {str(A)}\n");CleanErrorContext._last_message_time=B
			import sys;sys.exit(f"Rate limit exceeded. {str(A)} Process terminated.");return _F
		return _F
def optimize(resource_type=_C,loop_threshold=10,time_window=5,ad_cooldown=150,content_trigger_threshold=3,max_retries=2,backoff_factor=2,debug=_F):
	H=debug;G=ad_cooldown;F=resource_type;E=backoff_factor;D=max_retries;C=content_trigger_threshold;B=time_window;A=loop_threshold
	if callable(F):I=F;return _create_wrapper(I,_C,A,B,G,C,D,E,H)
	if A<2:raise ValueError(f"loop_threshold must be at least 2, got {A}")
	if B<=0:raise ValueError(f"time_window must be positive, got {B}")
	if C<1:raise ValueError(f"content_trigger_threshold must be at least 1, got {C}")
	if D<0:raise ValueError(f"max_retries must be non-negative, got {D}")
	if E<=0:raise ValueError(f"backoff_factor must be positive, got {E}")
	def J(func):return _create_wrapper(func,F,A,B,G,C,D,E,H)
	return J
def _create_wrapper(func,resource_type,loop_threshold,time_window,ad_cooldown,content_trigger_threshold,max_retries,backoff_factor,debug):
	X=max_retries;W=content_trigger_threshold;P=time_window;K=resource_type;H=debug;A=func;B=[];I=0;E=0;F=_F;Q=time.time();c=1800
	@functools.wraps(A)
	def C(*d,**e):
		b='timestamp';a='environment';V='error';O='function';N='loop';nonlocal I,E,F,Q;C=time.time();R=_F
		if C-Q>c:F=_F;Q=C
		G=0
		while _E:
			B.append(C)
			while B and C-B[0]>P:B.pop(0)
			S=len(B)>=loop_threshold
			if H and S:print(f"[OPTIMIZE] Đã phát hiện vòng lặp cho {A.__name__}: {len(B)} lần gọi trong {P}s")
			if S:
				E+=1
				if H:print(f"[OPTIMIZE] Số lần phát hiện vòng lặp liên tiếp: {E}/{W}")
			else:E=0
			f=E>=W and C-I>=ad_cooldown and not F
			if f:
				I=C;E=0;R=_E;F=_E
				if H:print(f"[OPTIMIZE] Đã kích hoạt nội dung cho {A.__name__}")
				try:
					from vnai.scope.promo import manager as J
					try:from vnai.scope.profile import inspector as T;U=T.examine().get(a,_D);J.present_content(environment=U,context=N)
					except ImportError:J.present_content(context=N)
				except ImportError:print(f"Phát hiện vòng lặp: Hàm '{A.__name__}' đang được gọi trong một vòng lặp")
				except Exception as D:
					if H:print(f"[OPTIMIZE] Lỗi khi hiển thị nội dung: {str(D)}")
			try:
				with CleanErrorContext():guardian.verify(A.__name__,K)
			except RateLimitExceeded as D:
				from vnai.beam.metrics import collector as L;L.record(V,{O:A.__name__,V:str(D),'context':'resource_verification',_G:K,'retry_attempt':G},priority='high')
				if not F:
					try:
						from vnai.scope.promo import manager as J
						try:from vnai.scope.profile import inspector as T;U=T.examine().get(a,_D);J.present_content(environment=U,context=N);F=_E;I=C
						except ImportError:J.present_content(context=N);F=_E;I=C
					except Exception:pass
				if G<X:
					M=backoff_factor**G;G+=1
					if hasattr(D,'retry_after')and D.retry_after:M=min(M,D.retry_after)
					if H:print(f"[OPTIMIZE] Đã đạt giới hạn tốc độ cho {A.__name__}, thử lại sau {M} giây (lần thử {G}/{X})")
					time.sleep(M);continue
				else:raise
			g=time.time();Y=_F;Z=_D
			try:h=A(*d,**e);Y=_E;return h
			except Exception as D:Z=str(D);raise
			finally:
				i=time.time()-g
				try:
					from vnai.beam.metrics import collector as L;L.record(O,{O:A.__name__,_G:K,'execution_time':i,'success':Y,V:Z,'in_loop':S,'loop_depth':len(B),'content_triggered':R,b:datetime.now().isoformat(),'retry_count':G if G>0 else _D})
					if R:L.record('ad_opportunity',{O:A.__name__,_G:K,'call_frequency':len(B)/P,'consecutive_loops':E,b:datetime.now().isoformat()})
				except ImportError:pass
			break
	return C
def rate_limit_status(resource_type=_C):return guardian.get_limit_status(resource_type)