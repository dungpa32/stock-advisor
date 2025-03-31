_C='category'
_B=True
_A=None
import time,threading,json
from datetime import datetime
from pathlib import Path
class Buffer:
	_instance=_A;_lock=threading.Lock()
	def __new__(A):
		with A._lock:
			if A._instance is _A:A._instance=super(Buffer,A).__new__(A);A._instance._initialize()
			return A._instance
	def _initialize(A):A.data=[];A.lock=threading.Lock();A.max_size=1000;A.backup_interval=300;A.home_dir=Path.home();A.project_dir=A.home_dir/'.vnstock';A.project_dir.mkdir(exist_ok=_B);A.data_dir=A.project_dir/'data';A.data_dir.mkdir(exist_ok=_B);A.backup_path=A.data_dir/'buffer_backup.json';A._load_from_backup();A._start_backup_thread()
	def _load_from_backup(A):
		if A.backup_path.exists():
			try:
				with open(A.backup_path,'r')as B:C=json.load(B)
				with A.lock:A.data=C
			except:pass
	def _save_to_backup(A):
		with A.lock:
			if not A.data:return
			try:
				with open(A.backup_path,'w')as B:json.dump(A.data,B)
			except:pass
	def _start_backup_thread(A):
		def B():
			while _B:time.sleep(A.backup_interval);A._save_to_backup()
		C=threading.Thread(target=B,daemon=_B);C.start()
	def add(A,item,category=_A):
		D='timestamp';C=category;B=item
		with A.lock:
			if isinstance(B,dict):
				if D not in B:B[D]=datetime.now().isoformat()
				if C:B[_C]=C
			A.data.append(B)
			if len(A.data)>A.max_size:A.data=A.data[-A.max_size:]
			if len(A.data)%100==0:A._save_to_backup()
			return len(A.data)
	def get(A,count=_A,category=_A):
		D=category;C=count
		with A.lock:
			if D:B=[A for A in A.data if A.get(_C)==D]
			else:B=A.data.copy()
			if C:return B[:C]
			else:return B
	def clear(A,category=_A):
		B=category
		with A.lock:
			if B:A.data=[A for A in A.data if A.get(_C)!=B]
			else:A.data=[]
			A._save_to_backup();return len(A.data)
	def size(A,category=_A):
		B=category
		with A.lock:
			if B:return len([A for A in A.data if A.get(_C)==B])
			else:return len(A.data)
buffer=Buffer()