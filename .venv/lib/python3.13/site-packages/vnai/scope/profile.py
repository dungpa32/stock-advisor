_W='type_confidence'
_V='detected_type'
_U='commercial_app'
_T='version'
_S='django'
_R='fastapi'
_Q='streamlit'
_P='indicators'
_O='commercial_indicators'
_N='likely_commercial'
_M='KAGGLE_KERNEL_RUN_TYPE'
_L='machine_id'
_K='domain'
_J='.git'
_I='backtesting'
_H='commercial_probability'
_G='timestamp'
_F='business_hours_usage'
_E='google.colab'
_D='unknown'
_C=False
_B=None
_A=True
import os,sys,platform,uuid,hashlib,psutil,threading,time,importlib.metadata
from datetime import datetime
import subprocess
from pathlib import Path
class Inspector:
	_instance=_B;_lock=_B
	def __new__(cls):
		import threading
		if cls._lock is _B:cls._lock=threading.Lock()
		with cls._lock:
			if cls._instance is _B:cls._instance=super(Inspector,cls).__new__(cls);cls._instance._initialize()
			return cls._instance
	def _initialize(self):self.cache={};self.cache_ttl=3600;self.last_examination=0;self.machine_id=_B;self._colab_auth_triggered=_C;self.home_dir=Path.home();self.project_dir=self.home_dir/'.vnstock';self.project_dir.mkdir(exist_ok=_A);self.id_dir=self.project_dir/'id';self.id_dir.mkdir(exist_ok=_A);self.machine_id_path=self.id_dir/'machine_id.txt';self.examine()
	def examine(self,force_refresh=_C):
		D='script';C='terminal';B='hosting_service';A='environment';current_time=time.time()
		if not force_refresh and current_time-self.last_examination<self.cache_ttl:return self.cache
		info={_G:datetime.now().isoformat(),'python_version':platform.python_version(),'os_name':platform.system(),'platform':platform.platform()};info[_L]=self.fingerprint()
		try:
			import importlib.util;ipython_spec=importlib.util.find_spec('IPython')
			if ipython_spec:
				from IPython import get_ipython;ipython=get_ipython()
				if ipython is not _B:
					info[A]='jupyter'
					if _E in sys.modules:info[B]='colab'
					elif _M in os.environ:info[B]='kaggle'
					else:info[B]='local_jupyter'
				elif sys.stdout.isatty():info[A]=C
				else:info[A]=D
			elif sys.stdout.isatty():info[A]=C
			else:info[A]=D
		except:info[A]=_D
		try:info['cpu_count']=os.cpu_count();info['memory_gb']=round(psutil.virtual_memory().total/1024**3,1)
		except:pass
		is_colab=_E in sys.modules
		if is_colab:info['is_colab']=_A;self.detect_colab_with_delayed_auth()
		try:info['commercial_usage']=self.enhanced_commercial_detection();info['project_context']=self.analyze_project_structure();info['git_info']=self.analyze_git_info();info['usage_pattern']=self.detect_usage_pattern();info['dependencies']=self.analyze_dependencies()
		except Exception as e:info['detection_error']=str(e)
		self.cache=info;self.last_examination=current_time;return info
	def fingerprint(self):
		if self.machine_id:return self.machine_id
		if self.machine_id_path.exists():
			try:
				with open(self.machine_id_path,'r')as f:self.machine_id=f.read().strip();return self.machine_id
			except:pass
		is_colab=self.detect_colab_with_delayed_auth()
		try:system_info=platform.node()+platform.platform()+platform.machine();self.machine_id=hashlib.md5(system_info.encode()).hexdigest()
		except:self.machine_id=str(uuid.uuid4())
		try:
			with open(self.machine_id_path,'w')as f:f.write(self.machine_id)
		except:pass
		return self.machine_id
	def detect_hosting(self):
		A='Google Colab';hosting_markers={'COLAB_GPU':A,_M:'Kaggle','BINDER_SERVICE_HOST':'Binder','CODESPACE_NAME':'GitHub Codespaces','STREAMLIT_SERVER_HEADLESS':'Streamlit Cloud','CLOUD_SHELL':'Cloud Shell'}
		for(env_var,host_name)in hosting_markers.items():
			if env_var in os.environ:return host_name
		if _E in sys.modules:return A
		return'local'
	def detect_commercial_usage(self):
		F='client';E='enterprise';D='dir_patterns';C='env_vars';B='file_patterns';A='env_domains';commercial_indicators={A:['.com','.io','.co',E,'corp','inc'],B:['invoice','payment','customer',F,'product','sale'],C:['COMPANY','BUSINESS','ENTERPRISE','CORPORATE'],D:['company','business',E,'corporate',F]};env_values=' '.join(os.environ.values()).lower();domain_match=any(domain in env_values for domain in commercial_indicators[A]);env_var_match=any(var in os.environ for var in commercial_indicators[C]);current_dir=os.getcwd().lower();dir_match=any(pattern in current_dir for pattern in commercial_indicators[D])
		try:files=[f.lower()for f in os.listdir()if os.path.isfile(f)];file_match=any(any(pattern in f for pattern in commercial_indicators[B])for f in files)
		except:file_match=_C
		indicators=[domain_match,env_var_match,dir_match,file_match];commercial_probability=sum(indicators)/len(indicators);return{_N:commercial_probability>.3,_H:commercial_probability,_O:{'domain_match':domain_match,'env_var_match':env_var_match,'dir_match':dir_match,'file_match':file_match}}
	def scan_packages(self):
		A='financetoolkit';package_groups={'vnstock_family':['vnstock','vnstock3','vnstock_ezchart','vnstock_data_pro','vnstock_market_data_pipeline','vnstock_ta','vnii','vnai'],'analytics':['openbb','pandas_ta'],'static_charts':['matplotlib','seaborn','altair'],'dashboard':[_Q,'voila','panel','shiny','dash'],'interactive_charts':['mplfinance','plotly','plotline','bokeh','pyecharts','highcharts-core','highcharts-stock','mplchart'],'datafeed':['yfinance','alpha_vantage','pandas-datareader','investpy'],'official_api':['ssi-fc-data','ssi-fctrading'],'risk_return':['pyfolio','empyrical','quantstats',A],'machine_learning':['scipy','sklearn','statsmodels','pytorch','tensorflow','keras','xgboost'],_P:['stochastic','talib','tqdm','finta',A,'tulipindicators'],_I:['vectorbt',_I,'bt','zipline','pyalgotrade','backtrader','pybacktest','fastquant','lean','ta','finmarketpy','qstrader'],'server':[_R,'flask','uvicorn','gunicorn'],'framework':['lightgbm','catboost',_S]};installed={}
		for(category,packages)in package_groups.items():
			installed[category]=[]
			for pkg in packages:
				try:version=importlib.metadata.version(pkg);installed[category].append({'name':pkg,_T:version})
				except:pass
		return installed
	def setup_vnstock_environment(self):
		env_file=self.id_dir/'environment.json';env_data={'accepted_agreement':_A,_G:datetime.now().isoformat(),_L:self.fingerprint()}
		try:
			with open(env_file,'w')as f:import json;json.dump(env_data,f)
			return _A
		except Exception as e:print(f"Failed to set up vnstock environment: {e}");return _C
	def detect_colab_with_delayed_auth(self,immediate=_C):
		is_colab=_E in sys.modules
		if is_colab and not self._colab_auth_triggered:
			if immediate:
				self._colab_auth_triggered=_A;user_id=self.get_or_create_user_id()
				if user_id and user_id!=self.machine_id:
					self.machine_id=user_id
					try:
						with open(self.machine_id_path,'w')as f:f.write(user_id)
					except:pass
			else:
				def delayed_auth():
					time.sleep(300);user_id=self.get_or_create_user_id()
					if user_id and user_id!=self.machine_id:
						self.machine_id=user_id
						try:
							with open(self.machine_id_path,'w')as f:f.write(user_id)
						except:pass
				thread=threading.Thread(target=delayed_auth,daemon=_A);thread.start()
		return is_colab
	def get_or_create_user_id(self):
		if self._colab_auth_triggered:return self.machine_id
		try:
			from google.colab import drive;print('\n📋 Kết nối tài khoản Google Drive để lưu các thiết lập của dự án.');print('Dữ liệu phiên làm việc với Colab của bạn sẽ bị xóa nếu không lưu trữ vào Google Drive.\n');self._colab_auth_triggered=_A;drive.mount('/content/drive');id_path='/content/drive/MyDrive/.vnstock/user_id.txt'
			if os.path.exists(id_path):
				with open(id_path,'r')as f:return f.read().strip()
			else:
				user_id=str(uuid.uuid4());os.makedirs(os.path.dirname(id_path),exist_ok=_A)
				with open(id_path,'w')as f:f.write(user_id)
				return user_id
		except Exception as e:return self.machine_id
	def analyze_project_structure(self):
		E='root_dirs';D='manage.py';C='wsgi.py';B='data_science';A='app.py';current_dir=os.getcwd();project_indicators={_U:['app','services','products','customers','billing'],'financial_tool':['portfolio',_I,'trading','strategy'],B:['models','notebooks','datasets','visualization'],'educational':['examples','lectures','assignments','slides']};project_type={}
		for(category,markers)in project_indicators.items():
			match_count=0
			for marker in markers:
				if os.path.exists(os.path.join(current_dir,marker)):match_count+=1
			if len(markers)>0:project_type[category]=match_count/len(markers)
		try:
			root_files=[f for f in os.listdir(current_dir)if os.path.isfile(os.path.join(current_dir,f))];root_dirs=[d for d in os.listdir(current_dir)if os.path.isdir(os.path.join(current_dir,d))];file_markers={'python_project':['setup.py','pyproject.toml','requirements.txt'],B:['notebook.ipynb','.ipynb_checkpoints'],'web_app':[A,C,D,'server.py'],'finance_app':['portfolio.py','trading.py','backtest.py']};file_project_type=_D
			for(ptype,markers)in file_markers.items():
				if any(marker in root_files for marker in markers):file_project_type=ptype;break
			frameworks=[];framework_markers={_S:[D,'settings.py'],'flask':[A,C],_Q:['streamlit_app.py',A],_R:['main.py',A]}
			for(framework,markers)in framework_markers.items():
				if any(marker in root_files for marker in markers):frameworks.append(framework)
		except Exception as e:root_files=[];root_dirs=[];file_project_type=_D;frameworks=[]
		return{'project_dir':current_dir,_V:max(project_type.items(),key=lambda x:x[1])[0]if project_type else _D,'file_type':file_project_type,'is_git_repo':_J in(root_dirs if E in locals()else[]),'frameworks':frameworks,'file_count':len(root_files)if'root_files'in locals()else 0,'directory_count':len(root_dirs)if E in locals()else 0,_W:project_type}
	def analyze_git_info(self):
		I='license_type';H='has_license';G='repo_path';F='rev-parse';E='/';D='has_git';C=':';B='git';A='@'
		try:
			result=subprocess.run([B,F,'--is-inside-work-tree'],capture_output=_A,text=_A)
			if result.returncode!=0:return{D:_C}
			repo_root=subprocess.run([B,F,'--show-toplevel'],capture_output=_A,text=_A);repo_path=repo_root.stdout.strip()if repo_root.stdout else _B;repo_name=os.path.basename(repo_path)if repo_path else _B;has_license=_C;license_type=_D
			if repo_path:
				license_files=[os.path.join(repo_path,'LICENSE'),os.path.join(repo_path,'LICENSE.txt'),os.path.join(repo_path,'LICENSE.md')]
				for license_file in license_files:
					if os.path.exists(license_file):
						has_license=_A
						try:
							with open(license_file,'r')as f:
								content=f.read().lower()
								if'mit license'in content:license_type='MIT'
								elif'apache license'in content:license_type='Apache'
								elif'gnu general public'in content:license_type='GPL'
								elif'bsd 'in content:license_type='BSD'
						except:pass
						break
			remote=subprocess.run([B,'config','--get','remote.origin.url'],capture_output=_A,text=_A);remote_url=remote.stdout.strip()if remote.stdout else _B
			if remote_url:
				remote_url=remote_url.strip();domain=_B
				if remote_url:
					if remote_url.startswith('git@')or A in remote_url and C in remote_url.split(A)[1]:domain=remote_url.split(A)[1].split(C)[0]
					elif remote_url.startswith('http'):
						url_parts=remote_url.split('//')
						if len(url_parts)>1:
							auth_and_domain=url_parts[1].split(E,1)[0]
							if A in auth_and_domain:domain=auth_and_domain.split(A)[-1]
							else:domain=auth_and_domain
					else:
						import re;domain_match=re.search('@([^:/]+)|https?://(?:[^@/]+@)?([^/]+)',remote_url)
						if domain_match:domain=domain_match.group(1)or domain_match.group(2)
				owner=_B;repo_name=_B
				if domain:
					if'github'in domain:
						if C in remote_url and A in remote_url:
							parts=remote_url.split(C)[-1].split(E)
							if len(parts)>=2:owner=parts[0];repo_name=parts[1].replace(_J,'')
						else:
							url_parts=remote_url.split('//')
							if len(url_parts)>1:
								path_parts=url_parts[1].split(E)
								if len(path_parts)>=3:
									domain_part=path_parts[0]
									if A in domain_part:owner_index=1
									else:owner_index=1
									if len(path_parts)>owner_index:owner=path_parts[owner_index]
									if len(path_parts)>owner_index+1:repo_name=path_parts[owner_index+1].replace(_J,'')
				commit_count=subprocess.run([B,'rev-list','--count','HEAD'],capture_output=_A,text=_A);branch_count=subprocess.run([B,'branch','--list'],capture_output=_A,text=_A);branch_count=len(branch_count.stdout.strip().split('\n'))if branch_count.stdout else 0;return{_K:domain,'owner':owner,'commit_count':int(commit_count.stdout.strip())if commit_count.stdout else 0,'branch_count':branch_count,D:_A,G:repo_path if G in locals()else _B,'repo_name':repo_name,H:has_license if H in locals()else _C,I:license_type if I in locals()else _D}
		except Exception as e:pass
		return{D:_C}
	def detect_usage_pattern(self):current_time=datetime.now();is_weekday=current_time.weekday()<5;hour=current_time.hour;is_business_hours=9<=hour<=18;return{_F:is_weekday and is_business_hours,'weekday':is_weekday,'hour':hour,_G:current_time.isoformat()}
	def enhanced_commercial_detection(self):
		basic=self.detect_commercial_usage()
		try:
			project_files=os.listdir(os.getcwd());commercial_frameworks=['django-oscar','opencart','magento','saleor','odoo','shopify','woocommerce'];framework_match=_C
			for framework in commercial_frameworks:
				if any(framework in f for f in project_files):framework_match=_A;break
			db_files=[f for f in project_files if'database'in f.lower()or'db_config'in f.lower()or f.endswith('.db')];has_database=len(db_files)>0
		except:framework_match=_C;has_database=_C
		domain_check=self.analyze_git_info();domain_is_commercial=_C
		if domain_check and domain_check.get(_K):commercial_tlds=['.com','.io','.co','.org','.net'];domain_is_commercial=any(tld in domain_check[_K]for tld in commercial_tlds)
		project_structure=self.analyze_project_structure();indicators=[basic[_H],framework_match,has_database,domain_is_commercial,project_structure.get(_W,{}).get(_U,0),self.detect_usage_pattern()[_F]];indicators=[i for i in indicators if i is not _B]
		if indicators:score=sum(1. if isinstance(i,bool)and i else i if isinstance(i,(int,float))else 0 for i in indicators)/len(indicators)
		else:score=0
		return{_H:score,_N:score>.4,_P:{'basic_indicators':basic[_O],'framework_match':framework_match,'has_database':has_database,'domain_is_commercial':domain_is_commercial,'project_structure':project_structure.get(_V),_F:self.detect_usage_pattern()[_F]}}
	def analyze_dependencies(self):
		A='has_commercial_deps'
		try:
			import pkg_resources;enterprise_packages=['snowflake-connector-python','databricks','azure','aws','google-cloud','stripe','atlassian','salesforce','bigquery','tableau','sap'];commercial_deps=[]
			for pkg in pkg_resources.working_set:
				if any(ent in pkg.key for ent in enterprise_packages):commercial_deps.append({'name':pkg.key,_T:pkg.version})
			return{A:len(commercial_deps)>0,'commercial_deps_count':len(commercial_deps),'commercial_deps':commercial_deps}
		except:return{A:_C}
inspector=Inspector()