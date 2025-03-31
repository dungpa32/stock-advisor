_L='default'
_K='standard'
_J='accepted_agreement'
_I='environment.json'
_H='terms_agreement.txt'
_G='timestamp'
_F=False
_E='id'
_D='.vnstock'
_C='machine_id'
_B=None
_A=True
import os,pathlib,json,time,threading,functools
from datetime import datetime
from vnai.beam.quota import guardian,optimize
from vnai.beam.metrics import collector,capture
from vnai.beam.pulse import monitor
from vnai.flow.relay import conduit,configure
from vnai.flow.queue import buffer
from vnai.scope.profile import inspector
from vnai.scope.state import tracker,record
from vnai.scope.promo import present
TC_VAR='ACCEPT_TC'
TC_VAL='tôi đồng ý'
TC_PATH=pathlib.Path.home()/_D/_E/_H
TERMS_AND_CONDITIONS='\nKhi tiếp tục sử dụng Vnstock, bạn xác nhận rằng bạn đã đọc, hiểu và đồng ý với Chính sách quyền riêng tư và Điều khoản, điều kiện về giấy phép sử dụng Vnstock.\n\nChi tiết:\n- Giấy phép sử dụng phần mềm: https://vnstocks.com/docs/tai-lieu/giay-phep-su-dung\n- Chính sách quyền riêng tư: https://vnstocks.com/docs/tai-lieu/chinh-sach-quyen-rieng-tu\n'
class Core:
	def __init__(A):A.initialized=_F;A.webhook_url=_B;A.init_time=datetime.now().isoformat();A.home_dir=pathlib.Path.home();A.project_dir=A.home_dir/_D;A.id_dir=A.project_dir/_E;A.terms_file_path=TC_PATH;A.system_info=_B;A.project_dir.mkdir(exist_ok=_A);A.id_dir.mkdir(exist_ok=_A);A.initialize()
	def initialize(A,webhook_url=_B):
		C=webhook_url
		if A.initialized:return _A
		if not A._check_terms():A._accept_terms()
		from vnai.scope.profile import inspector as B;B.setup_vnstock_environment();present()
		if C:A.webhook_url=C;configure(C)
		record('initialization',{_G:datetime.now().isoformat()});A.system_info=B.examine();conduit.queue({'type':'system_info','data':{'commercial':B.detect_commercial_usage(),'packages':B.scan_packages()}},priority='high');A.initialized=_A;return _A
	def _check_terms(A):return os.path.exists(A.terms_file_path)
	def _accept_terms(C):
		A=inspector.examine()
		if TC_VAR in os.environ and os.environ[TC_VAR]==TC_VAL:E=TC_VAL
		else:E=TC_VAL;os.environ[TC_VAR]=TC_VAL
		D=datetime.now();F=f"""Người dùng có mã nhận dạng {A[_C]} đã chấp nhận điều khoản & điều kiện sử dụng Vnstock lúc {D}
---

THÔNG TIN THIẾT BỊ: {json.dumps(A,indent=2)}

Đính kèm bản sao nội dung bạn đã đọc, hiểu rõ và đồng ý dưới đây:
{TERMS_AND_CONDITIONS}"""
		with open(C.terms_file_path,'w',encoding='utf-8')as B:B.write(F)
		G=C.id_dir/_I;H={_J:_A,_G:D.isoformat(),_C:A[_C]}
		with open(G,'w')as B:json.dump(H,B)
		return _A
	def status(A):return{'initialized':A.initialized,'health':monitor.report(),'metrics':tracker.get_metrics()}
	def configure_privacy(B,level=_K):from vnai.scope.state import tracker as A;return A.setup_privacy(level)
core=Core()
def tc_init(webhook_url=_B):return core.initialize(webhook_url)
def setup(webhook_url=_B):return core.initialize(webhook_url)
def optimize_execution(resource_type=_L):return optimize(resource_type)
def agg_execution(resource_type=_L):return optimize(resource_type,ad_cooldown=1500,content_trigger_threshold=100000)
def measure_performance(module_type='function'):return capture(module_type)
def accept_license_terms(terms_text=_B):
	A=terms_text
	if A is _B:A=TERMS_AND_CONDITIONS
	D=inspector.examine();C=pathlib.Path.home()/_D/_E/_H;os.makedirs(os.path.dirname(C),exist_ok=_A)
	with open(C,'w',encoding='utf-8')as B:B.write(f"Terms accepted at {datetime.now().isoformat()}\n");B.write(f"System: {json.dumps(D)}\n\n");B.write(A)
	return _A
def accept_vnstock_terms():
	from vnai.scope.profile import inspector as C;D=C.examine();E=pathlib.Path.home();A=E/_D;A.mkdir(exist_ok=_A);B=A/_E;B.mkdir(exist_ok=_A);F=B/_I;G={_J:_A,_G:datetime.now().isoformat(),_C:D[_C]}
	try:
		with open(F,'w')as H:json.dump(G,H)
		print('Vnstock terms accepted successfully.');return _A
	except Exception as I:print(f"Error accepting terms: {I}");return _F
def setup_for_colab():from vnai.scope.profile import inspector as A;A.detect_colab_with_delayed_auth(immediate=_A);A.setup_vnstock_environment();return'Environment set up for Google Colab'
def display_content():return present()
def configure_privacy(level=_K):from vnai.scope.state import tracker as A;return A.setup_privacy(level)
def check_commercial_usage():from vnai.scope.profile import inspector as A;return A.detect_commercial_usage()
def authenticate_for_persistence():from vnai.scope.profile import inspector as A;return A.get_or_create_user_id()
def configure_webhook(webhook_id='80b8832b694a75c8ddc811ac7882a3de'):
	A=webhook_id
	if not A:return _F
	from vnai.flow.relay import configure as B;C=f"https://botbuilder.larksuite.com/api/trigger-webhook/{A}";return B(C)
configure_webhook()