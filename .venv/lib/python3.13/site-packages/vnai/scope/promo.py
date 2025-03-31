_G='init'
_F='simple'
_E='markdown'
_D=True
_C=None
_B='terminal'
_A='html'
import requests
from datetime import datetime
import random,threading,time,urllib.parse
class ContentManager:
	_instance=_C;_lock=threading.Lock()
	def __new__(A):
		with A._lock:
			if A._instance is _C:A._instance=super(ContentManager,A).__new__(A);A._instance._initialize()
			return A._instance
	def _initialize(A):A.last_display=0;A.display_interval=86400;A.content_base_url='https://vnstock-beam.hf.space/content-delivery';A.target_url='https://vnstocks.com/lp-khoa-hoc-python-chung-khoan';A.image_url='https://vnstocks.com/img/trang-chu-vnstock-python-api-phan-tich-giao-dich-chung-khoan.jpg';A._start_periodic_display()
	def _start_periodic_display(A):
		def B():
			while _D:
				B=random.randint(7200,21600);time.sleep(B);C=time.time()
				if C-A.last_display>=A.display_interval:A.present_content(context='periodic')
		C=threading.Thread(target=B,daemon=_D);C.start()
	def fetch_remote_content(B,context=_G,html=_D):
		try:
			C={'context':context,_A:'true'if html else'false'};D=f"{B.content_base_url}?{urllib.parse.urlencode(C)}";A=requests.get(D,timeout=3)
			if A.status_code==200:return A.text
			return
		except:return
	def present_content(C,environment=_C,context=_G):
		I='jupyter';H='unknown';E=context;A=environment;C.last_display=time.time()
		if A is _C:
			try:from vnai.scope.profile import inspector as J;A=J.examine().get('environment',H)
			except:A=H
		if A==I:B=C.fetch_remote_content(context=E,html=_D)
		else:B=C.fetch_remote_content(context=E,html=False)
		D=C._generate_fallback_content(E)
		if A==I:
			try:
				from IPython.display import display as F,HTML as G,Markdown as K
				if B:F(G(B))
				else:
					try:F(K(D[_E]))
					except:F(G(D[_A]))
			except:pass
		elif A==_B:
			if B:print(B)
			else:print(D[_B])
		else:print(D[_F])
	def _generate_fallback_content(B,context):
		A={_A:'',_E:'',_B:'',_F:''}
		if context=='loop':A[_A]=f'''
            <div style="border: 1px solid #e74c3c; padding: 15px; border-radius: 5px; margin: 10px 0;">
                <h3 style="color: #e74c3c;">⚠️ Bạn đang sử dụng vòng lặp với quá nhiều requests</h3>
                <p>Để tránh bị giới hạn tốc độ và tối ưu hiệu suất:</p>
                <ul>
                    <li>Thêm thời gian chờ giữa các lần gọi API</li>
                    <li>Sử dụng xử lý theo batch thay vì lặp liên tục</li>
                    <li>Tham gia gói tài trợ <a href="https://vnstocks.com/insiders-program" style="color: #3498db;">Vnstock Insider</a> để tăng 5X giới hạn API</li>
                </ul>
            </div>
            ''';A[_E]='\n## ⚠️ Bạn đang sử dụng vòng lặp với quá nhiều requests\n\nĐể tránh bị giới hạn tốc độ và tối ưu hiệu suất:\n* Thêm thời gian chờ giữa các lần gọi API\n* Sử dụng xử lý theo batch thay vì lặp liên tục\n* Tham gia gói tài trợ [Vnstock Insider](https://vnstocks.com/insiders-program) để tăng 5X giới hạn API\n            ';A[_B]='\n╔═════════════════════════════════════════════════════════════════╗\n║                                                                 ║\n║   🚫 ĐANG BỊ CHẶN BỞI GIỚI HẠN API? GIẢI PHÁP Ở ĐÂY!            ║\n║                                                                 ║\n║   ✓ Tăng ngay 500% tốc độ gọi API - Không còn lỗi RateLimit     ║\n║   ✓ Tiết kiệm 85% thời gian chờ đợi giữa các request            ║\n║                                                                 ║\n║   ➤ NÂNG CẤP NGAY VỚI GÓI TÀI TRỢ VNSTOCK:                      ║\n║     https://vnstocks.com/insiders-program                       ║\n║                                                                 ║\n╚═════════════════════════════════════════════════════════════════╝\n            ';A[_F]='🚫 Đang bị giới hạn API? Tăng tốc độ gọi API lên 500% với gói Vnstock Insider: https://vnstocks.com/insiders-program'
		else:A[_A]=f'''
            <div style="border: 1px solid #3498db; padding: 15px; border-radius: 5px; margin: 10px 0;">
                <h3 style="color: #3498db;">👋 Chào mừng bạn đến với Vnstock!</h3>
                <p>Cảm ơn bạn đã sử dụng thư viện phân tích chứng khoán #1 tại Việt Nam cho Python</p>
                <ul>
                    <li>Tài liệu: <a href="https://vnstocks.com/docs/category/s%E1%BB%95-tay-h%C6%B0%E1%BB%9Bng-d%E1%BA%ABn" style="color: #3498db;">vnstocks.com/docs</a></li>
                    <li>Cộng đồng: <a href="https://www.facebook.com/groups/vnstock.official" style="color: #3498db;">vnstocks.com/community</a></li>
                </ul>
                <p>Khám phá các tính năng mới nhất và tham gia cộng đồng để nhận hỗ trợ.</p>
            </div>
            ''';A[_E]='\n## 👋 Chào mừng bạn đến với Vnstock!\n\nCảm ơn bạn đã sử dụng package phân tích chứng khoán #1 tại Việt Nam\n\n* Tài liệu: [vnstocks.com/docs](https://vnstocks.com/docs)\n* Cộng đồng: [vnstocks.com/community](https://vnstocks.com/community)\n\nKhám phá các tính năng mới nhất và tham gia cộng đồng để nhận hỗ trợ.\n            ';A[_B]='\n╔══════════════════════════════════════════════════════════╗\n║                                                          ║\n║  👋 Chào mừng bạn đến với Vnstock!                       ║\n║                                                          ║\n║  Cảm ơn bạn đã sử dụng package phân tích                 ║\n║  chứng khoán #1 tại Việt Nam                             ║\n║                                                          ║\n║  ✓ Tài liệu: https://vnstocks.com/docs                   ║\n║  ✓ Cộng đồng: https://vnstocks.com/community             ║\n║                                                          ║\n║  Khám phá các tính năng mới nhất và tham gia             ║\n║  cộng đồng để nhận hỗ trợ.                               ║\n║                                                          ║\n╚══════════════════════════════════════════════════════════╝\n            ';A[_F]='👋 Chào mừng bạn đến với Vnstock! Tài liệu: https://vnstocks.com/docs | Cộng đồng: https://vnstocks.com/community'
		return A
manager=ContentManager()
def present(context=_G):return manager.present_content(context=context)