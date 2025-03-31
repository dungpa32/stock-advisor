import streamlit as st
from vnstock import Vnstock
import pandas as pd
from datetime import datetime
import utils
import ui
from prophet import Prophet
import prophet

st.set_page_config(
    page_title="Phân tích Chứng khoán Việt Nam",
    page_icon="📈",
    layout="wide"
)

st.title("Dữ liệu Lịch sử Chứng khoán Việt Nam")

# Hiển thị sidebar và lấy tham số đầu vào
stock_symbol, start_date, end_date, rsi_period, macd_fast, macd_slow, macd_signal, submit = ui.display_sidebar()

# Xử lý khi nhấn nút "Hiển thị kết quả"
if submit:
    st.write(f"Đang phân tích dữ liệu cho mã {stock_symbol} từ {start_date} đến {end_date}")
    
    try:
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        with st.spinner('Đang tải dữ liệu...'):
            df = Vnstock().stock(symbol=stock_symbol, source='VCI')
            history = df.quote.history(start=start_str, end=end_str)

        if history is not None and not history.empty:
            with st.spinner('Đang tính toán các chỉ báo kỹ thuật...'):
                # Tính toán các chỉ báo kỹ thuật
                history['RSI'] = utils.calculate_rsi(history, rsi_period)
                history['MACD'], history['Signal'] = utils.calculate_macd(history, macd_fast, macd_slow, macd_signal)
                history['MACD_Histogram'] = history['MACD'] - history['Signal']
                history['EMA20'] = utils.calculate_ema(history, 20)
                history['EMA50'] = utils.calculate_ema(history, 50)
                history['BB_Upper'], history['BB_Middle'], history['BB_Lower'] = utils.calculate_bollinger_bands(history)
                history['EMA9'] = utils.calculate_ema(history, 9)
                history['VWAP'] = utils.calculate_vwap(history)
                
                # Tạo tín hiệu giao dịch
                signals = utils.generate_rsi_macd_signals(history)
                history = pd.concat([history, signals], axis=1)
            
            with st.spinner('Đang thực hiện dự đoán giá...'):
                # Dự đoán với Prophet
                prophet_df = history[['time', 'close']].rename(columns={'time': 'ds', 'close': 'y'})
                model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
                model.fit(prophet_df)
                future = model.make_future_dataframe(periods=30)  # Dự đoán 30 ngày tiếp theo
                forecast = model.predict(future)
            
            # Hiển thị kết quả
            ui.display_results(history, stock_symbol, start_date, end_date, forecast)
        
        else:
            st.warning(f"Không tìm thấy dữ liệu cho mã {stock_symbol} trong khoảng thời gian này.")
    
    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi lấy dữ liệu: {str(e)}")
        st.write("Vui lòng kiểm tra lại mã chứng khoán hoặc kết nối.")
else:
    st.info("Vui lòng nhập mã chứng khoán và chọn khoảng thời gian, sau đó nhấn 'Hiển thị kết quả'")

# Footer
st.markdown("---")
st.write(f"Cập nhật lần cuối: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.markdown(f"Powered by Streamlit, Vnstock, Prophet={prophet.__version__} và các thư viện phân tích kỹ thuật")
