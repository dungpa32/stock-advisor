import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import utils


def display_sidebar():
    """
    Hiển thị sidebar cho người dùng nhập các tham số
    
    Returns:
        tuple: (stock_symbol, start_date, end_date, rsi_period, macd_fast, macd_slow, macd_signal, submit)
    """
    with st.sidebar:
        st.header("Tham số đầu vào")
        
        stock_symbol = st.text_input("Mã chứng khoán (VN)", "VNM")
        
        # Thiết lập ngày mặc định (1 năm gần đây)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # Chọn khoảng thời gian
        date_col1, date_col2 = st.columns(2)
        with date_col1:
            start_date = st.date_input("Ngày bắt đầu", start_date)
        with date_col2:
            end_date = st.date_input("Ngày kết thúc", end_date)
        
        st.subheader("Tùy chỉnh chỉ báo kỹ thuật")
        
        # RSI
        rsi_period = st.slider("Chu kỳ RSI", min_value=7, max_value=30, value=14, step=1)
        
        # MACD
        macd_col1, macd_col2 = st.columns(2)
        with macd_col1:
            macd_fast = st.slider("MACD EMA nhanh", min_value=8, max_value=20, value=12, step=1)
            macd_signal = st.slider("MACD Signal", min_value=5, max_value=15, value=9, step=1)
        with macd_col2:
            macd_slow = st.slider("MACD EMA chậm", min_value=20, max_value=35, value=26, step=1)
        
        submit = st.button("Hiển thị kết quả", use_container_width=True)
        
    return stock_symbol, start_date, end_date, rsi_period, macd_fast, macd_slow, macd_signal, submit


def display_results(history, stock_symbol, start_date, end_date, forecast):
    """
    Hiển thị kết quả phân tích và dự đoán
    
    Args:
        history (DataFrame): DataFrame chứa dữ liệu lịch sử và các chỉ báo
        stock_symbol (str): Mã chứng khoán
        start_date (datetime): Ngày bắt đầu
        end_date (datetime): Ngày kết thúc
        forecast (DataFrame): DataFrame chứa dự đoán từ Prophet
    """
    # Tạo các tab hiển thị
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Tổng quan", 
        "Biểu đồ giá & Chỉ báo", 
        "Tín hiệu giao dịch", 
        "Dự đoán giá", 
        "Dữ liệu chi tiết"
    ])
    
    with tab1:
        display_overview(history, stock_symbol, start_date, end_date)
    
    with tab2:
        display_charts(history, stock_symbol)
    
    with tab3:
        display_signals(history, stock_symbol)
    
    with tab4:
        display_forecast(forecast, stock_symbol, history)
    
    with tab5:
        display_data_table(history)


def display_overview(history, stock_symbol, start_date, end_date):
    """
    Hiển thị tổng quan về cổ phiếu
    
    Args:
        history (DataFrame): DataFrame chứa dữ liệu lịch sử
        stock_symbol (str): Mã chứng khoán
        start_date (datetime): Ngày bắt đầu
        end_date (datetime): Ngày kết thúc
    """
    st.header(f"Tổng quan về {stock_symbol}")
    
    # Tính toán hiệu suất
    performance = utils.calculate_performance(history)
    
    # Hiển thị thông tin cơ bản
    col1, col2, col3 = st.columns(3)
    
    with col1:
        latest_price = history['close'].iloc[-1] if not history.empty else 0
        previous_price = history['close'].iloc[-2] if len(history) > 1 else 0
        price_change = latest_price - previous_price
        price_change_pct = (price_change / previous_price * 100) if previous_price != 0 else 0
        
        st.metric(
            label="Giá hiện tại", 
            value=f"{latest_price:,.0f} VND",
            delta=f"{price_change:,.0f} ({price_change_pct:.2f}%)"
        )
        
        st.metric(
            label="Khối lượng giao dịch gần nhất",
            value=f"{history['volume'].iloc[-1]:,.0f}" if not history.empty else "N/A"
        )
    
    with col2:
        highest_price = history['high'].max() if not history.empty else 0
        lowest_price = history['low'].min() if not history.empty else 0
        
        st.metric(
            label="Giá cao nhất (trong khoảng thời gian)",
            value=f"{highest_price:,.0f} VND"
        )
        
        st.metric(
            label="Giá thấp nhất (trong khoảng thời gian)",
            value=f"{lowest_price:,.0f} VND"
        )
    
    with col3:
        st.metric(
            label="Hiệu suất chiến lược",
            value=f"{performance['strategy_return']:.2f}%",
            delta=f"{performance['outperformance']:.2f}% so với thị trường"
        )
        
        st.metric(
            label="Tỷ lệ thắng",
            value=f"{performance['win_rate']:.2f}%"
        )
    
    # Thống kê tín hiệu mua/bán
    st.subheader("Thống kê tín hiệu giao dịch")
    
    if 'Strong_Buy' in history.columns and 'Strong_Sell' in history.columns:
        buy_signals = history[history['Strong_Buy'] == True]
        sell_signals = history[history['Strong_Sell'] == True]
        
        signal_col1, signal_col2, signal_col3 = st.columns(3)
        
        with signal_col1:
            st.info(f"Tổng số tín hiệu MUA: {len(buy_signals)}")
        
        with signal_col2:
            st.error(f"Tổng số tín hiệu BÁN: {len(sell_signals)}")
        
        with signal_col3:
            last_signal = "MUA" if history['Strong_Buy'].iloc[-1] else ("BÁN" if history['Strong_Sell'].iloc[-1] else "KHÔNG CÓ")
            signal_color = "green" if last_signal == "MUA" else ("red" if last_signal == "BÁN" else "gray")
            st.markdown(f"Tín hiệu mới nhất: <span style='color:{signal_color};font-weight:bold;'>{last_signal}</span>", unsafe_allow_html=True)
    
    # Hiển thị biểu đồ tổng quan
    st.subheader("Biểu đồ giá")
    
    fig = go.Figure(data=[go.Candlestick(
        x=history['time'],
        open=history['open'],
        high=history['high'],
        low=history['low'],
        close=history['close'],
        name='Giá'
    )])
    
    fig.update_layout(
        title=f'Biểu đồ nến {stock_symbol} từ {start_date} đến {end_date}',
        xaxis_title='Ngày',
        yaxis_title='Giá (VND)',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_charts(history, stock_symbol):
    """
    Hiển thị biểu đồ giá và các chỉ báo kỹ thuật
    
    Args:
        history (DataFrame): DataFrame chứa dữ liệu lịch sử và các chỉ báo
        stock_symbol (str): Mã chứng khoán
    """
    st.header(f"Biểu đồ giá và chỉ báo kỹ thuật cho {stock_symbol}")
    
    # Tạo biểu đồ tổng hợp với nhiều subplot
    fig = make_subplots(
        rows=4, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            "Giá & EMA", 
            "Khối lượng giao dịch",
            "RSI",
            "MACD"
        ),
        row_heights=[0.5, 0.15, 0.15, 0.2]
    )
    
    # Thêm dữ liệu giá và đường EMA
    fig.add_trace(
        go.Candlestick(
            x=history['time'],
            open=history['open'],
            high=history['high'],
            low=history['low'],
            close=history['close'],
            name='Giá'
        ),
        row=1, col=1
    )
    
    # Thêm Bollinger Bands
    fig.add_trace(
        go.Scatter(
            x=history['time'],
            y=history['BB_Upper'],
            line=dict(color='rgba(250, 128, 114, 0.7)'),
            name='BB Upper'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=history['time'],
            y=history['BB_Middle'],
            line=dict(color='rgba(0, 128, 128, 0.7)'),
            name='BB Middle'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=history['time'],
            y=history['BB_Lower'],
            line=dict(color='rgba(144, 238, 144, 0.7)'),
            name='BB Lower'
        ),
        row=1, col=1
    )
    
    # Thêm các đường EMA
    fig.add_trace(
        go.Scatter(
            x=history['time'],
            y=history['EMA20'],
            line=dict(color='orange', width=1),
            name='EMA20'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=history['time'],
            y=history['EMA50'],
            line=dict(color='purple', width=1),
            name='EMA50'
        ),
        row=1, col=1
    )
    
    # Thêm chỉ báo VWAP
    fig.add_trace(
        go.Scatter(
            x=history['time'],
            y=history['VWAP'],
            line=dict(color='blue', width=1, dash='dot'),
            name='VWAP'
        ),
        row=1, col=1
    )
    
    # Thêm khối lượng giao dịch
    fig.add_trace(
        go.Bar(
            x=history['time'],
            y=history['volume'],
            name='Khối lượng',
            marker=dict(color='rgba(0, 0, 128, 0.5)')
        ),
        row=2, col=1
    )
    
    # Thêm RSI
    fig.add_trace(
        go.Scatter(
            x=history['time'],
            y=history['RSI'],
            line=dict(color='blue', width=1),
            name='RSI'
        ),
        row=3, col=1
    )
    
    # Thêm đường tham chiếu RSI 70 và 30
    fig.add_trace(
        go.Scatter(
            x=history['time'],
            y=[70] * len(history),
            line=dict(color='red', width=0.5, dash='dash'),
            name='RSI 70'
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=history['time'],
            y=[30] * len(history),
            line=dict(color='green', width=0.5, dash='dash'),
            name='RSI 30'
        ),
        row=3, col=1
    )
    
    # Thêm MACD và Signal
    fig.add_trace(
        go.Scatter(
            x=history['time'],
            y=history['MACD'],
            line=dict(color='blue', width=1),
            name='MACD'
        ),
        row=4, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=history['time'],
            y=history['Signal'],
            line=dict(color='red', width=1),
            name='Signal'
        ),
        row=4, col=1
    )
    
    # Thêm MACD Histogram
    colors = ['green' if val >= 0 else 'red' for val in history['MACD_Histogram']]
    
    fig.add_trace(
        go.Bar(
            x=history['time'],
            y=history['MACD_Histogram'],
            name='MACD Histogram',
            marker=dict(color=colors)
        ),
        row=4, col=1
    )
    
    # Cập nhật layout
    fig.update_layout(
        title=f'Phân tích kỹ thuật cho {stock_symbol}',
        xaxis_title='Ngày',
        height=1000,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    fig.update_yaxes(title_text="Giá (VND)", row=1, col=1)
    fig.update_yaxes(title_text="Khối lượng", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_yaxes(title_text="MACD", row=4, col=1)
    
    st.plotly_chart(fig, use_container_width=True)


def display_signals(history, stock_symbol):
    """
    Hiển thị tín hiệu giao dịch
    
    Args:
        history (DataFrame): DataFrame chứa dữ liệu lịch sử và tín hiệu
        stock_symbol (str): Mã chứng khoán
    """
    st.header(f"Tín hiệu giao dịch cho {stock_symbol}")
    
    # Tạo biểu đồ giá với tín hiệu
    fig = go.Figure()
    
    # Thêm candlestick cho giá
    fig.add_trace(
        go.Candlestick(
            x=history['time'],
            open=history['open'],
            high=history['high'],
            low=history['low'],
            close=history['close'],
            name='Giá'
        )
    )
    
    # Thêm EMA
    fig.add_trace(
        go.Scatter(
            x=history['time'],
            y=history['EMA20'],
            line=dict(color='orange', width=1),
            name='EMA20'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=history['time'],
            y=history['EMA50'],
            line=dict(color='purple', width=1),
            name='EMA50'
        )
    )
    
    # Thêm tín hiệu mua
    buy_signals = history[history['Strong_Buy'] == True]
    if not buy_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_signals['time'],
                y=buy_signals['close'] * 0.98,  # Đặt điểm mua hơi thấp hơn giá đóng cửa để dễ nhìn
                mode='markers',
                marker=dict(
                    symbol='triangle-up',
                    size=15,
                    color='green',
                    line=dict(width=2, color='darkgreen')
                ),
                name='Tín hiệu MUA'
            )
        )
    
    # Thêm tín hiệu bán
    sell_signals = history[history['Strong_Sell'] == True]
    if not sell_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_signals['time'],
                y=sell_signals['close'] * 1.02,  # Đặt điểm bán hơi cao hơn giá đóng cửa để dễ nhìn
                mode='markers',
                marker=dict(
                    symbol='triangle-down',
                    size=15,
                    color='red',
                    line=dict(width=2, color='darkred')
                ),
                name='Tín hiệu BÁN'
            )
        )
    
    # Cập nhật layout
    fig.update_layout(
        title=f'Tín hiệu giao dịch cho {stock_symbol}',
        xaxis_title='Ngày',
        yaxis_title='Giá (VND)',
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Thêm biểu đồ timeline các tín hiệu mua bán theo ngày
    st.subheader("Biểu đồ timeline tín hiệu mua bán theo ngày")
    
    # Tạo DataFrame cho timeline
    timeline_df = pd.DataFrame(columns=['Ngày', 'Loại tín hiệu', 'Giá'])
    
    # Thêm tín hiệu mua
    if not buy_signals.empty:
        buy_df = pd.DataFrame({
            'Ngày': buy_signals['time'],
            'Loại tín hiệu': 'MUA',
            'Giá': buy_signals['close'],
            'RSI': buy_signals['RSI'].round(2),
            'MACD': buy_signals['MACD'].round(2)
        })
        timeline_df = pd.concat([timeline_df, buy_df])
    
    # Thêm tín hiệu bán
    if not sell_signals.empty:
        sell_df = pd.DataFrame({
            'Ngày': sell_signals['time'],
            'Loại tín hiệu': 'BÁN',
            'Giá': sell_signals['close'],
            'RSI': sell_signals['RSI'].round(2),
            'MACD': sell_signals['MACD'].round(2)
        })
        timeline_df = pd.concat([timeline_df, sell_df])
    
    # Sắp xếp theo ngày
    if not timeline_df.empty:
        timeline_df = timeline_df.sort_values('Ngày')
        timeline_df['Text'] = timeline_df.apply(
            lambda row: f"{row['Loại tín hiệu']} - Giá: {row['Giá']:,.0f} - RSI: {row['RSI']} - MACD: {row['MACD']}", 
            axis=1
        )
        
        # Tạo biểu đồ timeline
        timeline_fig = go.Figure()
        
        # Tạo màu cho từng loại tín hiệu
        colors = {'MUA': 'green', 'BÁN': 'red'}
        
        for signal_type in timeline_df['Loại tín hiệu'].unique():
            signal_data = timeline_df[timeline_df['Loại tín hiệu'] == signal_type]
            
            timeline_fig.add_trace(go.Scatter(
                x=signal_data['Ngày'],
                y=[0] * len(signal_data),
                mode='markers+text',
                marker=dict(
                    symbol='circle',
                    size=15,
                    color=colors[signal_type],
                    line=dict(width=2, color='darkgray')
                ),
                text=signal_data['Loại tín hiệu'],
                textposition='top center',
                name=signal_type,
                hovertext=signal_data['Text'],
                hoverinfo='text'
            ))
        
        # Cập nhật layout cho timeline
        timeline_fig.update_layout(
            title="Timeline tín hiệu mua bán theo ngày",
            showlegend=True,
            height=250,
            yaxis=dict(
                showticklabels=False,
                zeroline=True
            ),
            xaxis=dict(
                title="Ngày"
            ),
            hovermode="closest"
        )
        
        st.plotly_chart(timeline_fig, use_container_width=True)
    else:
        st.info("Không có tín hiệu mua bán trong khoảng thời gian này.")
    
    # Hiển thị bảng tín hiệu
    st.subheader("Chi tiết tín hiệu giao dịch")
    
    signals_tab1, signals_tab2 = st.tabs(["Tín hiệu MUA", "Tín hiệu BÁN"])
    
    with signals_tab1:
        if not buy_signals.empty:
            buy_signals_display = buy_signals[['time', 'close', 'RSI', 'MACD', 'Signal']].rename(
                columns={
                    'time': 'Ngày',
                    'close': 'Giá đóng cửa',
                    'RSI': 'RSI',
                    'MACD': 'MACD',
                    'Signal': 'Signal'
                }
            )
            st.dataframe(buy_signals_display, use_container_width=True)
        else:
            st.info("Không có tín hiệu MUA trong khoảng thời gian này.")
    
    with signals_tab2:
        if not sell_signals.empty:
            sell_signals_display = sell_signals[['time', 'close', 'RSI', 'MACD', 'Signal']].rename(
                columns={
                    'time': 'Ngày',
                    'close': 'Giá đóng cửa',
                    'RSI': 'RSI',
                    'MACD': 'MACD',
                    'Signal': 'Signal'
                }
            )
            st.dataframe(sell_signals_display, use_container_width=True)
        else:
            st.info("Không có tín hiệu BÁN trong khoảng thời gian này.")


def display_forecast(forecast, stock_symbol, history):
    """
    Hiển thị dự đoán giá từ Prophet
    
    Args:
        forecast (DataFrame): DataFrame chứa dự đoán từ Prophet
        stock_symbol (str): Mã chứng khoán
        history (DataFrame): DataFrame chứa dữ liệu lịch sử
    """
    st.header(f"Dự đoán giá cho {stock_symbol}")
    
    # Tạo biểu đồ dự đoán
    fig = go.Figure()
    
    # Thêm dữ liệu lịch sử
    history_df = history.copy()
    history_df['time'] = pd.to_datetime(history_df['time'])
    
    fig.add_trace(
        go.Scatter(
            x=history_df['time'],
            y=history_df['close'],
            mode='lines',
            name='Dữ liệu thực tế',
            line=dict(color='blue')
        )
    )
    
    # Thêm dự đoán
    forecast_df = forecast.copy()
    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
    
    fig.add_trace(
        go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat'],
            mode='lines',
            name='Dự đoán',
            line=dict(color='red')
        )
    )
    
    # Thêm khoảng tin cậy
    fig.add_trace(
        go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat_upper'],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat_lower'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.2)',
            name='Khoảng tin cậy 95%'
        )
    )
    
    # Cập nhật layout
    fig.update_layout(
        title=f'Dự đoán giá cho {stock_symbol}',
        xaxis_title='Ngày',
        yaxis_title='Giá (VND)',
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Hiển thị dự đoán trong 7 ngày tới
    st.subheader("Dự đoán giá trong 7 ngày tới")
    
    # Lọc dữ liệu dự đoán trong tương lai
    latest_date = history_df['time'].max()
    future_forecast = forecast_df[forecast_df['ds'] > latest_date].head(7)
    
    if not future_forecast.empty:
        # Hiển thị bảng dự đoán
        forecast_table = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        forecast_table.columns = ['Ngày', 'Giá dự đoán', 'Giá thấp nhất', 'Giá cao nhất']
        
        # Format giá trị
        forecast_table['Giá dự đoán'] = forecast_table['Giá dự đoán'].map('{:,.0f}'.format)
        forecast_table['Giá thấp nhất'] = forecast_table['Giá thấp nhất'].map('{:,.0f}'.format)
        forecast_table['Giá cao nhất'] = forecast_table['Giá cao nhất'].map('{:,.0f}'.format)
        
        st.dataframe(forecast_table, use_container_width=True)
    else:
        st.info("Không có dữ liệu dự đoán cho 7 ngày tới.")
    
    # Hiển thị các thành phần mùa vụ
    st.subheader("Thành phần mùa vụ trong dự đoán")
    
    # Tạo các tab cho các thành phần khác nhau
    components_tab1, components_tab2, components_tab3 = st.tabs([
        "Xu hướng", "Mùa vụ hàng tuần", "Mùa vụ hàng năm"
    ])
    
    with components_tab1:
        fig_trend = go.Figure()
        
        fig_trend.add_trace(
            go.Scatter(
                x=forecast_df['ds'],
                y=forecast_df['trend'],
                mode='lines',
                name='Xu hướng',
                line=dict(color='blue')
            )
        )
        
        fig_trend.update_layout(
            title='Xu hướng dài hạn',
            xaxis_title='Ngày',
            yaxis_title='Giá (VND)',
            height=400
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with components_tab2:
        if 'weekly' in forecast_df.columns:
            days = ['Thứ 2', 'Thứ 3', 'Thứ 4', 'Thứ 5', 'Thứ 6', 'Thứ 7', 'Chủ nhật']
            weekly_effect = forecast_df.groupby(forecast_df['ds'].dt.weekday)['weekly'].mean().reset_index()
            weekly_effect.columns = ['Ngày', 'Ảnh hưởng']
            weekly_effect['Ngày'] = weekly_effect['Ngày'].apply(lambda x: days[x])
            
            fig_weekly = go.Figure(
                go.Bar(
                    x=weekly_effect['Ngày'],
                    y=weekly_effect['Ảnh hưởng'],
                    marker_color='lightblue'
                )
            )
            
            fig_weekly.update_layout(
                title='Ảnh hưởng theo ngày trong tuần',
                xaxis_title='Ngày trong tuần',
                yaxis_title='Ảnh hưởng (VND)',
                height=400
            )
            
            st.plotly_chart(fig_weekly, use_container_width=True)
        else:
            st.info("Không có dữ liệu về ảnh hưởng hàng tuần.")
    
    with components_tab3:
        if 'yearly' in forecast_df.columns:
            months = ['Tháng 1', 'Tháng 2', 'Tháng 3', 'Tháng 4', 'Tháng 5', 'Tháng 6', 'Tháng 7', 'Tháng 8', 'Tháng 9', 'Tháng 10', 'Tháng 11', 'Tháng 12']
            yearly_effect = forecast_df.groupby(forecast_df['ds'].dt.month)['yearly'].mean().reset_index()
            yearly_effect.columns = ['Tháng', 'Ảnh hưởng']
            yearly_effect['Tháng'] = yearly_effect['Tháng'].apply(lambda x: months[x-1])
            
            fig_yearly = go.Figure(
                go.Bar(
                    x=yearly_effect['Tháng'],
                    y=yearly_effect['Ảnh hưởng'],
                    marker_color='lightgreen'
                )
            )
            
            fig_yearly.update_layout(
                title='Ảnh hưởng theo tháng trong năm',
                xaxis_title='Tháng',
                yaxis_title='Ảnh hưởng (VND)',
                height=400
            )
            
            st.plotly_chart(fig_yearly, use_container_width=True)
        else:
            st.info("Không có dữ liệu về ảnh hưởng hàng năm.")


def display_data_table(history):
    """
    Hiển thị bảng dữ liệu chi tiết
    
    Args:
        history (DataFrame): DataFrame chứa dữ liệu lịch sử và các chỉ báo
    """
    st.header("Dữ liệu chi tiết")
    
    # Tạo bản sao của DataFrame để định dạng lại
    display_df = history.copy()
    
    # Chọn các cột để hiển thị
    display_columns = [
        'time', 'open', 'high', 'low', 'close', 'volume', 
        'RSI', 'MACD', 'Signal', 'MACD_Histogram', 
        'EMA20', 'EMA50', 'BB_Upper', 'BB_Middle', 'BB_Lower'
    ]
    
    # Kiểm tra xem các cột có tồn tại không
    available_columns = [col for col in display_columns if col in display_df.columns]
    
    if available_columns:
        # Định dạng lại tên cột để hiển thị
        column_rename = {
            'time': 'Ngày',
            'open': 'Mở cửa',
            'high': 'Cao nhất',
            'low': 'Thấp nhất',
            'close': 'Đóng cửa',
            'volume': 'Khối lượng',
            'RSI': 'RSI',
            'MACD': 'MACD',
            'Signal': 'Signal',
            'MACD_Histogram': 'Histogram',
            'EMA20': 'EMA20',
            'EMA50': 'EMA50',
            'BB_Upper': 'BB Trên',
            'BB_Middle': 'BB Giữa',
            'BB_Lower': 'BB Dưới'
        }
        
        # Chỉ đổi tên các cột có sẵn
        rename_dict = {col: column_rename[col] for col in available_columns if col in column_rename}
        display_df = display_df[available_columns].rename(columns=rename_dict)
        
        # Hiển thị bảng dữ liệu với khả năng phân trang
        st.dataframe(display_df, use_container_width=True)
        
        # Tùy chọn tải xuống dữ liệu
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Tải xuống dữ liệu (CSV)",
            data=csv,
            file_name="stock_data.csv",
            mime="text/csv",
        )
    else:
        st.warning("Không có dữ liệu để hiển thị.")
