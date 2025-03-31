import streamlit as st
from vnstock import Vnstock
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

def calculate_rsi(data, periods=14):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(data, span=20):
    return data['close'].ewm(span=span, adjust=False).mean()

def calculate_macd(data):
    exp1 = data['close'].ewm(span=12, adjust=False).mean()
    exp2 = data['close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def calculate_bollinger_bands(data, window=20, num_std=2):
    sma = data['close'].rolling(window=window).mean()
    std = data['close'].rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, sma, lower_band

def calculate_volume_breakout(data, window=20):
    volume_sma = data['volume'].rolling(window=window).mean()
    return data['volume'] > volume_sma * 1.5

def calculate_vwap(data):
    """Calculate VWAP (Volume Weighted Average Price)"""
    return (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()

def calculate_ema9(data):
    """Calculate EMA 9"""
    return data['close'].ewm(span=9, adjust=False).mean()

def detect_candlestick_patterns(data):
    """Detect bullish candlestick patterns"""
    patterns = pd.Series(index=data.index, data=False)
    
    # Bullish Engulfing
    patterns |= (data['open'].shift(1) > data['close'].shift(1)) & \
               (data['close'] > data['open']) & \
               (data['open'] <= data['close'].shift(1)) & \
               (data['close'] >= data['open'].shift(1))
    
    # Hammer
    body_size = abs(data['close'] - data['open'])
    lower_shadow = data[['open', 'close']].min(axis=1) - data['low']
    upper_shadow = data['high'] - data[['open', 'close']].max(axis=1)
    
    patterns |= (lower_shadow > 2 * body_size) & \
               (upper_shadow <= 0.1 * lower_shadow)
    
    # Doji
    patterns |= abs(data['close'] - data['open']) <= (data['high'] - data['low']) * 0.1
    
    return patterns

def detect_bearish_patterns(data):
    """Phát hiện các mô hình nến đảo chiều giảm"""
    patterns = pd.Series(index=data.index, data=False)
    
    # Bearish Engulfing
    patterns |= (data['open'].shift(1) < data['close'].shift(1)) & \
               (data['close'] < data['open']) & \
               (data['open'] >= data['close'].shift(1)) & \
               (data['close'] <= data['open'].shift(1))
    
    # Shooting Star
    body_size = abs(data['close'] - data['open'])
    upper_shadow = data['high'] - data[['open', 'close']].max(axis=1)
    lower_shadow = data[['open', 'close']].min(axis=1) - data['low']
    
    patterns |= (upper_shadow > 2 * body_size) & \
               (lower_shadow <= 0.1 * upper_shadow) & \
               (data['close'] < data['close'].shift(1))
    
    # Evening Star
    patterns |= (data['close'].shift(2) > data['open'].shift(2)) & \
               (abs(data['close'].shift(1) - data['open'].shift(1)) < 0.1 * data['close'].shift(2)) & \
               (data['close'] < data['open']) & \
               (data['close'] < data['close'].shift(2))
    
    return patterns

def detect_resistance_levels(data, lookback=20, threshold=0.02):
    """Phát hiện các vùng kháng cự"""
    resistance_levels = pd.Series(index=data.index, data=False)
    
    for i in range(lookback, len(data)):
        local_high = data['high'].iloc[i-lookback:i].max()
        price_range = local_high * threshold
        
        # Kiểm tra giá hiện tại có đang tiếp cận vùng kháng cự
        resistance_levels.iloc[i] = (data['high'].iloc[i] >= local_high - price_range) & \
                                  (data['high'].iloc[i] <= local_high + price_range)
    
    return resistance_levels

def calculate_t0_indicators(data):
    """Tính toán các chỉ báo cho giao dịch T0, sắp xếp theo độ tin cậy giảm dần"""
    
    indicators = pd.DataFrame(index=data.index)
    
    # 1. Momentum và Khối lượng (Độ tin cậy cao nhất)
    indicators['volume_ma5'] = data['volume'].rolling(window=5).mean()
    indicators['volume_ratio'] = data['volume'] / indicators['volume_ma5']
    indicators['price_change'] = data['close'].pct_change()
    
    # 2. Stochastic RSI (Độ tin cậy cao)
    # Tính toán Fast Stochastic của RSI
    rsi_data = pd.Series(calculate_rsi(data))
    stoch_rsi = (rsi_data - rsi_data.rolling(14).min()) / (rsi_data.rolling(14).max() - rsi_data.rolling(14).min())
    indicators['stoch_rsi'] = stoch_rsi
    
    # 3. MFI - Money Flow Index (Độ tin cậy khá)
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    money_flow = typical_price * data['volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
    mfi = 100 - (100 / (1 + positive_flow / negative_flow))
    indicators['mfi'] = mfi
    
    # 4. Support/Resistance Breakout (Độ tin cậy trung bình)
    indicators['prev_high'] = data['high'].rolling(5).max().shift(1)
    indicators['prev_low'] = data['low'].rolling(5).min().shift(1)
    
    return indicators

def generate_t0_signals(data, indicators):
    """Tạo tín hiệu giao dịch T0 dựa trên các chỉ báo"""
    
    signals = pd.DataFrame(index=data.index)
    signals['T0_Signal'] = 'Hold'
    signals['T0_Strength'] = 0
    signals['T0_Explanation'] = ''
    
    # Tiêu chí mua:
    buy_conditions = []
    
    # 1. Volume Breakout (trọng số: 3)
    volume_breakout = (indicators['volume_ratio'] > 2.0)
    if volume_breakout.any():
        buy_conditions.append(('Khối lượng đột biến (>200% MA5)', 3))
    
    # 2. Stochastic RSI (trọng số: 2)
    stoch_rsi_oversold = (indicators['stoch_rsi'] < 0.2)
    if stoch_rsi_oversold.any():
        buy_conditions.append(('Stochastic RSI quá bán', 2))
    
    # 3. MFI (trọng số: 2)
    mfi_oversold = (indicators['mfi'] < 20)
    if mfi_oversold.any():
        buy_conditions.append(('MFI quá bán (<20)', 2))
    
    # 4. Support Breakout (trọng số: 1)
    support_break = (data['close'] > indicators['prev_high'])
    if support_break.any():
        buy_conditions.append(('Breakout kháng cự', 1))

    # Tổng hợp tín hiệu
    for idx in signals.index:
        current_conditions = []
        total_strength = 0
        
        if volume_breakout[idx]:
            current_conditions.append('Khối lượng đột biến')
            total_strength += 3
        if stoch_rsi_oversold[idx]:
            current_conditions.append('Stoch RSI quá bán')
            total_strength += 2
        if mfi_oversold[idx]:
            current_conditions.append('MFI quá bán')
            total_strength += 2
        if support_break[idx]:
            current_conditions.append('Breakout kháng cự')
            total_strength += 1
            
        if total_strength >= 5:  # Có ít nhất 3 tiêu chí
            signals.loc[idx, 'T0_Signal'] = 'Strong Buy'
            signals.loc[idx, 'T0_Strength'] = total_strength
            signals.loc[idx, 'T0_Explanation'] = '<br>'.join([
                '<b>Tín hiệu T0 mạnh</b>',
                f'Độ mạnh: {total_strength}/8',
                f'Lý do: {", ".join(current_conditions)}'
            ])
    
    return signals

# Update the generate_signals function to include Final_Signal
def generate_signals(data):
    signals = pd.DataFrame(index=data.index)
    
    # Calculate new indicators
    data['EMA9'] = calculate_ema9(data)
    data['VWAP'] = calculate_vwap(data)
    data['Bullish_Patterns'] = detect_candlestick_patterns(data)
    
    # Define conditions
    ema_crossover = (data['EMA9'] > data['EMA20']) & (data['EMA9'].shift(1) <= data['EMA20'].shift(1))
    above_vwap = data['close'] > data['VWAP']
    macd_crossover = (data['MACD'] > data['Signal']) & (data['MACD'].shift(1) <= data['Signal'].shift(1))
    rsi_not_overbought = data['RSI'] < 70
    volume_breakout = data['Volume_Breakout']
    bullish_patterns = data['Bullish_Patterns']
    
    # Count buy conditions
    buy_conditions = pd.DataFrame(index=data.index)
    buy_conditions['EMA_Cross'] = ema_crossover
    buy_conditions['Above_VWAP'] = above_vwap
    buy_conditions['MACD_Cross'] = macd_crossover
    buy_conditions['RSI_Good'] = rsi_not_overbought
    buy_conditions['Volume_Break'] = volume_breakout
    buy_conditions['Bullish_Pattern'] = bullish_patterns
    
    # Generate final signals
    conditions_met = buy_conditions.sum(axis=1)
    
    signals['Buy_Signal'] = conditions_met >= 3  # At least 3 conditions met
    signals['Signal_Strength'] = conditions_met
    
    # Add Final_Signal column
    signals['Final_Signal'] = 'Hold'
    signals.loc[signals['Buy_Signal'] & (signals['Signal_Strength'] >= 5), 'Final_Signal'] = 'Strong Buy'
    signals.loc[signals['Buy_Signal'] & (signals['Signal_Strength'] >= 3), 'Final_Signal'] = 'Buy'
    
    # Generate explanation text
    def generate_explanation(row):
        reasons = []
        if row['EMA_Cross']: reasons.append("EMA9 cắt EMA20 từ dưới lên")
        if row['Above_VWAP']: reasons.append("Giá trên VWAP")
        if row['MACD_Cross']: reasons.append("MACD cắt Signal từ dưới lên")
        if row['RSI_Good']: reasons.append("RSI < 70")
        if row['Volume_Break']: reasons.append("Khối lượng đột biến")
        if row['Bullish_Pattern']: reasons.append("Xuất hiện mô hình nến tăng")
        return "<br>".join(reasons)
    
    signals['Signal_Explanation'] = buy_conditions.apply(generate_explanation, axis=1)
    
    return signals

def generate_general_buy_signals(data):
    """
    Generate general buy signals for swing/position trading
    """
    signals = pd.DataFrame(index=data.index)
    
    # Calculate indicators
    data['SMA20'] = data['close'].rolling(window=20).mean()
    data['MA20_Volume'] = data['volume'].rolling(window=20).mean()
    
    # Define conditions
    conditions = {
        'trend': {
            'ema9_above_ema20': (data['EMA9'] > data['EMA20']),
            'price_above_sma20': (data['close'] > data['SMA20']),
            'ema_crossover': (data['EMA9'] > data['EMA20']) & (data['EMA9'].shift(1) <= data['EMA20'].shift(1))
        },
        'momentum': {
            'macd_crossover': (data['MACD'] > data['Signal']) & (data['MACD'].shift(1) <= data['Signal'].shift(1)),
            'rsi_good': (data['RSI'] < 70) & (data['RSI'] > 30)
        },
        'volume': {
            'volume_breakout': data['volume'] > data['MA20_Volume'] * 1.5
        },
        'patterns': {
            'bullish_patterns': data['Bullish_Patterns']
        }
    }
    
    # Calculate signal strength
    strength = 0
    reasons = []
    
    for category, category_conditions in conditions.items():
        for name, condition in category_conditions.items():
            if condition.any():
                strength += 1
                reasons.append(name)
    
    # Generate signals
    signals['Signal'] = 'Hold'
    signals.loc[strength >= 5, 'Signal'] = 'Strong Buy'
    signals.loc[(strength >= 3) & (strength < 5), 'Signal'] = 'Buy'
    signals['Strength'] = strength
    signals['Reasons'] = pd.Series([', '.join(reasons) if reasons else 'No signals' for _ in range(len(signals))])
    
    return signals

def generate_t0_buy_signals(data):
    """
    Generate T0 (intraday) buy signals
    """
    signals = pd.DataFrame(index=data.index)
    
    # Calculate T0 specific indicators
    data['MA5_Volume'] = data['volume'].rolling(window=5).mean()
    data['Price_Change'] = data['close'].pct_change()
    data['High_5D'] = data['high'].rolling(window=5).max()
    
    # Define T0 conditions
    conditions = {
        'price': {
            'price_up': (data['close'] > data['open']),
            'price_momentum': (data['Price_Change'] > 0)
        },
        'volume': {
            'volume_surge': (data['volume'] > data['MA5_Volume'] * 2)
        },
        'technical': {
            'stoch_rsi_good': (data['stoch_rsi'] < 0.2),
            'mfi_good': (data['mfi'] < 20),
            'breakout': (data['close'] > data['High_5D'].shift(1))
        }
    }
    
    # Calculate T0 signal strength
    strength = 0
    reasons = []
    
    for category, category_conditions in conditions.items():
        for name, condition in category_conditions.items():
            if condition.any():
                strength += 1
                reasons.append(name)
    
    # Generate T0 signals
    signals['T0_Signal'] = 'Hold'
    signals.loc[strength >= 4, 'T0_Signal'] = 'Strong Buy T0'
    signals.loc[(strength >= 2) & (strength < 4), 'T0_Signal'] = 'Buy T0'
    signals['T0_Strength'] = strength
    signals['T0_Reasons'] = pd.Series([', '.join(reasons) if reasons else 'No signals' for _ in range(len(signals))])
    
    return signals

def generate_general_sell_signals(data):
    """Generate general sell signals with improved conditions"""
    signals = pd.DataFrame(index=data.index)
    
    # Thêm các chỉ báo mới
    data['Bearish_Patterns'] = detect_bearish_patterns(data)
    data['Resistance_Level'] = detect_resistance_levels(data)
    data['LongTerm_EMA200'] = data['close'].ewm(span=200, adjust=False).mean()
    data['MoneyFlow'] = (data['high'] + data['low'] + data['close'])/3 * data['volume']
    
    conditions = {
        'trend': {
            'ema9_below_ema20': (data['EMA9'] < data['EMA20']) & 
                               (data['EMA9'].shift(1) >= data['EMA20'].shift(1)), # 2 points
            'price_below_ema20': (data['close'] < data['EMA20']), # 1 point
            'below_longterm_trend': (data['close'] < data['LongTerm_EMA200']) # 2 points
        },
        'momentum': {
            'macd_crossover_down': (data['MACD'] < data['Signal']) & 
                                 (data['MACD'].shift(1) >= data['Signal'].shift(1)), # 2 points
            'rsi_overbought': (data['RSI'] > 70) & 
                            (data['RSI'] < data['RSI'].shift(1)), # 2 points
        },
        'volume_and_patterns': {
            'bearish_pattern': data['Bearish_Patterns'], # 2 points
            'at_resistance': data['Resistance_Level'], # 1 point
            'high_volume_red': (data['volume'] > data['MA20_Volume'] * 1.5) & 
                             (data['close'] < data['open']) # 2 points
        },
        'moneyflow': {
            'declining_money_flow': (data['MoneyFlow'] < data['MoneyFlow'].shift(1)) & 
                                  (data['MoneyFlow'].shift(1) < data['MoneyFlow'].shift(2)) # 2 points
        }
    }
    
    # Cập nhật điểm cho các điều kiện
    condition_points = {
        'ema9_below_ema20': 2,
        'price_below_ema20': 1,
        'below_longterm_trend': 2,
        'macd_crossover_down': 2,
        'rsi_overbought': 2,
        'bearish_pattern': 2,
        'at_resistance': 1,
        'high_volume_red': 2,
        'declining_money_flow': 2
    }
    
    # Tính toán tín hiệu
    strength = 0
    reasons = []
    
    for category, category_conditions in conditions.items():
        for name, condition in category_conditions.items():
            if condition.any():
                strength += condition_points[name]
                reasons.append(name)
    
    # Cập nhật ngưỡng phát tín hiệu
    signals['Sell_Signal'] = 'Hold'
    signals['Sell_Strength'] = strength
    signals['Sell_Reasons'] = pd.Series([', '.join(reasons) if reasons else 'No signals' 
                                       for _ in range(len(signals))])
    
    # Strong Sell: >= 8 points (≈50% tổng điểm)
    # Sell: >= 6 points (≈35% tổng điểm)
    signals.loc[strength >= 8, 'Sell_Signal'] = 'Strong Sell'
    signals.loc[(strength >= 6) & (strength < 8), 'Sell_Signal'] = 'Sell'
    
    return signals

def generate_t0_sell_signals(data):
    """
    Generate T0 (intraday) sell signals
    """
    signals = pd.DataFrame(index=data.index)
    
    # Define T0 sell conditions with points
    conditions = {
        'price_action': {
            'price_down': (data['close'] < data['open']), # 1 point
            'lower_low': (data['low'] < data['low'].shift(1)), # 1 point
            'below_vwap': (data['close'] < data['VWAP']) # 1 point
        },
        'momentum': {
            'rsi_falling': (data['RSI'] < data['RSI'].shift(1)) & 
                         (data['RSI'].shift(1) < data['RSI'].shift(2)), # 2 points
            'stoch_rsi_high': (data['stoch_rsi'] > 0.8) & 
                            (data['stoch_rsi'] < data['stoch_rsi'].shift(1)) # 2 points
        },
        'volume': {
            'volume_price_divergence': (data['volume'] > data['MA5_Volume'] * 2) & 
                                     (data['close'] < data['open']) # 3 points
        }
    }
    
    # Points for each condition
    condition_points = {
        'price_down': 1,
        'lower_low': 1,
        'below_vwap': 1,
        'rsi_falling': 2,
        'stoch_rsi_high': 2,
        'volume_price_divergence': 3
    }
    
    # Calculate T0 sell signals
    strength = 0
    reasons = []
    
    for category, category_conditions in conditions.items():
        for name, condition in category_conditions.items():
            if condition.any():
                strength += condition_points[name]
                reasons.append(name)
    
    # Generate T0 sell signals based on total points (max 10 points)
    signals['T0_Sell_Signal'] = 'Hold'
    signals['T0_Sell_Strength'] = strength
    signals['T0_Sell_Reasons'] = pd.Series([', '.join(reasons) if reasons else 'No signals' 
                                          for _ in range(len(signals))])
    
    # Strong Sell T0: >= 7 points (70%)
    # Sell T0: >= 5 points (50%)
    signals.loc[strength >= 7, 'T0_Sell_Signal'] = 'Strong Sell T0'
    signals.loc[(strength >= 5) & (strength < 7), 'T0_Sell_Signal'] = 'Sell T0'
    
    return signals

def process_trading_decisions(data):
    """
    Process and combine all trading signals with detailed explanations
    """
    # Generate all signals
    general_buy = generate_general_buy_signals(data)
    t0_buy = generate_t0_buy_signals(data)
    general_sell = generate_general_sell_signals(data)
    t0_sell = generate_t0_sell_signals(data)
    
    # Combine signals
    signals = pd.DataFrame(index=data.index)
    signals['Final_Signal'] = 'Hold'
    signals['Signal_Type'] = ''
    signals['Signal_Strength'] = 0
    signals['Explanation'] = ''
    
    # Priority: Strong Sell > Sell > Strong Buy > Buy > Hold
    for idx in signals.index:
        # Check Strong Sell signals first
        if general_sell.loc[idx, 'Sell_Signal'] == 'Strong Sell':
            signals.loc[idx, 'Final_Signal'] = 'Strong Sell'
            signals.loc[idx, 'Signal_Type'] = 'General'
            signals.loc[idx, 'Signal_Strength'] = general_sell.loc[idx, 'Sell_Strength']
            signals.loc[idx, 'Explanation'] = f"Strong Sell: {general_sell.loc[idx, 'Sell_Reasons']}"
        
        elif t0_sell.loc[idx, 'T0_Sell_Signal'] == 'Strong Sell T0':
            signals.loc[idx, 'Final_Signal'] = 'Strong Sell T0'
            signals.loc[idx, 'Signal_Type'] = 'T0'
            signals.loc[idx, 'Signal_Strength'] = t0_sell.loc[idx, 'T0_Sell_Strength']
            signals.loc[idx, 'Explanation'] = f"Strong Sell T0: {t0_sell.loc[idx, 'T0_Sell_Reasons']}"
        
        elif general_sell.loc[idx, 'Sell_Signal'] == 'Sell':
            signals.loc[idx, 'Final_Signal'] = 'Sell'
            signals.loc[idx, 'Signal_Type'] = 'General'
            signals.loc[idx, 'Signal_Strength'] = general_sell.loc[idx, 'Sell_Strength']
            signals.loc[idx, 'Explanation'] = f"Sell: {general_sell.loc[idx, 'Sell_Reasons']}"
        
        elif t0_sell.loc[idx, 'T0_Sell_Signal'] == 'Sell T0':
            signals.loc[idx, 'Final_Signal'] = 'Sell T0'
            signals.loc[idx, 'Signal_Type'] = 'T0'
            signals.loc[idx, 'Signal_Strength'] = t0_sell.loc[idx, 'T0_Sell_Strength']
            signals.loc[idx, 'Explanation'] = f"Sell T0: {t0_sell.loc[idx, 'T0_Sell_Reasons']}"
        
        elif general_buy.loc[idx, 'Signal'] in ['Strong Buy', 'Buy']:
            signals.loc[idx, 'Final_Signal'] = general_buy.loc[idx, 'Signal']
            signals.loc[idx, 'Signal_Type'] = 'General'
            signals.loc[idx, 'Signal_Strength'] = general_buy.loc[idx, 'Strength']
            signals.loc[idx, 'Explanation'] = f"{general_buy.loc[idx, 'Signal']}: {general_buy.loc[idx, 'Reasons']}"
        
        elif t0_buy.loc[idx, 'T0_Signal'] in ['Strong Buy T0', 'Buy T0']:
            signals.loc[idx, 'Final_Signal'] = t0_buy.loc[idx, 'T0_Signal']
            signals.loc[idx, 'Signal_Type'] = 'T0'
            signals.loc[idx, 'Signal_Strength'] = t0_buy.loc[idx, 'T0_Strength']
            signals.loc[idx, 'Explanation'] = f"{t0_buy.loc[idx, 'T0_Signal']}: {t0_buy.loc[idx, 'T0_Reasons']}"
    
    return signals

def calculate_money_flow(data):
    """Calculate Money Flow for technical analysis
    
    Parameters:
    data (pd.DataFrame): DataFrame containing OHLCV data
    
    Returns:
    pd.Series: Money Flow values
    """
    # Calculate typical price
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    
    # Calculate raw money flow
    raw_money_flow = typical_price * data['volume']
    
    # Determine if money flow is positive or negative based on price movement
    positive_flow = raw_money_flow.where(data['close'] > data['close'].shift(1), 0)
    negative_flow = raw_money_flow.where(data['close'] < data['close'].shift(1), 0)
    
    # Calculate 14-period moving averages
    positive_mf = positive_flow.rolling(window=14).sum()
    negative_mf = negative_flow.rolling(window=14).sum()
    
    # Calculate money flow ratio and index
    mf_ratio = positive_mf / negative_mf
    money_flow_index = 100 - (100 / (1 + mf_ratio))
    
    return money_flow_index

# Tiêu đề ứng dụng
st.title("Dữ liệu Lịch sử Chứng khoán Việt Nam")

# Input controls
with st.sidebar:
    st.subheader("Tham số đầu vào")
    
    # Stock symbol input
    stock_symbol = st.text_input("Mã chứng khoán:", value="KHG").upper()
    
    # Date range input
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Ngày bắt đầu", 
                                 value=datetime(2025, 2, 1),
                                 format="YYYY-MM-DD")
    with col2:
        end_date = st.date_input("Ngày kết thúc", 
                               value=datetime(2025, 3, 28),
                               format="YYYY-MM-DD")
    
    # Submit button
    submit = st.button("Hiển thị kết quả", type="primary")

# Main content
if submit:
    st.write(f"Đang phân tích dữ liệu cho mã {stock_symbol} từ {start_date} đến {end_date}")
    
    try:
        # Convert dates to string format
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        # Get stock data
        df = Vnstock().stock(symbol=stock_symbol, source='VCI')
        history = df.quote.history(start=start_str, end=end_str)

        if history is not None and not history.empty:
            # Calculate all technical indicators
            history['RSI'] = calculate_rsi(history)
            history['EMA20'] = calculate_ema(history, 20)
            history['EMA50'] = calculate_ema(history, 50)
            history['MACD'], history['Signal'] = calculate_macd(history)
            history['MACD_Histogram'] = history['MACD'] - history['Signal']
            history['BB_Upper'], history['BB_Middle'], history['BB_Lower'] = calculate_bollinger_bands(history)
            history['Volume_Breakout'] = calculate_volume_breakout(history)
            t0_indicators = calculate_t0_indicators(history)
            
            # Generate trading signals
            signals = generate_signals(history)
            t0_signals = generate_t0_signals(history, t0_indicators)
            history = pd.concat([history, signals, t0_signals], axis=1)
            
            # Calculate additional indicators
            history['EMA200'] = calculate_ema(history, 200)
            history['MoneyFlow'] = calculate_money_flow(history)
            history['Bearish_Patterns'] = detect_bearish_patterns(history)
            history['Resistance_Level'] = detect_resistance_levels(history)
            
            # Hiển thị dữ liệu dưới dạng bảng có thể đóng/mở
            with st.expander("Xem dữ liệu lịch sử chi tiết", expanded=False):
                st.subheader("Dữ liệu lịch sử")
                st.dataframe(history)

            # Vẽ biểu đồ giá đóng cửa
            st.subheader("Biểu đồ giá đóng cửa")
            fig_line = px.line(history, x="time", y="close", 
                         title=f"Giá đóng cửa của {stock_symbol} từ {start_date} đến {end_date}",
                         labels={"time": "Ngày", "close": f"Giá đóng cửa {stock_symbol} (VND)"})
            st.plotly_chart(fig_line)

            # Vẽ biểu đồ nến
            st.subheader("Biểu đồ nến giao dịch")
            fig_candle = go.Figure(data=[go.Candlestick(x=history['time'],
                                                       open=history['open'],
                                                       high=history['high'],
                                                       low=history['low'],
                                                       close=history['close'])])
            
            fig_candle.update_layout(
                title=f'Biểu đồ nến {stock_symbol} từ {start_date} đến {end_date}',
                yaxis_title=f'Giá {stock_symbol} (VND)',
                xaxis_title='Thời gian'
            )

            # Add EMA to candlestick chart
            fig_candle.add_trace(go.Scatter(x=history['time'], y=history['EMA20'],
                                          name='EMA20', line=dict(color='orange')))
            fig_candle.add_trace(go.Scatter(x=history['time'], y=history['EMA50'],
                                          name='EMA50', line=dict(color='blue')))
            
            # Add Bollinger Bands to candlestick chart
            fig_candle.add_trace(go.Scatter(x=history['time'], y=history['BB_Upper'],
                                          name='BB Upper', line=dict(color='gray', dash='dash')))
            fig_candle.add_trace(go.Scatter(x=history['time'], y=history['BB_Lower'],
                                          name='BB Lower', line=dict(color='gray', dash='dash')))
            fig_candle.add_trace(go.Scatter(x=history['time'], y=history['EMA9'],
                               name='EMA9', line=dict(color='purple')))
            fig_candle.add_trace(go.Scatter(x=history['time'], y=history['VWAP'],
                               name='VWAP', line=dict(color='blue', dash='dot')))
            st.plotly_chart(fig_candle)

            # Create tabs for technical indicators
            st.subheader("Chỉ báo kỹ thuật")
            tab_rsi, tab_macd, tab_signals = st.tabs(["RSI", "MACD", "Tín hiệu tổng hợp"])
            
            # RSI Tab
            with tab_rsi:
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=history['time'], y=history['RSI'],
                                           name='RSI', line=dict(color='purple')))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                fig_rsi.update_layout(
                    title=f'Chỉ báo RSI (14) - {stock_symbol}',
                    yaxis_title='RSI',
                    height=400  # Adjust height for better visualization
                )
                st.plotly_chart(fig_rsi, use_container_width=True)
                
            # MACD Tab
            with tab_macd:
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=history['time'], y=history['MACD'],
                                            name='MACD', line=dict(color='blue')))
                fig_macd.add_trace(go.Scatter(x=history['time'], y=history['Signal'],
                                            name='Signal', line=dict(color='orange')))
                fig_macd.add_trace(go.Bar(x=history['time'], y=history['MACD_Histogram'],
                                        name='Histogram'))
                fig_macd.update_layout(
                    title=f'Chỉ báo MACD (12,26,9) - {stock_symbol}',
                    yaxis_title='MACD',
                    height=400  # Adjust height for better visualization
                )
                st.plotly_chart(fig_macd, use_container_width=True)

            # Update the tab_signals section with proper date formatting
            with tab_signals:
                st.markdown("### Điều kiện tín hiệu giao dịch:")
                
                # Buy conditions
                st.markdown("#### Điều kiện MUA:")
                st.markdown("""
                1. EMA 9 cắt EMA 20 từ dưới lên ➔ Xu hướng tăng
                2. Giá nằm trên VWAP ➔ Lực mua mạnh
                3. MACD Line cắt Signal Line từ dưới lên ➔ Động lượng tăng
                4. RSI < 70 ➔ Chưa quá mua
                5. Breakout với volume cao
                6. Mô hình nến tăng
                """)
                
                # Sell conditions
                st.markdown("#### Điều kiện BÁN:")
                st.markdown("""
                1. EMA 9 cắt EMA 20 từ trên xuống ➔ Xu hướng giảm
                2. Giá nằm dưới VWAP ➔ Lực bán mạnh
                3. MACD Line cắt Signal Line từ trên xuống ➔ Động lượng giảm
                4. RSI > 70 và đang giảm ➔ Quá mua
                5. Volume cao với nến đỏ ➔ Áp lực bán mạnh
                """)
                
                # Display signals
                buy_signals = history[history['Final_Signal'].str.contains('Buy', na=False)]
                sell_signals = history[history['Final_Signal'].str.contains('Sell', na=False)]
                
                if not buy_signals.empty:
                    st.markdown("### Các điểm MUA được phát hiện:")
                    for idx, row in buy_signals.iterrows():
                        signal_date = pd.to_datetime(row['time']).strftime('%Y-%m-%d')
                        st.markdown(f"""
                        **Ngày {signal_date}**
                        - Giá: {row['close']:,.0f} VND
                        - Tín hiệu: {row['Final_Signal']}
                        - Chi tiết: {row['Signal_Explanation']}
                        """)
                
                if not sell_signals.empty:
                    st.markdown("### Các điểm BÁN được phát hiện:")
                    for idx, row in sell_signals.iterrows():
                        signal_date = pd.to_datetime(row['time']).strftime('%Y-%m-%d')
                        st.markdown(f"""
                        **Ngày {signal_date}**
                        - Giá: {row['close']:,.0f} VND
                        - Tín hiệu: {row['Final_Signal']}
                        - Chi tiết: {row['Signal_Explanation']}
                        """)

            # Display trading signals in collapsible section
            with st.expander("Xem chi tiết tín hiệu giao dịch", expanded=False):
                st.subheader("Tín hiệu giao dịch")
                signals_df = history[['time', 'close', 'RSI']].copy()
                if 'Final_Signal' in history.columns:
                    signals_df['Tín hiệu'] = history['Final_Signal']
                else:
                    signals_df['Tín hiệu'] = 'Hold'
                signals_df.columns = ['Ngày', 'Giá đóng cửa', 'RSI', 'Tín hiệu']
                st.dataframe(signals_df)

            # Add trading signals visualization
            st.subheader("Biểu đồ tín hiệu giao dịch")
            fig_signals = go.Figure()

            # Add price line
            fig_signals.add_trace(go.Scatter(
                x=history['time'],
                y=history['close'],
                name='Giá đóng cửa',
                line=dict(color='black', width=1)
            ))

            # Add buy signals with hover text (with safety checks)
            if 'Final_Signal' in history.columns:
                buy_points = history[history['Final_Signal'].isin(['Buy', 'Strong Buy'])]
                if not buy_points.empty:
                    fig_signals.add_trace(go.Scatter(
                        x=buy_points['time'],
                        y=buy_points['close'],
                        mode='markers',
                        name='Tín hiệu Mua',
                        marker=dict(color='green', size=10, symbol='triangle-up'),
                        text=buy_points.get('Signal_Explanation', ''),
                        hovertemplate='<b>Tín hiệu Mua</b><br>' +
                                      'Giá: %{y:,.0f} VND<br>' +
                                      'Thời gian: %{x}<br>' +
                                      '%{text}<extra></extra>'
                    ))

            # Add sell signals with hover text
            sell_points = history[history['Final_Signal'].isin(['Sell', 'Strong Sell'])]
            fig_signals.add_trace(go.Scatter(
                x=sell_points['time'],
                y=sell_points['close'],
                mode='markers',
                name='Tín hiệu Bán',
                marker=dict(color='red', size=10, symbol='triangle-down'),
                text=sell_points['Signal_Explanation'],
                hovertemplate='<b>Tín hiệu Bán</b><br>' +
                              'Giá: %{y:,.0f} VND<br>' +
                              'Thời gian: %{x}<br>' +
                              '%{text}<extra></extra>'
            ))

            fig_signals.update_layout(
                title=f'Tín hiệu giao dịch {stock_symbol} theo thời gian',
                xaxis_title='Thời gian',
                yaxis_title=f'Giá {stock_symbol} (VND)',
                hovermode='closest'
            )
            st.plotly_chart(fig_signals)

            # Update the trading signals visualization section
            st.subheader("Biểu đồ tín hiệu giao dịch")
            fig_signals = go.Figure()

            # Add price line
            fig_signals.add_trace(go.Scatter(
                x=history['time'],
                y=history['close'],
                name='Giá đóng cửa',
                line=dict(color='black', width=1)
            ))

            # Add buy and sell signals
            if 'Final_Signal' in history.columns:
                # Buy signals
                buy_points = history[history['Final_Signal'].str.contains('Buy', na=False)]
                if not buy_points.empty:
                    fig_signals.add_trace(go.Scatter(
                        x=buy_points['time'],
                        y=buy_points['close'],
                        mode='markers',
                        name='Tín hiệu Mua',
                        marker=dict(color='green', size=10, symbol='triangle-up'),
                        text=buy_points['Signal_Explanation'],  # Changed from 'Explanation'
                        hovertemplate='<b>Tín hiệu Mua</b><br>' +
                                     'Giá: %{y:,.0f} VND<br>' +
                                     'Thời gian: %{x}<br>' +
                                     '%{text}<extra></extra>'
                    ))
                
                # Sell signals
                sell_points = history[history['Final_Signal'].str.contains('Sell', na=False)]
                if not sell_points.empty:
                    fig_signals.add_trace(go.Scatter(
                        x=sell_points['time'],
                        y=sell_points['close'],
                        mode='markers',
                        name='Tín hiệu Bán',
                        marker=dict(color='red', size=10, symbol='triangle-down'),
                        text=sell_points['Signal_Explanation'],  # Changed from 'Explanation'
                        hovertemplate='<b>Tín hiệu Bán</b><br>' +
                                     'Giá: %{y:,.0f} VND<br>' +
                                     'Thời gian: %{x}<br>' +
                                     '%{text}<extra></extra>'
                    ))

            # Update layout with better formatting
            fig_signals.update_layout(
                title={
                    'text': f'Tín hiệu giao dịch {stock_symbol}',
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                xaxis_title='Thời gian',
                yaxis_title=f'Giá {stock_symbol} (VND)',
                hovermode='closest',
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )

            # Add range slider
            fig_signals.update_xaxes(rangeslider_visible=True)

            # Display the chart
            st.plotly_chart(fig_signals, use_container_width=True)

            # Add signal summary
            with st.expander("Tổng hợp tín hiệu giao dịch", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Tín hiệu Mua")
                    if not buy_points.empty:
                        for idx, row in buy_points.iterrows():
                            st.markdown(f"""
                            🔺 **{row['Final_Signal']}** ({pd.to_datetime(row['time']).strftime('%Y-%m-%d')})
                            - Giá: {row['close']:,.0f} VND
                            - {row['Signal_Explanation']}  # Changed from 'Explanation'
                            """)
                    else:
                        st.write("Không có tín hiệu mua")
                        
                with col2:
                    st.subheader("Tín hiệu Bán")
                    if not sell_points.empty:
                        for idx, row in sell_points.iterrows():
                            st.markdown(f"""
                            🔻 **{row['Final_Signal']}** ({pd.to_datetime(row['time']).strftime('%Y-%m-%d')})
                            - Giá: {row['close']:,.0f} VND
                            - {row['Signal_Explanation']}  # Changed from 'Explanation'
                            """)
                    else:
                        st.write("Không có tín hiệu bán")
        else:
            st.warning(f"Không tìm thấy dữ liệu cho mã {stock_symbol} trong khoảng thời gian này.")

    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi lấy dữ liệu: {str(e)}")
        st.write("Vui lòng kiểm tra lại mã chứng khoán hoặc kết nối.")

else:
    st.info("Vui lòng nhập mã chứng khoán và chọn khoảng thời gian, sau đó nhấn 'Hiển thị kết quả'")

# Footer
st.write(f"Cập nhật lần cuối: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")