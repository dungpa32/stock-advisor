import numpy as np
import pandas as pd


def calculate_rsi(data, period=14):
    """
    Tính toán chỉ báo RSI (Relative Strength Index)
    
    Args:
        data (DataFrame): DataFrame chứa dữ liệu giá
        period (int, optional): Chu kỳ của RSI. Mặc định là 14.
    
    Returns:
        Series: Series chứa giá trị RSI
    """
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """
    Tính toán chỉ báo MACD (Moving Average Convergence Divergence)
    
    Args:
        data (DataFrame): DataFrame chứa dữ liệu giá
        fast_period (int, optional): Chu kỳ của đường EMA nhanh. Mặc định là 12.
        slow_period (int, optional): Chu kỳ của đường EMA chậm. Mặc định là 26.
        signal_period (int, optional): Chu kỳ của đường tín hiệu. Mặc định là 9.
    
    Returns:
        tuple: (MACD, Signal)
    """
    exp1 = data['close'].ewm(span=fast_period, adjust=False).mean()
    exp2 = data['close'].ewm(span=slow_period, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    
    return macd, signal


def calculate_ema(data, period):
    """
    Tính toán đường trung bình động lũy thừa (EMA)
    
    Args:
        data (DataFrame): DataFrame chứa dữ liệu giá
        period (int): Chu kỳ của EMA
    
    Returns:
        Series: Series chứa giá trị EMA
    """
    return data['close'].ewm(span=period, adjust=False).mean()


def calculate_ema9(data):
    """
    Tính toán EMA với chu kỳ 9 ngày (thường dùng kết hợp với MACD)
    
    Args:
        data (DataFrame): DataFrame chứa dữ liệu giá
    
    Returns:
        Series: Series chứa giá trị EMA9
    """
    return data['close'].ewm(span=9, adjust=False).mean()


def calculate_bollinger_bands(data, period=20, std_dev=2):
    """
    Tính toán dải Bollinger Bands
    
    Args:
        data (DataFrame): DataFrame chứa dữ liệu giá
        period (int, optional): Chu kỳ của trung bình động. Mặc định là 20.
        std_dev (int, optional): Số lần độ lệch chuẩn. Mặc định là 2.
    
    Returns:
        tuple: (Upper Band, Middle Band, Lower Band)
    """
    middle_band = data['close'].rolling(window=period).mean()
    std = data['close'].rolling(window=period).std()
    upper_band = middle_band + (std_dev * std)
    lower_band = middle_band - (std_dev * std)
    
    return upper_band, middle_band, lower_band


def calculate_vwap(data):
    """
    Tính toán chỉ báo VWAP (Volume Weighted Average Price)
    
    Args:
        data (DataFrame): DataFrame chứa dữ liệu giá và khối lượng
    
    Returns:
        Series: Series chứa giá trị VWAP
    """
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    vwap = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
    return vwap


def generate_rsi_macd_signals(data):
    """
    Tạo tín hiệu giao dịch dựa trên RSI và MACD
    
    Args:
        data (DataFrame): DataFrame chứa dữ liệu giá và các chỉ báo kỹ thuật
    
    Returns:
        DataFrame: DataFrame chứa tín hiệu mua/bán
    """
    signals = pd.DataFrame(index=data.index)
    
    # RSI Signals
    signals['RSI_Buy'] = ((data['RSI'] < 30) & (data['RSI'].shift(1) >= 30))
    signals['RSI_Sell'] = ((data['RSI'] > 70) & (data['RSI'].shift(1) <= 70))
    
    # MACD Signals
    signals['MACD_Buy'] = ((data['MACD'] > data['Signal']) & 
                          (data['MACD'].shift(1) <= data['Signal'].shift(1)))
    signals['MACD_Sell'] = ((data['MACD'] < data['Signal']) & 
                           (data['MACD'].shift(1) >= data['Signal'].shift(1)))
    
    # EMA Signals
    signals['EMA_Buy'] = ((data['close'] > data['EMA50']) & 
                         (data['close'].shift(1) <= data['EMA50'].shift(1)))
    signals['EMA_Sell'] = ((data['close'] < data['EMA50']) & 
                          (data['close'].shift(1) >= data['EMA50'].shift(1)))
    
    # Bollinger Bands Signals
    signals['BB_Buy'] = (data['close'] < data['BB_Lower'])
    signals['BB_Sell'] = (data['close'] > data['BB_Upper'])
    
    # Combined Signals - more conservative approach (require multiple indicators)
    signals['Strong_Buy'] = ((signals['RSI_Buy'] & signals['MACD_Buy']) | 
                            (signals['MACD_Buy'] & signals['EMA_Buy']) |
                            (signals['RSI_Buy'] & signals['BB_Buy']))
    
    signals['Strong_Sell'] = ((signals['RSI_Sell'] & signals['MACD_Sell']) | 
                             (signals['MACD_Sell'] & signals['EMA_Sell']) |
                             (signals['RSI_Sell'] & signals['BB_Sell']))
    
    return signals


def calculate_performance(data):
    """
    Tính toán hiệu suất của chiến lược giao dịch
    
    Args:
        data (DataFrame): DataFrame chứa dữ liệu giá và tín hiệu
    
    Returns:
        dict: Dictionary chứa các chỉ số hiệu suất
    """
    # Giả lập chiến lược giao dịch đơn giản
    data = data.copy()
    data['Position'] = 0  # 0: không có vị thế, 1: đang mua
    
    # Mua khi có tín hiệu Strong_Buy
    data.loc[data['Strong_Buy'], 'Position'] = 1
    
    # Bán khi có tín hiệu Strong_Sell
    data.loc[data['Strong_Sell'], 'Position'] = 0
    
    # Forward fill: giữ vị thế cho đến khi có tín hiệu ngược lại
    data['Position'] = data['Position'].replace(to_replace=0, method='ffill')
    
    # Tính toán lợi nhuận theo ngày
    data['Returns'] = data['close'].pct_change()
    data['Strategy_Returns'] = data['Position'].shift(1) * data['Returns']
    
    # Tính toán hiệu suất
    cumulative_returns = (1 + data['Returns']).cumprod() - 1
    cumulative_strategy_returns = (1 + data['Strategy_Returns']).cumprod() - 1
    
    # Tính toán các chỉ số
    total_days = len(data)
    winning_days = len(data[data['Strategy_Returns'] > 0])
    losing_days = len(data[data['Strategy_Returns'] < 0])
    
    if total_days > 0 and cumulative_returns.iloc[-1] != 0:
        win_rate = winning_days / (winning_days + losing_days) if (winning_days + losing_days) > 0 else 0
        strategy_return = cumulative_strategy_returns.iloc[-1]
        market_return = cumulative_returns.iloc[-1]
        outperformance = strategy_return - market_return
        
        return {
            'win_rate': win_rate * 100,
            'strategy_return': strategy_return * 100,
            'market_return': market_return * 100,
            'outperformance': outperformance * 100
        }
    else:
        return {
            'win_rate': 0,
            'strategy_return': 0,
            'market_return': 0,
            'outperformance': 0
        }
