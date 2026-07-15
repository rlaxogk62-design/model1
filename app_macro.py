import streamlit as st
import ccxt
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import warnings
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
import os

warnings.filterwarnings('ignore')

# --- 페이지 설정 ---
st.set_page_config(page_title="Macro Fusion | Live Trading", page_icon="🌍", layout="wide")
st_autorefresh(interval=60000, limit=None, key="auto_refresh_timer") # 60초 자동 새로고침

st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] { background-color: #0E1117; color: #EAECEF; }
    .main-title { font-size: 45px; font-weight: 900; color: #FFD700; text-align: center; }
    .sub-title { font-size: 18px; color: #A0AEC0; text-align: center; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🌍 Macro Fusion Pro Tracker</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Real-time BTC & Macro Indicators (NASDAQ, DXY, WTI, Yield) + Live Paper Trading</p>', unsafe_allow_html=True)

@st.cache_data(ttl=50) # 50초 캐싱
def get_live_macro_data(days=5):
    # 1. BTC 데이터 수집 (Kraken)
    exchange = ccxt.kraken()
    limit = days * 96
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='15m', limit=limit)
    btc_df = pd.DataFrame(ohlcv, columns=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'])
    btc_df['Datetime'] = pd.to_datetime(btc_df['Datetime'], unit='ms')
    btc_df.set_index('Datetime', inplace=True)
    btc_df.index = btc_df.index.tz_localize('UTC').tz_convert('Asia/Seoul').tz_localize(None)
    btc_df.rename(columns={'Close': 'BTC_Close'}, inplace=True)
    
    # 2. 매크로 데이터 수집 (yfinance)
    tickers = {'NASDAQ': 'NQ=F', 'DXY': 'DX-Y.NYB', 'OIL': 'CL=F', 'TREASURY': '^TNX'}
    df_dict = {}
    for name, ticker in tickers.items():
        tmp_df = yf.Ticker(ticker).history(interval='15m', period=f'{days}d')
        if not tmp_df.empty:
            tmp_df = tmp_df[['Close']]
            tmp_df.columns = [f'{name}_Close']
            tmp_df.index = tmp_df.index.tz_convert('Asia/Seoul').tz_localize(None)
            df_dict[name] = tmp_df
            
    # 3. 데이터 병합
    merged_df = btc_df.copy().sort_index()
    for name in tickers.keys():
        if name in df_dict:
            tmp = df_dict[name].dropna().sort_index()
            merged_df = pd.merge_asof(merged_df, tmp, left_index=True, right_index=True, direction='backward')
    
    merged_df.dropna(inplace=True)
    
    # 4. 피처 엔지니어링
    for name in ['BTC', 'NASDAQ', 'DXY', 'OIL', 'TREASURY']:
        if f'{name}_Close' in merged_df.columns:
            merged_df[f'{name}_Ret_15m'] = merged_df[f'{name}_Close'].pct_change(1)
            merged_df[f'{name}_Ret_1H'] = merged_df[f'{name}_Close'].pct_change(4)
            merged_df[f'{name}_Ret_4H'] = merged_df[f'{name}_Close'].pct_change(16)
            
    # Base_Model_Pred (훈련 시 사용했던 단순화 로직)
    merged_df['Base_Model_Pred'] = np.where(merged_df['BTC_Close'].rolling(7).mean() > merged_df['BTC_Close'].rolling(20).mean(), 2, 0)
    
    # ATR 계산용
    merged_df['Prev_Close'] = merged_df['BTC_Close'].shift(1)
    merged_df['TR'] = np.abs(merged_df['High'] - merged_df['Low'])
    merged_df['ATR_Approx'] = merged_df['TR'].rolling(window=14).mean().fillna(merged_df['BTC_Close'] * 0.003)
    
    merged_df.dropna(inplace=True)
    return merged_df

try:
    # 데이터 로드
    df = get_live_macro_data(days=5)
    
    # 모델 로드 및 추론
    model_path = './data/model/xgboost_macro_15m.pkl'
    if not os.path.exists(model_path):
        model_path = 'xgboost_macro_15m.pkl'
    model_macro = joblib.load(model_path)
    
    feature_cols = ['Base_Model_Pred', 'BTC_Ret_15m', 'BTC_Ret_1H', 'BTC_Ret_4H', 
                    'NASDAQ_Ret_15m', 'NASDAQ_Ret_1H', 'NASDAQ_Ret_4H', 
                    'DXY_Ret_15m', 'DXY_Ret_1H', 'DXY_Ret_4H', 
                    'OIL_Ret_15m', 'OIL_Ret_1H', 'OIL_Ret_4H', 
                    'TREASURY_Ret_15m', 'TREASURY_Ret_1H', 'TREASURY_Ret_4H']
    
    df['Macro_Pred'] = model_macro.predict(df[feature_cols])
    
    # --- 1. 차트 시각화 ---
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                    open=df['Open'], high=df['High'], low=df['Low'], close=df['BTC_Close'],
                    increasing_line_color='#00E676', decreasing_line_color='#FF3D00', name='BTC')])

    pred_up = df[df['Macro_Pred'] == 2]
    pred_down = df[df['Macro_Pred'] == 0]

    fig.add_trace(go.Scatter(x=pred_up.index, y=pred_up['Low'] * 0.995,
                             mode='markers', marker=dict(symbol='triangle-up', size=16, color='cyan', line=dict(width=2, color='white')), name='🟢 Macro Long'))
    fig.add_trace(go.Scatter(x=pred_down.index, y=pred_down['High'] * 1.005,
                             mode='markers', marker=dict(symbol='triangle-down', size=16, color='magenta', line=dict(width=2, color='white')), name='🔴 Macro Short'))

    fig.update_layout(title='Live Macro Signals (Last 5 Days)', template='plotly_dark', xaxis_rangeslider_visible=False, height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # --- 2. 실시간 모의투자 (Paper Trading) 시뮬레이션 ---
    st.markdown("### 💸 Live Paper Trading Simulation (최근 5일 진행형 모의투자)")
    
    # 최적화 파라미터 적용 (이전 Optuna 결과 기반)
    LEVERAGE = 11
    TP_MULT = 5.0
    SL_MULT = 1.0
    TRADE_RATIO = 0.1
    FEE_RATE = 0.0004
    
    balance = 10000.0
    position = None
    entry_price = 0
    qty = 0
    
    history_dates = []
    history_balances = []
    
    for i in range(len(df)):
        row = df.iloc[i]
        current_price = row['BTC_Close']
        atr = row['ATR_Approx']
        
        if position == 'LONG':
            if current_price >= entry_price + (atr * TP_MULT) or current_price <= entry_price - (atr * SL_MULT):
                profit = (qty * entry_price * ((current_price - entry_price) / entry_price * LEVERAGE)) - (qty * entry_price * FEE_RATE * 2)
                balance += profit
                position = None
        elif position == 'SHORT':
            if current_price <= entry_price - (atr * TP_MULT) or current_price >= entry_price + (atr * SL_MULT):
                profit = (qty * entry_price * ((entry_price - current_price) / entry_price * LEVERAGE)) - (qty * entry_price * FEE_RATE * 2)
                balance += profit
                position = None
                
        if position is None:
            if row['Macro_Pred'] == 2:
                position = 'LONG'
                entry_price = current_price
                qty = (balance * TRADE_RATIO) / entry_price * LEVERAGE
            elif row['Macro_Pred'] == 0:
                position = 'SHORT'
                entry_price = current_price
                qty = (balance * TRADE_RATIO) / entry_price * LEVERAGE
                
        history_dates.append(df.index[i])
        history_balances.append(balance)
        
    col1, col2, col3 = st.columns(3)
    current_roi = (balance - 10000.0) / 10000.0 * 100
    col1.metric("초기 모의 자산", "$10,000.00")
    col2.metric("현재 모의 자산", f"${balance:,.2f}", f"{current_roi:.2f}%")
    col3.metric("현재 포지션 상태", "대기 (None)" if position is None else position)
    
    fig_roi = go.Figure(data=[go.Scatter(x=history_dates, y=history_balances, mode='lines', line=dict(color='gold', width=3))])
    fig_roi.update_layout(title='Mock Trading Equity Curve (Real-time Updated)', template='plotly_dark', height=300)
    st.plotly_chart(fig_roi, use_container_width=True)

except Exception as e:
    st.error(f"데이터 수집 또는 모델 추론 중 오류 발생: {e}")
