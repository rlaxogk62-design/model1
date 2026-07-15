import streamlit as st
import ccxt
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import warnings
from datetime import datetime, timedelta, date
from streamlit_autorefresh import st_autorefresh
import os

warnings.filterwarnings('ignore')

# --- 페이지 설정 ---
st.set_page_config(page_title="Macro Fusion | Live Trading", page_icon="🌍", layout="wide")
st_autorefresh(interval=60000, limit=None, key="auto_refresh_timer")

st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] { background-color: #0E1117; color: #EAECEF; }
    .main-title { font-size: 45px; font-weight: 900; color: #FFD700; text-align: center; }
    .sub-title { font-size: 18px; color: #A0AEC0; text-align: center; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🌍 Macro Fusion Pro Tracker</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Real-time BTC & Macro Indicators + Dual Strategy Paper Trading</p>', unsafe_allow_html=True)

st.sidebar.title("⚙️ System Status")
st.sidebar.markdown("---")

# 차트 표시용 슬라이더
chart_days = st.sidebar.slider("📊 차트 표시 기간 (일)", min_value=1, max_value=7, value=2)

# 시뮬레이션용 시작일
default_start_date = date(datetime.now().year, 7, 15)
start_date = st.sidebar.date_input("📅 시뮬레이션 시작일", value=default_start_date)

@st.cache_data(ttl=50)
def get_live_macro_data(start_d):
    days = (datetime.now().date() - start_d).days
    if days <= 0:
        days = 1

    exchange = ccxt.kraken()
    limit = (days + 2) * 96
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='15m', limit=limit)
    btc_df = pd.DataFrame(ohlcv, columns=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'])
    btc_df['Datetime'] = pd.to_datetime(btc_df['Datetime'], unit='ms')
    btc_df.set_index('Datetime', inplace=True)
    btc_df.index = btc_df.index.tz_localize('UTC').tz_convert('Asia/Seoul').tz_localize(None).astype('datetime64[ns]')
    btc_df.rename(columns={'Close': 'BTC_Close'}, inplace=True)

    tickers = {'NASDAQ': 'NQ=F', 'DXY': 'DX-Y.NYB', 'OIL': 'CL=F', 'TREASURY': '^TNX'}
    df_dict = {}
    for name, ticker in tickers.items():
        tmp_df = yf.Ticker(ticker).history(interval='15m', period=f'{days+2}d')
        if not tmp_df.empty:
            tmp_df = tmp_df[['Close']]
            tmp_df.columns = [f'{name}_Close']
            tmp_df.index = tmp_df.index.tz_convert('Asia/Seoul').tz_localize(None).astype('datetime64[ns]')
            df_dict[name] = tmp_df

    merged_df = btc_df.copy().sort_index()
    for name in tickers.keys():
        if name in df_dict:
            tmp = df_dict[name].dropna().sort_index()
            merged_df = pd.merge_asof(merged_df, tmp, left_index=True, right_index=True, direction='backward')

    merged_df.dropna(inplace=True)

    for name in ['BTC', 'NASDAQ', 'DXY', 'OIL', 'TREASURY']:
        if f'{name}_Close' in merged_df.columns:
            merged_df[f'{name}_Ret_15m'] = merged_df[f'{name}_Close'].pct_change(1)
            merged_df[f'{name}_Ret_1H'] = merged_df[f'{name}_Close'].pct_change(4)
            merged_df[f'{name}_Ret_4H'] = merged_df[f'{name}_Close'].pct_change(16)

    merged_df['Base_Model_Pred'] = np.where(merged_df['BTC_Close'].rolling(7).mean() > merged_df['BTC_Close'].rolling(20).mean(), 2, 0)

    merged_df['Prev_Close'] = merged_df['BTC_Close'].shift(1)
    merged_df['TR'] = np.abs(merged_df['High'] - merged_df['Low'])
    merged_df['ATR_Approx'] = merged_df['TR'].rolling(window=14).mean().fillna(merged_df['BTC_Close'] * 0.003)

    merged_df.dropna(inplace=True)
    return merged_df

try:
    df = get_live_macro_data(start_date)

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
    chart_points = chart_days * 96
    chart_df = df.iloc[-chart_points:] if len(df) > chart_points else df

    fig = go.Figure(data=[go.Candlestick(x=chart_df.index,
                    open=chart_df['Open'], high=chart_df['High'], low=chart_df['Low'], close=chart_df['BTC_Close'],
                    increasing_line_color='#00E676', decreasing_line_color='#FF3D00', name='BTC')])

    pred_up = chart_df[chart_df['Macro_Pred'] == 2]
    pred_down = chart_df[chart_df['Macro_Pred'] == 0]

    fig.add_trace(go.Scatter(x=pred_up.index, y=pred_up['Low'] * 0.995,
                             mode='markers', marker=dict(symbol='triangle-up', size=16, color='cyan', line=dict(width=2, color='white')), name='🟢 Macro Long'))
    fig.add_trace(go.Scatter(x=pred_down.index, y=pred_down['High'] * 1.005,
                             mode='markers', marker=dict(symbol='triangle-down', size=16, color='magenta', line=dict(width=2, color='white')), name='🔴 Macro Short'))

    fig.update_layout(
        title=f'Live Macro Signals (Last {chart_days} Days)',
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        height=500,
        uirevision='live_chart',
        dragmode='pan'
    )
    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': False})

    # --- 2. 과거 데이터 기반 모의투자 (Paper Trading) 시뮬레이션 ---
    st.markdown(f"### 💸 Dual Paper Trading Simulation (Since {start_date} 00:00)")

    sim_start_dt = pd.to_datetime(start_date)
    sim_df = df[df.index >= sim_start_dt]

    FEE_RATE = 0.0004

    # (A) 매크로 모델 (AI 최적화 결과)
    MACRO_LEVERAGE = 10
    MACRO_TRADE_RATIO = 0.2
    MACRO_TP_MULT = 1.0
    MACRO_SL_MULT = 1.0

    m_balance = 10000.0
    m_position = None
    m_entry_price = 0.0
    m_qty = 0.0
    m_balances = []

    for i in range(len(sim_df)):
        if m_balance <= 100:
            m_balances.append(m_balance)
            continue

        row = sim_df.iloc[i]
        c_price = row['BTC_Close']
        c_high = row['High']
        c_low = row['Low']
        c_atr = row['ATR_Approx']
        c_pred = row['Macro_Pred']

        if m_position == 'LONG':
            liq_price = m_entry_price * (1 - 1/MACRO_LEVERAGE)
            if c_low <= liq_price:
                m_balance -= (m_entry_price * m_qty / MACRO_LEVERAGE)
                m_position = None
            elif c_high >= m_entry_price + (c_atr * MACRO_TP_MULT) or c_low <= m_entry_price - (c_atr * MACRO_SL_MULT):
                profit = (m_qty * (c_price - m_entry_price)) - (m_qty * c_price * FEE_RATE) - (m_qty * m_entry_price * FEE_RATE)
                m_balance += profit
                m_position = None
        elif m_position == 'SHORT':
            liq_price = m_entry_price * (1 + 1/MACRO_LEVERAGE)
            if c_high >= liq_price:
                m_balance -= (m_entry_price * m_qty / MACRO_LEVERAGE)
                m_position = None
            elif c_low <= m_entry_price - (c_atr * MACRO_TP_MULT) or c_high >= m_entry_price + (c_atr * MACRO_SL_MULT):
                profit = (m_qty * (m_entry_price - c_price)) - (m_qty * c_price * FEE_RATE) - (m_qty * m_entry_price * FEE_RATE)
                m_balance += profit
                m_position = None

        if m_position is None:
            if c_pred == 2:
                m_position = 'LONG'
                m_entry_price = c_price
                m_qty = (m_balance * MACRO_TRADE_RATIO * MACRO_LEVERAGE) / c_price
            elif c_pred == 0:
                m_position = 'SHORT'
                m_entry_price = c_price
                m_qty = (m_balance * MACRO_TRADE_RATIO * MACRO_LEVERAGE) / c_price

        # 실시간 자산 계산
        m_equity = m_balance
        if m_position == 'LONG':
            m_equity += (m_qty * (c_price - m_entry_price)) - (m_qty * c_price * FEE_RATE) - (m_qty * m_entry_price * FEE_RATE)
        elif m_position == 'SHORT':
            m_equity += (m_qty * (m_entry_price - c_price)) - (m_qty * c_price * FEE_RATE) - (m_qty * m_entry_price * FEE_RATE)
        m_balances.append(m_equity)

    # (B) 사용자 커스텀 로직 (스위칭 & 횡보 10회 청산)
    CUSTOM_LEVERAGE = 10
    CUSTOM_TRADE_RATIO = 0.2
    CUSTOM_NEUTRAL_LIMIT = 10

    c_balance = 10000.0
    c_position = None
    c_entry_price = 0.0
    c_qty = 0.0
    c_balances = []
    neutral_count = 0

    for i in range(len(sim_df)):
        if c_balance <= 100:
            c_balances.append(c_balance)
            continue

        row = sim_df.iloc[i]
        c_price = row['BTC_Close']
        c_high = row['High']
        c_low = row['Low']
        c_pred = row['Macro_Pred']

        # 횡보 카운트 업데이트
        if c_pred == 1:
            neutral_count += 1
        else:
            neutral_count = 0

        # 기존 포지션 관리
        if c_position == 'LONG':
            liq_price = c_entry_price * (1 - 1/CUSTOM_LEVERAGE)
            if c_low <= liq_price:
                c_balance -= (c_entry_price * c_qty / CUSTOM_LEVERAGE)
                c_position = None
                neutral_count = 0
            elif neutral_count >= CUSTOM_NEUTRAL_LIMIT:
                # 횡보 10연속 강제 청산
                profit = (c_qty * (c_price - c_entry_price)) - (c_qty * c_price * FEE_RATE) - (c_qty * c_entry_price * FEE_RATE)
                c_balance += profit
                c_position = None
                neutral_count = 0
            elif c_pred == 0:
                # 스위칭 (Long -> Short)
                profit = (c_qty * (c_price - c_entry_price)) - (c_qty * c_price * FEE_RATE) - (c_qty * c_entry_price * FEE_RATE)
                c_balance += profit
                c_position = 'SHORT'
                c_entry_price = c_price
                c_qty = (c_balance * CUSTOM_TRADE_RATIO * CUSTOM_LEVERAGE) / c_price

        elif c_position == 'SHORT':
            liq_price = c_entry_price * (1 + 1/CUSTOM_LEVERAGE)
            if c_high >= liq_price:
                c_balance -= (c_entry_price * c_qty / CUSTOM_LEVERAGE)
                c_position = None
                neutral_count = 0
            elif neutral_count >= CUSTOM_NEUTRAL_LIMIT:
                # 횡보 10연속 강제 청산
                profit = (c_qty * (c_entry_price - c_price)) - (c_qty * c_price * FEE_RATE) - (c_qty * c_entry_price * FEE_RATE)
                c_balance += profit
                c_position = None
                neutral_count = 0
            elif c_pred == 2:
                # 스위칭 (Short -> Long)
                profit = (c_qty * (c_entry_price - c_price)) - (c_qty * c_price * FEE_RATE) - (c_qty * c_entry_price * FEE_RATE)
                c_balance += profit
                c_position = 'LONG'
                c_entry_price = c_price
                c_qty = (c_balance * CUSTOM_TRADE_RATIO * CUSTOM_LEVERAGE) / c_price

        # 신규 진입 (포지션 없을 때)
        if c_position is None:
            if c_pred == 2:
                c_position = 'LONG'
                c_entry_price = c_price
                c_qty = (c_balance * CUSTOM_TRADE_RATIO * CUSTOM_LEVERAGE) / c_price
            elif c_pred == 0:
                c_position = 'SHORT'
                c_entry_price = c_price
                c_qty = (c_balance * CUSTOM_TRADE_RATIO * CUSTOM_LEVERAGE) / c_price

        # 실시간 자산 계산
        c_equity = c_balance
        if c_position == 'LONG':
            c_equity += (c_qty * (c_price - c_entry_price)) - (c_qty * c_price * FEE_RATE) - (c_qty * c_entry_price * FEE_RATE)
        elif c_position == 'SHORT':
            c_equity += (c_qty * (c_entry_price - c_price)) - (c_qty * c_price * FEE_RATE) - (c_qty * c_entry_price * FEE_RATE)
        c_balances.append(c_equity)

    # --- 3. 결과 시각화 ---
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🌍 매크로 모델 (TP/SL 1.0, 10x, 20%)")
        final_m_equity = m_balances[-1] if m_balances else 10000.0
        macro_roi = (final_m_equity - 10000.0) / 10000.0 * 100
        st.metric("실시간 자산 (Real-time Equity)", f"${final_m_equity:,.2f}", f"{macro_roi:.2f}%")
        st.metric("현재 포지션", "대기 (None)" if m_position is None else f"{m_position} (진입가: ${m_entry_price:,.2f})")

    with col2:
        st.markdown("#### 💡 사용자 맞춤 로직 (스위칭 & 횡보 10회 청산)")
        final_c_equity = c_balances[-1] if c_balances else 10000.0
        custom_roi = (final_c_equity - 10000.0) / 10000.0 * 100
        st.metric("실시간 자산 (Real-time Equity)", f"${final_c_equity:,.2f}", f"{custom_roi:.2f}%")
        st.metric("현재 포지션", "대기 (None)" if c_position is None else f"{c_position} (진입가: ${c_entry_price:,.2f})")

    fig_roi = go.Figure()
    fig_roi.add_trace(go.Scatter(x=sim_df.index, y=m_balances, mode='lines', line=dict(color='gold', width=3), name='Macro Strategy (TP/SL 1.0)'))
    fig_roi.add_trace(go.Scatter(x=sim_df.index, y=c_balances, mode='lines', line=dict(color='#00E676', width=3), name='Custom Strategy (Switching)'))
    fig_roi.update_layout(title=f'Dual Mock Trading Equity Curve (Since {start_date} 00:00)', template='plotly_dark', height=400)
    st.plotly_chart(fig_roi, use_container_width=True)

except Exception as e:
    st.error(f"데이터 수집 또는 모델 추론 중 오류 발생: {e}")
