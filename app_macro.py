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
st.markdown('<p class="sub-title">Real-time BTC & Macro Indicators + Dual Strategy Paper Trading</p>', unsafe_allow_html=True)

st.sidebar.title("⚙️ System Status")
st.sidebar.markdown("---")
chart_days = st.sidebar.slider("📊 차트 표시 기간 (일)", min_value=1, max_value=30, value=5)

@st.cache_data(ttl=50)
def get_live_macro_data(days=5):
    # 1. BTC 데이터 수집 (Kraken)
    exchange = ccxt.kraken()
    limit = days * 96
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='15m', limit=limit)
    btc_df = pd.DataFrame(ohlcv, columns=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'])
    btc_df['Datetime'] = pd.to_datetime(btc_df['Datetime'], unit='ms')
    btc_df.set_index('Datetime', inplace=True)
    btc_df.index = btc_df.index.tz_localize('UTC').tz_convert('Asia/Seoul').tz_localize(None).astype('datetime64[ns]')
    btc_df.rename(columns={'Close': 'BTC_Close'}, inplace=True)

    # 2. 매크로 데이터 수집 (yfinance)
    tickers = {'NASDAQ': 'NQ=F', 'DXY': 'DX-Y.NYB', 'OIL': 'CL=F', 'TREASURY': '^TNX'}
    df_dict = {}
    for name, ticker in tickers.items():
        tmp_df = yf.Ticker(ticker).history(interval='15m', period=f'{days}d')
        if not tmp_df.empty:
            tmp_df = tmp_df[['Close']]
            tmp_df.columns = [f'{name}_Close']
            tmp_df.index = tmp_df.index.tz_convert('Asia/Seoul').tz_localize(None).astype('datetime64[ns]')
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
    df = get_live_macro_data(days=chart_days)

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

    fig.update_layout(
        title=f'Live Macro Signals (Last {chart_days} Days)',
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        height=500,
        uirevision='live_chart',
        dragmode='pan'
    )
    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': False})

    # --- 2. 실시간 모의투자 (Paper Trading) 시뮬레이션 ---
    st.markdown("### 💸 Dual Live Paper Trading Simulation (현실적 레버리지 반영)")

    # (A) 매크로 모델 상태 초기화
    if 'macro_balance' not in st.session_state:
        st.session_state.macro_balance = 10000.0
        st.session_state.macro_position = None
        st.session_state.macro_entry_price = 0.0
        st.session_state.macro_qty = 0.0
        st.session_state.history_dates = []
        st.session_state.macro_history_balances = []

    # (B) 사용자 커스텀 로직(모델1) 상태 초기화
    if 'custom_balance' not in st.session_state:
        st.session_state.custom_balance = 10000.0
        st.session_state.custom_position = None
        st.session_state.custom_entry_price = 0.0
        st.session_state.custom_qty = 0.0
        st.session_state.custom_history_balances = []
        # 로직 카운터
        st.session_state.custom_long_count = 0
        st.session_state.custom_short_count = 0
        st.session_state.custom_neutral_streak = 0

    # --- 공통 파라미터 ---
    LEVERAGE = 19       # 사용자 최적화 결과 반영
    TRADE_RATIO = 0.5   # 50% 시드 비중
    FEE_RATE = 0.0004
    # 매크로 최적화 파라미터
    MACRO_TP_MULT = 3.5
    MACRO_SL_MULT = 3.0

    # 현재 실시간 데이터
    current_row = df.iloc[-1]
    current_price = current_row['BTC_Close']
    current_high = current_row['High']
    current_low = current_row['Low']
    atr = current_row['ATR_Approx']
    current_macro_pred = current_row['Macro_Pred']
    current_base_pred = current_row['Base_Model_Pred']
    now_time = pd.Timestamp.now()

    # -----------------------------------------------------
    # (A) 매크로 모델 매매 로직
    # -----------------------------------------------------
    if st.session_state.macro_balance > 100:
        if st.session_state.macro_position == 'LONG':
            liq_price = st.session_state.macro_entry_price * (1 - 1/LEVERAGE)
            if current_low <= liq_price:
                st.session_state.macro_balance -= (st.session_state.macro_entry_price * st.session_state.macro_qty / LEVERAGE)
                st.session_state.macro_position = None
            elif current_high >= st.session_state.macro_entry_price + (atr * MACRO_TP_MULT) or current_low <= st.session_state.macro_entry_price - (atr * MACRO_SL_MULT):
                profit = (st.session_state.macro_qty * (current_price - st.session_state.macro_entry_price)) - (st.session_state.macro_qty * current_price * FEE_RATE) - (st.session_state.macro_qty * st.session_state.macro_entry_price * FEE_RATE)
                st.session_state.macro_balance += profit
                st.session_state.macro_position = None

        elif st.session_state.macro_position == 'SHORT':
            liq_price = st.session_state.macro_entry_price * (1 + 1/LEVERAGE)
            if current_high >= liq_price:
                st.session_state.macro_balance -= (st.session_state.macro_entry_price * st.session_state.macro_qty / LEVERAGE)
                st.session_state.macro_position = None
            elif current_low <= st.session_state.macro_entry_price - (atr * MACRO_TP_MULT) or current_high >= st.session_state.macro_entry_price + (atr * MACRO_SL_MULT):
                profit = (st.session_state.macro_qty * (st.session_state.macro_entry_price - current_price)) - (st.session_state.macro_qty * current_price * FEE_RATE) - (st.session_state.macro_qty * st.session_state.macro_entry_price * FEE_RATE)
                st.session_state.macro_balance += profit
                st.session_state.macro_position = None

        if st.session_state.macro_position is None:
            if current_macro_pred == 2:
                st.session_state.macro_position = 'LONG'
                st.session_state.macro_entry_price = current_price
                st.session_state.macro_qty = (st.session_state.macro_balance * TRADE_RATIO * LEVERAGE) / current_price
            elif current_macro_pred == 0:
                st.session_state.macro_position = 'SHORT'
                st.session_state.macro_entry_price = current_price
                st.session_state.macro_qty = (st.session_state.macro_balance * TRADE_RATIO * LEVERAGE) / current_price

    # -----------------------------------------------------
    # (B) 사용자 커스텀 로직 매매 로직 (최적화 파라미터 적용)
    # -----------------------------------------------------
    if st.session_state.custom_balance > 100:
        if current_base_pred == 1:
            st.session_state.custom_neutral_streak += 1
        else:
            st.session_state.custom_neutral_streak = 0

        if current_base_pred == 2:
            st.session_state.custom_long_count += 1
        elif current_base_pred == 0:
            st.session_state.custom_short_count += 1

        if st.session_state.custom_position == 'LONG':
            liq_price = st.session_state.custom_entry_price * (1 - 1/LEVERAGE)
            if current_low <= liq_price:
                st.session_state.custom_balance -= (st.session_state.custom_entry_price * st.session_state.custom_qty / LEVERAGE)
                st.session_state.custom_position = None
                st.session_state.custom_long_count = 0; st.session_state.custom_short_count = 0; st.session_state.custom_neutral_streak = 0
            else:
                # 최적화된 횡보 10회 청산
                if st.session_state.custom_neutral_streak >= 10:
                    profit = (st.session_state.custom_qty * (current_price - st.session_state.custom_entry_price)) - (st.session_state.custom_qty * current_price * FEE_RATE) - (st.session_state.custom_qty * st.session_state.custom_entry_price * FEE_RATE)
                    st.session_state.custom_balance += profit
                    st.session_state.custom_position = None
                    st.session_state.custom_long_count = 0; st.session_state.custom_short_count = 0; st.session_state.custom_neutral_streak = 0
                # 최적화된 스위칭 1회 신호
                elif st.session_state.custom_short_count >= 1:
                    profit = (st.session_state.custom_qty * (current_price - st.session_state.custom_entry_price)) - (st.session_state.custom_qty * current_price * FEE_RATE) - (st.session_state.custom_qty * st.session_state.custom_entry_price * FEE_RATE)
                    st.session_state.custom_balance += profit
                    st.session_state.custom_position = 'SHORT'
                    st.session_state.custom_entry_price = current_price
                    st.session_state.custom_qty = (st.session_state.custom_balance * TRADE_RATIO * LEVERAGE) / current_price
                    st.session_state.custom_long_count = 0; st.session_state.custom_short_count = 0; st.session_state.custom_neutral_streak = 0

        elif st.session_state.custom_position == 'SHORT':
            liq_price = st.session_state.custom_entry_price * (1 + 1/LEVERAGE)
            if current_high >= liq_price:
                st.session_state.custom_balance -= (st.session_state.custom_entry_price * st.session_state.custom_qty / LEVERAGE)
                st.session_state.custom_position = None
                st.session_state.custom_long_count = 0; st.session_state.custom_short_count = 0; st.session_state.custom_neutral_streak = 0
            else:
                # 최적화된 횡보 10회 청산
                if st.session_state.custom_neutral_streak >= 10:
                    profit = (st.session_state.custom_qty * (st.session_state.custom_entry_price - current_price)) - (st.session_state.custom_qty * current_price * FEE_RATE) - (st.session_state.custom_qty * st.session_state.custom_entry_price * FEE_RATE)
                    st.session_state.custom_balance += profit
                    st.session_state.custom_position = None
                    st.session_state.custom_long_count = 0; st.session_state.custom_short_count = 0; st.session_state.custom_neutral_streak = 0
                # 최적화된 스위칭 1회 신호
                elif st.session_state.custom_long_count >= 1:
                    profit = (st.session_state.custom_qty * (st.session_state.custom_entry_price - current_price)) - (st.session_state.custom_qty * current_price * FEE_RATE) - (st.session_state.custom_qty * st.session_state.custom_entry_price * FEE_RATE)
                    st.session_state.custom_balance += profit
                    st.session_state.custom_position = 'LONG'
                    st.session_state.custom_entry_price = current_price
                    st.session_state.custom_qty = (st.session_state.custom_balance * TRADE_RATIO * LEVERAGE) / current_price
                    st.session_state.custom_long_count = 0; st.session_state.custom_short_count = 0; st.session_state.custom_neutral_streak = 0

        if st.session_state.custom_position is None:
            # 최적화된 진입 1회 신호
            if st.session_state.custom_long_count >= 1:
                st.session_state.custom_position = 'LONG'
                st.session_state.custom_entry_price = current_price
                st.session_state.custom_qty = (st.session_state.custom_balance * TRADE_RATIO * LEVERAGE) / current_price
                st.session_state.custom_long_count = 0; st.session_state.custom_short_count = 0; st.session_state.custom_neutral_streak = 0
            elif st.session_state.custom_short_count >= 1:
                st.session_state.custom_position = 'SHORT'
                st.session_state.custom_entry_price = current_price
                st.session_state.custom_qty = (st.session_state.custom_balance * TRADE_RATIO * LEVERAGE) / current_price
                st.session_state.custom_long_count = 0; st.session_state.custom_short_count = 0; st.session_state.custom_neutral_streak = 0

    # -----------------------------------------------------
    # 기록 및 시각화
    # -----------------------------------------------------
    st.session_state.history_dates.append(now_time)
    st.session_state.macro_history_balances.append(st.session_state.macro_balance)
    st.session_state.custom_history_balances.append(st.session_state.custom_balance)

    if len(st.session_state.history_dates) > 1000:
        st.session_state.history_dates = st.session_state.history_dates[-1000:]
        st.session_state.macro_history_balances = st.session_state.macro_history_balances[-1000:]
        st.session_state.custom_history_balances = st.session_state.custom_history_balances[-1000:]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🌍 매크로 모델 (Macro Pred)")
        macro_roi = (st.session_state.macro_balance - 10000.0) / 10000.0 * 100
        st.metric("자산 (Live)", f"${st.session_state.macro_balance:,.2f}", f"{macro_roi:.2f}%")
        st.metric("포지션", "대기 (None)" if st.session_state.macro_position is None else st.session_state.macro_position)

    with col2:
        st.markdown("#### 💡 사용자 맞춤 로직 (Opt. 19x)")
        custom_roi = (st.session_state.custom_balance - 10000.0) / 10000.0 * 100
        st.metric("자산 (Live)", f"${st.session_state.custom_balance:,.2f}", f"{custom_roi:.2f}%")
        st.metric("포지션", "대기 (None)" if st.session_state.custom_position is None else st.session_state.custom_position)

    fig_roi = go.Figure()
    fig_roi.add_trace(go.Scatter(x=st.session_state.history_dates, y=st.session_state.macro_history_balances, mode='lines', line=dict(color='gold', width=3), name='Macro Strategy'))
    fig_roi.add_trace(go.Scatter(x=st.session_state.history_dates, y=st.session_state.custom_history_balances, mode='lines', line=dict(color='#00E676', width=3), name='Optimized Custom Strategy'))
    fig_roi.update_layout(title='Dual Mock Trading Equity Curve (Starts Now)', template='plotly_dark', height=300)
    st.plotly_chart(fig_roi, use_container_width=True)

except Exception as e:
    st.error(f"데이터 수집 또는 모델 추론 중 오류 발생: {e}")
