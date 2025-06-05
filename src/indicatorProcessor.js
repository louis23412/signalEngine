const isValidNumber = (value) => {
  if (value == null) return false;
  const num = typeof value === 'string' ? Number(value) : value;
  return typeof num === 'number' && !isNaN(num) && isFinite(num);
};

class IndicatorProcessor {
  compute(candles) {
    if (!Array.isArray(candles) || candles.length < 11) return { error: true };
    const validCandles = candles.filter(c =>
      isValidNumber(c.close) && isValidNumber(c.high) && isValidNumber(c.low) &&
      isValidNumber(c.volume) && c.volume >= 0 && isValidNumber(c.timestamp)
    );
    if (validCandles.length < 11) {
      return { error: true };
    }
    const close = validCandles.map(c => Number(c.close));
    const high = validCandles.map(c => Number(c.high));
    const low = validCandles.map(c => Number(c.low));
    const volume = validCandles.map(c => Number(c.volume));
    const lastClose = close[close.length - 1];
    const volumeMean = volume.reduce((sum, v) => sum + (isValidNumber(v) ? v : 0), 0) / volume.length || 1;
    const volumeStd = Math.sqrt(
      volume.reduce((sum, v) => sum + (isValidNumber(v) ? (v - volumeMean) ** 2 : 0), 0) / volume.length
    ) || 1;
    const volumeZScore = isValidNumber(volume[volume.length - 1])
      ? (volume[volume.length - 1] - volumeMean) / volumeStd
      : 0;
    const indicators = {
      close,
      high,
      low,
      volume,
      lastClose,
      volumeZScore: Math.min(Math.max(volumeZScore, -3), 3),
      rsi: this.#computeRSI(close, 14),
      macd: this.#computeMACD(close, 8, 21, 5),
      atr: this.#computeATR(high, low, close, 14),
      isTrending: 0,
      isRanging: 0,
      marketPhase: 'neutral'
    };
    const lastMacd = indicators.macd[indicators.macd.length - 1];
    const lastRsi = indicators.rsi[indicators.rsi.length - 1];
    indicators.isTrending = lastMacd && isValidNumber(lastMacd.MACD) && isValidNumber(lastMacd.signal) && lastMacd.MACD > lastMacd.signal ? 1 : 0;
    indicators.isRanging = isValidNumber(lastRsi) && lastRsi > 30 && lastRsi < 70 ? 1 : 0;
    indicators.marketPhase = indicators.isTrending ? 'trending' : indicators.isRanging ? 'ranging' : 'volatile';
    indicators.rsiMin = Math.min(...indicators.rsi.filter(isValidNumber)) || 0;
    indicators.rsiMax = Math.max(...indicators.rsi.filter(isValidNumber)) || 100;
    indicators.macdMin = Math.min(...indicators.macd.map(m => m.MACD - m.signal).filter(isValidNumber)) || -1;
    indicators.macdMax = Math.max(...indicators.macd.map(m => m.MACD - m.signal).filter(isValidNumber)) || 1;
    indicators.atrMin = Math.min(...indicators.atr.filter(isValidNumber)) || 0;
    indicators.atrMax = Math.max(...indicators.atr.filter(isValidNumber)) || 1;
    return indicators;
  }

  #computeRSI(values, period) {
    if (!Array.isArray(values) || values.length < period + 1 || !values.every(isValidNumber)) {
      return new Array(Math.max(0, values.length - period)).fill(50);
    }
    let gains = 0, losses = 0;
    const rsi = new Array(values.length - period);
    for (let i = 1; i <= period; i++) {
      const delta = values[i] - values[i - 1];
      if (!isValidNumber(delta)) return new Array(values.length - period).fill(50);
      gains += delta > 0 ? delta : 0;
      losses += delta < 0 ? -delta : 0;
    }
    let avgGain = gains / period;
    let avgLoss = losses / period;
    if (!isValidNumber(avgGain) || !isValidNumber(avgLoss)) return new Array(values.length - period).fill(50);
    for (let i = period, j = 0; j < rsi.length; i++, j++) {
      const delta = values[i] - values[i - 1];
      const gain = delta > 0 ? delta : 0;
      const loss = delta < 0 ? -delta : 0;
      avgGain = (avgGain * (period - 1) + gain) / period;
      avgLoss = (avgLoss * (period - 1) + loss) / period;
      if (!isValidNumber(avgGain) || !isValidNumber(avgLoss)) {
        rsi[j] = 50;
        continue;
      }
      const rs = avgLoss <= 0 ? 100 : avgGain / avgLoss;
      rsi[j] = isValidNumber(rs) ? 100 - (100 / (1 + rs)) : 50;
    }
    return rsi;
  }

  #computeMACD(values, fastPeriod, slowPeriod, signalPeriod) {
    if (!Array.isArray(values) || values.length < slowPeriod + signalPeriod || !values.every(isValidNumber)) {
      return new Array(Math.max(0, values.length - slowPeriod - signalPeriod + 1)).fill({ MACD: 0, signal: 0, histogram: 0 });
    }
    const median = values.slice().sort((a, b) => a - b)[Math.floor(values.length / 2)] || 1;
    const normalizedValues = values.map(v => v / median);
    const fastEMA = this.#computeEMA(normalizedValues, fastPeriod);
    const slowEMA = this.#computeEMA(normalizedValues, slowPeriod);
    const macdLine = new Array(normalizedValues.length - (slowPeriod - fastPeriod)).fill(0);
    for (let i = slowPeriod - fastPeriod, j = 0; i < normalizedValues.length; i++, j++) {
      const diff = fastEMA[i] - slowEMA[i - (slowPeriod - fastPeriod)];
      macdLine[j] = isValidNumber(diff) ? Math.min(Math.max(diff, -1000), 1000) : 0;
    }
    const signalLine = this.#computeEMA(macdLine.slice(0, macdLine.length - signalPeriod + 1), signalPeriod);
    const macd = new Array(macdLine.length - signalPeriod + 1).fill({ MACD: 0, signal: 0, histogram: 0 });
    for (let i = signalPeriod - 1, j = 0; i < macdLine.length; i++, j++) {
      const signal = signalLine[j] || 0;
      const macdVal = macdLine[i];
      const hist = macdVal - signal;
      macd[j] = {
        MACD: isValidNumber(macdVal) ? macdVal : 0,
        signal: isValidNumber(signal) ? signal : 0,
        histogram: isValidNumber(hist) ? hist : 0
      };
    }
    return macd;
  }

  #computeEMA(values, period) {
    if (!Array.isArray(values) || values.length < period || !values.every(isValidNumber)) return new Array(values.length).fill(0);
    const alpha = 2 / (period + 1);
    const ema = new Array(values.length);
    const initialValues = values.slice(0, period);
    const initialAvg = initialValues.reduce((sum, v) => sum + v, 0) / period;
    if (!isValidNumber(initialAvg)) return new Array(values.length).fill(0);
    ema[period - 1] = Math.min(Math.max(initialAvg, -10000), 10000);
    for (let i = period; i < values.length; i++) {
      const next = alpha * values[i] + (1 - alpha) * ema[i - 1];
      ema[i] = isValidNumber(next) ? Math.min(Math.max(next, -10000), 10000) : 0;
    }
    for (let i = 0; i < period - 1; i++) {
      ema[i] = 0;
    }
    return ema;
  }

  #computeATR(high, low, close, period) {
    if (
      !Array.isArray(high) || high.length < period + 1 ||
      high.length !== low.length || low.length !== close.length ||
      !high.every(isValidNumber) || !low.every(isValidNumber) || !close.every(isValidNumber)
    ) return new Array(Math.max(0, high.length - period)).fill(0);
    const tr = new Array(close.length - 1);
    for (let i = 1; i < close.length; i++) {
      const highLow = high[i] - low[i];
      const highClose = Math.abs(high[i] - close[i - 1]);
      const lowClose = Math.abs(low[i] - close[i - 1]);
      tr[i - 1] = isValidNumber(highLow) && isValidNumber(highClose) && isValidNumber(lowClose)
        ? Math.max(highLow, highClose, lowClose)
        : 0;
    }
    const atr = new Array(tr.length - period + 1);
    let sum = tr.slice(0, period).reduce((s, v) => s + (isValidNumber(v) ? v : 0), 0);
    atr[0] = isValidNumber(sum) ? Math.min(Math.max(sum / period, 0), 1000) : 0;
    for (let i = period, j = 1; i < tr.length; i++, j++) {
      const next = (atr[j - 1] * (period - 1) + (isValidNumber(tr[i]) ? tr[i] : 0)) / period;
      atr[j] = isValidNumber(next) ? Math.min(Math.max(next, 0), 1000) : 0;
    }
    return atr;
  }
}

export default IndicatorProcessor;