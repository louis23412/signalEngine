import fs from 'fs';
import path from 'path';
import zlib from 'zlib';
import Database from 'better-sqlite3';

const directoryPath = path.join(import.meta.dirname, '..', 'state')

const truncateToDecimals = (value, decimals) => {
  const factor = Math.pow(10, decimals);
  return Math.floor(value * factor) / factor;
};

const isValidNumber = (value) => {
  if (value == null) return false;
  const num = typeof value === 'string' ? Number(value) : value;
  return typeof num === 'number' && !isNaN(num) && isFinite(num);
};

const sigmoid = (x) => isValidNumber(x) ? 1 / (1 + Math.exp(-Math.min(Math.max(x, -100), 100))) : 0;

const softmax = (arr) => {
  if (!arr.every(isValidNumber)) return arr.map(() => 1 / arr.length);
  const max = Math.max(...arr);
  const exp = arr.map(x => Math.exp(x - max));
  const sum = exp.reduce((a, b) => a + b, 0) || 1;
  return exp.map(x => x / sum);
};

class Transformer {
  #inputSize = 6;
  #hiddenSize = 8;
  #outputSize = 1;
  #heads = 4;
  #weightsQ;
  #weightsK;
  #weightsV;
  #weightsOut;
  #biasQ;
  #biasK;
  #biasV;
  #biasOut;
  #learningRate = 0.0005;

  constructor() {
    const xavierInit = (rows, cols) => Array(rows).fill().map(() => Array(cols).fill().map(() => (Math.random() - 0.5) * Math.sqrt(2 / (rows + cols))));
    this.#weightsQ = xavierInit(this.#inputSize, this.#hiddenSize);
    this.#weightsK = xavierInit(this.#inputSize, this.#hiddenSize);
    this.#weightsV = xavierInit(this.#inputSize, this.#hiddenSize);
    this.#weightsOut = xavierInit(this.#hiddenSize, this.#outputSize);
    this.#biasQ = Array(this.#hiddenSize).fill(0);
    this.#biasK = Array(this.#hiddenSize).fill(0);
    this.#biasV = Array(this.#hiddenSize).fill(0);
    this.#biasOut = Array(this.#outputSize).fill(0);
  }

  getWeightsQ() { return this.#weightsQ; }
  getWeightsK() { return this.#weightsK; }
  getWeightsV() { return this.#weightsV; }
  getWeightsOut() { return this.#weightsOut; }
  getBiasQ() { return this.#biasQ; }
  getBiasK() { return this.#biasK; }
  getBiasV() { return this.#biasV; }
  getBiasOut() { return this.#biasOut; }

  setWeightsQ(weights) {
    if (this.#isValidMatrix(weights, this.#inputSize, this.#hiddenSize)) this.#weightsQ = weights;
  }
  setWeightsK(weights) {
    if (this.#isValidMatrix(weights, this.#inputSize, this.#hiddenSize)) this.#weightsK = weights;
  }
  setWeightsV(weights) {
    if (this.#isValidMatrix(weights, this.#inputSize, this.#hiddenSize)) this.#weightsV = weights;
  }
  setWeightsOut(weights) {
    if (this.#isValidMatrix(weights, this.#hiddenSize, this.#outputSize)) this.#weightsOut = weights;
  }
  setBiasQ(bias) {
    if (this.#isValidArray(bias, this.#hiddenSize)) this.#biasQ = bias;
  }
  setBiasK(bias) {
    if (this.#isValidArray(bias, this.#hiddenSize)) this.#biasK = bias;
  }
  setBiasV(bias) {
    if (this.#isValidArray(bias, this.#hiddenSize)) this.#biasV = bias;
  }
  setBiasOut(bias) {
    if (this.#isValidArray(bias, this.#outputSize)) this.#biasOut = bias;
  }

  #isValidMatrix(matrix, rows, cols) {
    return Array.isArray(matrix) &&
           matrix.length === rows &&
           matrix.every(row => Array.isArray(row) && row.length === cols && row.every(isValidNumber));
  }

  #isValidArray(arr, len) {
    return Array.isArray(arr) && arr.length === len && arr.every(isValidNumber);
  }

  forward(inputs) {
    if (inputs.length !== this.#inputSize || !inputs.every(isValidNumber)) return [0];
    const query = Array(this.#hiddenSize).fill(0);
    const key = Array(this.#hiddenSize).fill(0);
    const value = Array(this.#hiddenSize).fill(0);
    for (let i = 0; i < this.#inputSize; i++) {
      for (let j = 0; j < this.#hiddenSize; j++) {
        query[j] += inputs[i] * this.#weightsQ[i][j];
        key[j] += inputs[i] * this.#weightsK[i][j];
        value[j] += inputs[i] * this.#weightsV[i][j];
      }
    }
    const q = query.map((x, j) => x + this.#biasQ[j]);
    const k = key.map((x, j) => x + this.#biasK[j]);
    const v = value.map((x, j) => x + this.#biasV[j]);
    const headSize = this.#hiddenSize / this.#heads;
    const scores = Array(this.#heads).fill().map(() => Array(this.#heads).fill(0));
    for (let i = 0; i < this.#heads; i++) {
      for (let j = 0; j < this.#heads; j++) {
        let sum = 0;
        for (let k = 0; k < headSize; k++) {
          sum += q[i * headSize + k] * k[j * headSize + k];
        }
        scores[i][j] = isValidNumber(sum) ? sum / Math.sqrt(headSize) : 0;
      }
    }
    const attention = scores.map(row => softmax(row.map(v => isValidNumber(v) ? v : 0)));
    const attended = Array(this.#hiddenSize).fill(0);
    for (let i = 0; i < this.#heads; i++) {
      for (let j = 0; j < this.#heads; j++) {
        for (let k = 0; k < headSize; k++) {
          attended[i * headSize + k] += attention[i][j] * v[j * headSize + k];
        }
      }
    }
    const output = Array(this.#outputSize).fill(0);
    for (let i = 0; i < this.#hiddenSize; i++) {
      for (let j = 0; j < this.#outputSize; j++) {
        output[j] += attended[i] * this.#weightsOut[i][j];
      }
    }
    return output.map((x, j) => sigmoid(isValidNumber(x + this.#biasOut[j]) ? x + this.#biasOut[j] : 0));
  }

  train(inputs, target) {
    if (inputs.length !== this.#inputSize || !inputs.every(isValidNumber) || !isValidNumber(target)) return;
    const output = this.forward(inputs);
    const error = target - output[0];
    const delta = Math.min(Math.max(error * output[0] * (1 - output[0]), -1), 1);
    const hiddenErrors = Array(this.#hiddenSize).fill(0);
    for (let i = 0; i < this.#hiddenSize; i++) {
      hiddenErrors[i] = delta * this.#weightsOut[i][0];
    }
    for (let i = 0; i < this.#hiddenSize; i++) {
      this.#weightsOut[i][0] += this.#learningRate * delta * hiddenErrors[i];
      this.#biasOut[0] += this.#learningRate * delta;
    }
    for (let i = 0; i < this.#inputSize; i++) {
      for (let j = 0; j < this.#hiddenSize; j++) {
        this.#weightsQ[i][j] += this.#learningRate * hiddenErrors[j] * inputs[i];
        this.#weightsK[i][j] += this.#learningRate * hiddenErrors[j] * inputs[i];
        this.#weightsV[i][j] += this.#learningRate * hiddenErrors[j] * inputs[i];
      }
    }
  }
}

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
    const volumeLast100 = volume.slice(-100);
    const volumeMean = volumeLast100.reduce((sum, v) => sum + (isValidNumber(v) ? v : 0), 0) / volumeLast100.length || 1;
    const volumeStd = Math.sqrt(
      volumeLast100.reduce((sum, v) => sum + (isValidNumber(v) ? (v - volumeMean) ** 2 : 0), 0) / volumeLast100.length
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
    indicators.rsiMin = Math.min(...indicators.rsi.slice(-50).filter(isValidNumber)) || 0;
    indicators.rsiMax = Math.max(...indicators.rsi.slice(-50).filter(isValidNumber)) || 100;
    indicators.macdMin = Math.min(...indicators.macd.map(m => m.MACD - m.signal).filter(isValidNumber)) || -1;
    indicators.macdMax = Math.max(...indicators.macd.map(m => m.MACD - m.signal).filter(isValidNumber)) || 1;
    indicators.atrMin = Math.min(...indicators.atr.slice(-50).filter(isValidNumber)) || 0;
    indicators.atrMax = Math.max(...indicators.atr.slice(-50).filter(isValidNumber)) || 1;
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

class NeuralSignalEngine {
  #transformer = new Transformer();
  #indicators = new IndicatorProcessor();
  #db;
  #config = {
    minMultiplier: 1,
    maxMultiplier: 2.5,
    baseConfidenceThreshold: 60,
    atrFactor: 10,
    stopFactor: 2.5,
    learningRate: 0.25
  };

  constructor() {
    fs.mkdirSync(directoryPath, { recursive: true });
    this.#db = new Database(path.join(directoryPath, 'neural_engine.db'), { fileMustExist: false });
    this.#initDatabase();
    this.#loadState();
  }

  #initDatabase() {
    this.#db.exec(`
      CREATE TABLE IF NOT EXISTS qtable (
        state_key TEXT PRIMARY KEY,
        buy REAL NOT NULL,
        hold REAL NOT NULL
      );
      CREATE TABLE IF NOT EXISTS patterns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        bucket_key TEXT NOT NULL,
        features TEXT NOT NULL,
        score REAL NOT NULL,
        UNIQUE(bucket_key, features)
      );
      CREATE TABLE IF NOT EXISTS open_trades (
        timestamp TEXT PRIMARY KEY,
        sellPrice REAL NOT NULL,
        stopLoss REAL NOT NULL,
        entryPrice REAL NOT NULL,
        confidence REAL NOT NULL,
        candlesHeld INTEGER NOT NULL,
        strategy TEXT NOT NULL,
        patternScore REAL NOT NULL,
        features TEXT NOT NULL,
        stateKey TEXT NOT NULL,
        dynamicThreshold REAL NOT NULL
      );
      CREATE TABLE IF NOT EXISTS candles (
        timestamp TEXT PRIMARY KEY,
        open REAL NOT NULL,
        high REAL NOT NULL,
        low REAL NOT NULL,
        close REAL NOT NULL,
        volume REAL NOT NULL
      );
      CREATE INDEX IF NOT EXISTS idx_bucket_key ON patterns(bucket_key);
      CREATE INDEX IF NOT EXISTS idx_open_trades_sellPrice ON open_trades(sellPrice);
      CREATE INDEX IF NOT EXISTS idx_open_trades_stopLoss ON open_trades(stopLoss);
      CREATE INDEX IF NOT EXISTS idx_candles_timestamp ON candles(timestamp);
    `);
  }

  #saveCompressedFile(filePath, data) {
    try {
      const jsonString = JSON.stringify(data);
      const compressed = zlib.brotliCompressSync(jsonString, {
        params: { [zlib.constants.BROTLI_PARAM_QUALITY]: zlib.constants.BROTLI_MAX_QUALITY }
      });
      fs.writeFileSync(filePath, compressed);
    } catch {}
  }

  #loadCompressedFile(filePath) {
    try {
      const compressed = fs.readFileSync(filePath);
      const decompressed = zlib.brotliDecompressSync(compressed);
      return JSON.parse(decompressed.toString('utf8'));
    } catch {
      return null;
    }
  }

  #loadState() {
    this.#loadNeuralState();
  }

  #saveState() {
    try {
      this.#saveNeuralState();
    } catch {}
  }

  #saveNeuralState() {
    const state = {
      transformer: {
        weightsQ: this.#transformer.getWeightsQ(),
        weightsK: this.#transformer.getWeightsK(),
        weightsV: this.#transformer.getWeightsV(),
        weightsOut: this.#transformer.getWeightsOut(),
        biasQ: this.#transformer.getBiasQ(),
        biasK: this.#transformer.getBiasK(),
        biasV: this.#transformer.getBiasV(),
        biasOut: this.#transformer.getBiasOut()
      }
    };
    this.#saveCompressedFile(path.join(directoryPath, 'neural_state.json'), state);
  }

  #loadNeuralState() {
    const state = this.#loadCompressedFile(path.join(directoryPath, 'neural_state.json'));
    if (state) {
      this.#transformer.setWeightsQ(state.transformer?.weightsQ);
      this.#transformer.setWeightsK(state.transformer?.weightsK);
      this.#transformer.setWeightsV(state.transformer?.weightsV);
      this.#transformer.setWeightsOut(state.transformer?.weightsOut);
      this.#transformer.setBiasQ(state.transformer?.biasQ);
      this.#transformer.setBiasK(state.transformer?.biasK);
      this.#transformer.setBiasV(state.transformer?.biasV);
      this.#transformer.setBiasOut(state.transformer?.biasOut);
    }
  }

  #extractFeatures(data) {
    const normalize = (value, min, max) => {
      if (!isValidNumber(value) || !isValidNumber(min) || !isValidNumber(max) || max === min) return 0;
      return Math.min(1, Math.max(0, (value - min) / (max - min)));
    };
    return [
      normalize(data.rsi[data.rsi.length - 1], data.rsiMin, data.rsiMax),
      normalize(data.macd[data.macd.length - 1].MACD - data.macd[data.macd.length - 1].signal, data.macdMin, data.macdMax),
      normalize(data.atr[data.atr.length - 1], data.atrMin, data.atrMax),
      Math.min(1, Math.max(-1, data.volumeZScore / 3)),
      data.isTrending,
      data.isRanging
    ];
  }

  #scorePattern(features) {
    const key = this.#generateFeatureKey(features);
    const stmt = this.#db.prepare(`SELECT score, features FROM patterns WHERE bucket_key = ?`);
    const patterns = stmt.all(key);
    if (!patterns || patterns.length === 0) return 0;
    let totalScore = 0, matchCount = 0;
    for (const pattern of patterns) {
      const patternFeatures = JSON.parse(pattern.features);
      if (features.every((f, i) => isValidNumber(f) && isValidNumber(patternFeatures[i]) && Math.abs(f - patternFeatures[i]) < 0.1)) {
        totalScore += pattern.score;
        matchCount++;
      }
    }
    return matchCount > 0 ? totalScore / matchCount : 0;
  }

  #computeDynamicThreshold(data, confidence, baseThreshold = this.#config.baseConfidenceThreshold) {
    const normalize = (value, min, max) => {
      if (!isValidNumber(value) || !isValidNumber(min) || !isValidNumber(max) || max === min) return 0.5;
      return Math.min(1, Math.max(0, (value - min) / (max - min)));
    };
    const atrNorm = normalize(data.atr[data.atr.length - 1], data.atrMin, data.atrMax);
    const rsiNorm = normalize(data.rsi[data.rsi.length - 1], data.rsiMin, data.rsiMax);
    const volumeNorm = isValidNumber(data.volumeZScore) ? Math.abs(data.volumeZScore) / 3 : 0;
    const volatilityScore = (atrNorm + volumeNorm + rsiNorm * 0.5) / 2.5;
    const marketCondition = data.isTrending ? 0.8 : data.isRanging ? 1.2 : 1.0;
    let dynamicThreshold = baseThreshold * volatilityScore * marketCondition;
    dynamicThreshold = Math.max(40, Math.min(80, isValidNumber(dynamicThreshold) ? dynamicThreshold : 60));
    if (!isValidNumber(confidence)) return parseFloat(dynamicThreshold.toFixed(3));
    const confidenceProximity = Math.abs(confidence - dynamicThreshold) / 100;
    return parseFloat((dynamicThreshold * (1 - 0.1 * confidenceProximity)).toFixed(3));
  }

  #generateFeatureKey(features) {
    if (!Array.isArray(features) || features.length !== 6) return 'default';
    const quantized = features.map(f => isValidNumber(f) ? Math.round(f * 1000000) / 1000000 : 0);
    return quantized.join('|');
  }

  #updateOpenTrades(candles) {
    if (!Array.isArray(candles) || candles.length === 0) return;

    const insertCandleStmt = this.#db.prepare(`
      INSERT OR IGNORE INTO candles (timestamp, open, high, low, close, volume)
      VALUES (?, ?, ?, ?, ?, ?)
    `);
    const newCandles = candles.filter(c => 
      isValidNumber(c.timestamp) &&
      isValidNumber(c.open) &&
      isValidNumber(c.high) &&
      isValidNumber(c.low) &&
      isValidNumber(c.close) &&
      isValidNumber(c.volume) &&
      c.volume >= 0
    );

    const transaction = this.#db.transaction(() => {
      for (const candle of newCandles) {
        insertCandleStmt.run(
          candle.timestamp,
          candle.open,
          candle.high,
          candle.low,
          candle.close,
          candle.volume
        );
      }
    });
    transaction();

    // Cleanup: Keep only the 1000 most recent candles
    const cleanupStmt = this.#db.prepare(`DELETE FROM candles WHERE timestamp NOT IN (SELECT timestamp FROM candles ORDER BY timestamp DESC LIMIT 1000)`);
    cleanupStmt.run();

    if (newCandles.length === 0) return;

    const minLow = Math.min(...newCandles.map(c => isValidNumber(c.low) ? c.low : Infinity));
    const maxHigh = Math.max(...newCandles.map(c => isValidNumber(c.high) ? c.high : -Infinity));
    if (!isValidNumber(minLow) || !isValidNumber(maxHigh)) return;

    const tradesStmt = this.#db.prepare(`
      SELECT timestamp, sellPrice, stopLoss, entryPrice, confidence, candlesHeld, strategy, patternScore, features, stateKey, dynamicThreshold
      FROM open_trades
      WHERE sellPrice BETWEEN ? AND ? OR stopLoss BETWEEN ? AND ?
    `);
    const trades = tradesStmt.all(minLow, maxHigh, minLow, maxHigh);

    const closedTrades = [];
    const keysToDelete = new Set();

    for (const trade of trades) {
      const features = JSON.parse(trade.features);
      for (const candle of newCandles) {
        if (!candle || !isValidNumber(candle.high) || !isValidNumber(candle.low)) continue;

        if (candle.high >= trade.sellPrice) {
          const outcome = (trade.sellPrice - trade.entryPrice) / trade.entryPrice;
          closedTrades.push({
            timestamp: Date.now(),
            entryPrice: trade.entryPrice,
            exitPrice: trade.sellPrice,
            confidence: trade.confidence,
            outcome: Math.min(Math.max(outcome, -1), 1),
            reward: outcome * (trade.confidence / 100),
            strategy: trade.strategy,
            patternScore: trade.patternScore,
            candlesHeld: trade.candlesHeld + newCandles.length,
            features,
            stateKey: trade.stateKey,
            dynamicThreshold: trade.dynamicThreshold
          });
          keysToDelete.add(trade.timestamp);
          break;
        } else if (candle.low <= trade.stopLoss) {
          const outcome = (trade.stopLoss - trade.entryPrice) / trade.entryPrice;
          closedTrades.push({
            timestamp: Date.now(),
            entryPrice: trade.entryPrice,
            exitPrice: trade.stopLoss,
            confidence: trade.confidence,
            outcome: Math.min(Math.max(outcome, -1), 1),
            reward: outcome * (trade.confidence / 100),
            strategy: trade.strategy,
            patternScore: trade.patternScore,
            candlesHeld: trade.candlesHeld + newCandles.length,
            features,
            stateKey: trade.stateKey,
            dynamicThreshold: trade.dynamicThreshold
          });
          keysToDelete.add(trade.timestamp);
          break;
        }
      }
    }

    if (closedTrades.length > 0) {
      const transaction = this.#db.transaction(() => {
        const insertPatternStmt = this.#db.prepare(`INSERT OR REPLACE INTO patterns (bucket_key, features, score) VALUES (?, ?, ?)`);
        const updateQTableStmt = this.#db.prepare(`INSERT OR REPLACE INTO qtable (state_key, buy, hold) VALUES (?, ?, COALESCE((SELECT hold FROM qtable WHERE state_key = ?), 0))`);
        const deleteTradeStmt = this.#db.prepare(`DELETE FROM open_trades WHERE timestamp = ?`);

        for (const trade of closedTrades) {
          const key = this.#generateFeatureKey(trade.features);
          insertPatternStmt.run(key, JSON.stringify(trade.features), trade.reward);

          const existingQ = this.#db.prepare(`SELECT buy, hold FROM qtable WHERE state_key = ?`).get(trade.stateKey) || { buy: 0, hold: 0 };
          updateQTableStmt.run(trade.stateKey, existingQ.buy + this.#config.learningRate * (trade.reward - existingQ.buy), trade.stateKey);

          this.#transformer.train(trade.features, trade.outcome > 0 ? 1 : 0);
        }

        for (const key of keysToDelete) {
          deleteTradeStmt.run(key);
        }
      });
      transaction();
    }
  }

  getSignalsAndHealth(candles, saveState = true) {
    if (!Array.isArray(candles) || candles.length === 0) {
      return { error: 'Invalid candle array type or length' };
    }

    this.#updateOpenTrades(candles);

    const timestampStmt = this.#db.prepare(`SELECT timestamp FROM candles WHERE timestamp IN (${candles.map(() => '?').join(',')})`);
    const existingTimestamps = new Set(timestampStmt.all(...candles.map(c => c.timestamp)).map(row => row.timestamp));
    const newCandles = candles.filter(c =>
      isValidNumber(c.timestamp) &&
      isValidNumber(c.open) &&
      isValidNumber(c.high) &&
      isValidNumber(c.low) &&
      isValidNumber(c.close) &&
      isValidNumber(c.volume) &&
      c.volume >= 0 &&
      !existingTimestamps.has(c.timestamp)
    );

    const fetchCandlesStmt = this.#db.prepare(`SELECT * FROM candles ORDER BY timestamp DESC LIMIT 100`);
    const recentCandles = newCandles.length > 0 ? [...newCandles, ...fetchCandlesStmt.all().filter(c => !newCandles.some(nc => nc.timestamp === c.timestamp))] : fetchCandlesStmt.all();
    if (recentCandles.length === 0) {
      return { error: 'Invalid candle array type or length' };
    }

    const indicators = this.#indicators.compute(recentCandles.slice(-100));
    if (indicators.error) {
      return { error: 'Indicators error' };
    }

    const features = this.#extractFeatures(indicators);
    const patternScore = this.#scorePattern(features);
    const confidence = this.#transformer.forward(features)[0] * 100 * (1 + patternScore);
    const dynamicThreshold = this.#computeDynamicThreshold(indicators, confidence, this.#config.baseConfidenceThreshold);
    const multiplier = this.#config.minMultiplier + (this.#config.maxMultiplier - this.#config.minMultiplier) * Math.max(0, (confidence - dynamicThreshold) / (100 - dynamicThreshold));
    let sellPrice = indicators.lastClose + this.#config.atrFactor * (indicators.atr[indicators.atr.length - 1] || 0);
    let stopLoss = indicators.lastClose - this.#config.stopFactor * (indicators.atr[indicators.atr.length - 1] || 0);
    sellPrice = isValidNumber(sellPrice) && sellPrice > indicators.lastClose && sellPrice <= indicators.lastClose * 1.3 ? sellPrice : indicators.lastClose * 1.001;
    stopLoss = isValidNumber(stopLoss) && stopLoss < indicators.lastClose && stopLoss > 0 && stopLoss >= indicators.lastClose * 0.7 ? stopLoss : indicators.lastClose * 0.999;

    const fee = 0.0021;
    const adjustedSellPrice = sellPrice * (1 - fee);
    const expectedReward = isValidNumber(adjustedSellPrice) && isValidNumber(indicators.lastClose) && indicators.lastClose !== 0
      ? (adjustedSellPrice - indicators.lastClose) / indicators.lastClose
      : 0;

    const stateKey = this.#generateFeatureKey(features);
    const qTableStmt = this.#db.prepare(`SELECT buy, hold FROM qtable WHERE state_key = ?`);
    let qValues = qTableStmt.get(stateKey);
    if (!qValues) {
      this.#db.prepare(`INSERT OR IGNORE INTO qtable (state_key, buy, hold) VALUES (?, 0, 0)`).run(stateKey);
      qValues = { buy: 0, hold: 0 };
    }

    const action = qValues.buy > qValues.hold ? 'buy' : 'hold';
    const entryPrice = indicators.lastClose;
    const key = Date.now().toString();

    const insertTradeStmt = this.#db.prepare(`
      INSERT INTO open_trades (timestamp, sellPrice, stopLoss, entryPrice, confidence, candlesHeld, strategy, patternScore, features, stateKey, dynamicThreshold)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);
    insertTradeStmt.run(
      key,
      truncateToDecimals(sellPrice, 2),
      truncateToDecimals(stopLoss, 2),
      entryPrice,
      confidence,
      0,
      action,
      patternScore,
      JSON.stringify(features),
      stateKey,
      dynamicThreshold
    );

    if (saveState) {
      this.#saveState();
    }

    const totalPatternsStmt = this.#db.prepare(`SELECT COUNT(*) as count FROM patterns`);
    const totalPatterns = totalPatternsStmt.get().count;

    return {
      currentConfidence: isValidNumber(confidence) ? truncateToDecimals(confidence, 3) : 0,
      suggestedConfidence: isValidNumber(dynamicThreshold) ? truncateToDecimals(dynamicThreshold, 3) : 0,
      multiplier: isValidNumber(multiplier) ? truncateToDecimals(multiplier, 3) : this.#config.minMultiplier,
      sellPrice: isValidNumber(sellPrice) ? truncateToDecimals(sellPrice, 2) : 0,
      stopLoss: isValidNumber(stopLoss) ? truncateToDecimals(stopLoss, 2) : 0,
      expectedReward: truncateToDecimals(expectedReward, 8),
      suggestedAction: action,
      activePatterns: totalPatterns
    };
  }

  dumpState() {
    this.#saveState();
  }
}

export default NeuralSignalEngine;