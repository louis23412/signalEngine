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

const relu = (x) => isValidNumber(x) ? Math.max(0, x) : 0;

const softmax = (arr) => {
  if (!arr.every(isValidNumber)) return arr.map(() => 1 / arr.length);
  const max = Math.max(...arr);
  const exp = arr.map(x => Math.exp(x - max));
  const sum = exp.reduce((a, b) => a + b, 0) || 1;
  return exp.map(x => x / sum);
};

class AutoEncoder {
  #inputSize = 6;
  #hiddenSize = 3;
  #weightsEncode;
  #weightsDecode;
  #biasEncode;
  #biasDecode;
  #learningRate = 0.01;

  constructor() {
    const xavierInit = (rows, cols) => Array(rows).fill().map(() => Array(cols).fill().map(() => (Math.random() - 0.5) * Math.sqrt(2 / (rows + cols))));
    this.#weightsEncode = xavierInit(this.#inputSize, this.#hiddenSize);
    this.#weightsDecode = xavierInit(this.#hiddenSize, this.#inputSize);
    this.#biasEncode = Array(this.#hiddenSize).fill(0);
    this.#biasDecode = Array(this.#inputSize).fill(0);
  }

  getWeightsEncode() { return this.#weightsEncode; }
  getWeightsDecode() { return this.#weightsDecode; }
  getBiasEncode() { return this.#biasEncode; }
  getBiasDecode() { return this.#biasDecode; }

  setWeightsEncode(weights) {
    if (this.#isValidMatrix(weights, this.#inputSize, this.#hiddenSize)) this.#weightsEncode = weights;
  }
  setWeightsDecode(weights) {
    if (this.#isValidMatrix(weights, this.#hiddenSize, this.#inputSize)) this.#weightsDecode = weights;
  }
  setBiasEncode(bias) {
    if (this.#isValidArray(bias, this.#hiddenSize)) this.#biasEncode = bias;
  }
  setBiasDecode(bias) {
    if (this.#isValidArray(bias, this.#inputSize)) this.#biasDecode = bias;
  }

  #isValidMatrix(matrix, rows, cols) {
    return Array.isArray(matrix) &&
           matrix.length === rows &&
           matrix.every(row => Array.isArray(row) && row.length === cols && row.every(isValidNumber));
  }

  #isValidArray(arr, len) {
    return Array.isArray(arr) && arr.length === len && arr.every(isValidNumber);
  }

  encode(inputs) {
    if (inputs.length !== this.#inputSize || !inputs.every(isValidNumber)) return Array(this.#hiddenSize).fill(0);
    const hidden = Array(this.#hiddenSize).fill(0);
    for (let i = 0; i < this.#inputSize; i++) {
      for (let j = 0; j < this.#hiddenSize; j++) {
        hidden[j] += inputs[i] * this.#weightsEncode[i][j];
      }
    }
    return hidden.map((x, j) => relu(x + this.#biasEncode[j]));
  }

  decode(hidden) {
    if (hidden.length !== this.#hiddenSize || !hidden.every(isValidNumber)) return Array(this.#inputSize).fill(0);
    const output = Array(this.#inputSize).fill(0);
    for (let i = 0; i < this.#hiddenSize; i++) {
      for (let j = 0; j < this.#inputSize; j++) {
        output[j] += hidden[i] * this.#weightsDecode[i][j];
      }
    }
    return output.map((x, j) => x + this.#biasDecode[j]);
  }

  train(inputs) {
    if (inputs.length !== this.#inputSize || !inputs.every(isValidNumber)) return;
    const hidden = this.encode(inputs);
    const output = this.decode(hidden);
    const errors = inputs.map((x, i) => x - output[i]);
    const hiddenErrors = Array(this.#hiddenSize).fill(0);
    for (let i = 0; i < this.#hiddenSize; i++) {
      for (let j = 0; j < this.#inputSize; j++) {
        hiddenErrors[i] += errors[j] * this.#weightsDecode[i][j];
      }
    }
    for (let i = 0; i < this.#hiddenSize; i++) {
      for (let j = 0; j < this.#inputSize; j++) {
        this.#weightsDecode[i][j] += this.#learningRate * errors[j] * hidden[i];
      }
    }
    for (let j = 0; j < this.#inputSize; j++) {
      this.#biasDecode[j] += this.#learningRate * errors[j];
    }
    for (let i = 0; i < this.#inputSize; i++) {
      for (let j = 0; j < this.#hiddenSize; j++) {
        this.#weightsEncode[i][j] += this.#learningRate * hiddenErrors[j] * inputs[i];
      }
    }
    for (let j = 0; j < this.#hiddenSize; j++) {
      this.#biasEncode[j] += this.#learningRate * hiddenErrors[j];
    }
  }
}

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
  #autoencoder = new AutoEncoder();
  #indicators = new IndicatorProcessor();
  #db;
  #candles = [];
  #trades = [];
  #openTrades = new Map();
  #candlesTimestamps = new Set();
  #tradeMetadata = new Map();
  #tradeBuckets = { sell: {}, stop: {} };
  #patternScoreCache = new Map();
  #bucketPriceRange = 10;
  #maxCandles = 1000;
  #maxTrades = 10000000;
  #state = {
    winRate: 0.5,
    tradeCount: 0,
    lastUpdate: null,
    avgReward: 0,
    signalReliability: 0.5
  };
  #lifetimeState = {
    totalTrades: 0,
    totalWins: 0,
    totalAccuracy: 50
  };
  #performanceBaseline = {
    avgWinRate: 0.5,
    avgReward: 0,
    avgConfidence: 50
  };
  #config = {
    minMultiplier: 1,
    maxMultiplier: 2.5,
    baseConfidenceThreshold: 60,
    atrFactor: 10,
    stopFactor: 2.5,
    learningRate: 0.25
  };
  #longTermBuffer = [];
  #stateChanged = {
    lifetimeState: false,
    performanceBaseline: false,
    neuralState: false,
    learningState: false,
    openTrades: false,
    candleEmbeddings: false
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
      CREATE INDEX IF NOT EXISTS idx_bucket_key ON patterns(bucket_key);
    `);
  }

  #saveCompressedFile(filePath, data) {
    try {
      const jsonString = JSON.stringify(data);
      const compressed = zlib.brotliCompressSync(jsonString, {
        params: {
          [zlib.constants.BROTLI_PARAM_QUALITY]: zlib.constants.BROTLI_MAX_QUALITY
        }
      });
      fs.writeFileSync(filePath, compressed);
    } catch (error) {
    }
  }

  #loadCompressedFile(filePath) {
    try {
      const compressed = fs.readFileSync(filePath);
      const decompressed = zlib.brotliDecompressSync(compressed);
      return JSON.parse(decompressed.toString('utf8'));
    } catch (error) {
      return null;
    }
  }

  #loadState() {
    const state = this.#loadCompressedFile(path.join(directoryPath, 'lifetime_state.json'));
    if (
      state &&
      isValidNumber(state.totalTrades) &&
      state.totalTrades >= 0 &&
      isValidNumber(state.totalWins) &&
      state.totalWins >= 0 &&
      state.totalWins <= state.totalTrades &&
      isValidNumber(state.totalAccuracy) &&
      state.totalAccuracy >= 0 &&
      state.totalAccuracy <= 100
    ) {
      this.#lifetimeState.totalTrades = state.totalTrades;
      this.#lifetimeState.totalWins = state.totalWins;
      this.#lifetimeState.totalAccuracy = state.totalAccuracy;
    }

    const baseline = this.#loadCompressedFile(path.join(directoryPath, 'performance_summary.json'));
    if (
      baseline &&
      isValidNumber(baseline.avgWinRate) &&
      isValidNumber(baseline.avgReward) &&
      isValidNumber(baseline.avgConfidence)
    ) {
      this.#performanceBaseline.avgWinRate = baseline.avgWinRate;
      this.#performanceBaseline.avgReward = baseline.avgReward;
      this.#performanceBaseline.avgConfidence = baseline.avgConfidence;
    }

    this.#loadNeuralState();
    this.#loadOpenTrades();
    this.#loadCandleEmbeddings();
  }

  #saveState(force = false) {
    try {
      if (force || this.#stateChanged.lifetimeState) {
        this.#saveCompressedFile(path.join(directoryPath, 'lifetime_state.json'), this.#lifetimeState);
      }
      if (force || this.#stateChanged.performanceBaseline) {
        this.#saveCompressedFile(path.join(directoryPath, 'performance_summary.json'), this.#performanceBaseline);
      }
      if (force || this.#stateChanged.neuralState) {
        this.#saveNeuralState();
      }
      if (force || this.#stateChanged.openTrades) {
        this.#saveOpenTrades();
      }
      if (force || this.#stateChanged.candleEmbeddings) {
        this.#saveCandleEmbeddings();
      }
    } catch {}
  }

  #saveNeuralState() {
    const state = {
      autoEncoder: {
        weightsEncode: this.#autoencoder.getWeightsEncode(),
        weightsDecode: this.#autoencoder.getWeightsDecode(),
        biasEncode: this.#autoencoder.getBiasEncode(),
        biasDecode: this.#autoencoder.getBiasDecode()
      },
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
      this.#autoencoder.setWeightsEncode(state.autoEncoder?.weightsEncode);
      this.#autoencoder.setWeightsDecode(state.autoEncoder?.weightsDecode);
      this.#autoencoder.setBiasEncode(state.autoEncoder?.biasEncode);
      this.#autoencoder.setBiasDecode(state.autoEncoder?.biasDecode);
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

  #saveOpenTrades() {
    const tradesArray = Array.from(this.#openTrades.values());
    this.#saveCompressedFile(path.join(directoryPath, 'open_trades.json'), tradesArray);
  }

  #loadOpenTrades() {
    const openTrades = this.#loadCompressedFile(path.join(directoryPath, 'open_trades.json'));
    if (Array.isArray(openTrades)) {
      this.#openTrades = new Map();
      openTrades
        .filter(trade => 
          isValidNumber(trade.timestamp) &&
          isValidNumber(trade.entryPrice) &&
          isValidNumber(trade.sellPrice) &&
          isValidNumber(trade.stopLoss) &&
          isValidNumber(trade.confidence) &&
          typeof trade.strategy === 'string' &&
          isValidNumber(trade.patternScore) &&
          Array.isArray(trade.features) && trade.features.length === 6 && trade.features.every(isValidNumber) &&
          typeof trade.stateKey === 'string' &&
          isValidNumber(trade.dynamicThreshold) &&
          isValidNumber(trade.candlesHeld) && trade.candlesHeld >= 0
        )
        .forEach(trade => {
          this.#openTrades.set(trade.timestamp.toString(), trade);
          const stmt = this.#db.prepare('INSERT OR IGNORE INTO qtable (state_key, buy, hold) VALUES (?, 0, 0)');
          stmt.run(trade.stateKey);
        });
    }
  }

  #saveCandleEmbeddings() {
    const state = { longTermBuffer: this.#longTermBuffer.slice(-this.#maxCandles / 2) };
    this.#saveCompressedFile(path.join(directoryPath, 'candle_embeddings.json'), state);
  }

  #loadCandleEmbeddings() {
    const state = this.#loadCompressedFile(path.join(directoryPath, 'candle_embeddings.json'));
    if (state && Array.isArray(state.longTermBuffer)) {
      this.#longTermBuffer = state.longTermBuffer
        .filter(embedding => Array.isArray(embedding) && embedding.every(isValidNumber))
        .slice(-this.#maxCandles);
    }
  }

  #compressCandles() {
    if (this.#candles.length < 100) return;
    const toCompress = this.#candles.slice(0, -50);
    const features = toCompress.map(c => [
      c.close,
      c.high - c.low,
      c.volume,
      c.close > c.open ? 1 : 0,
      isValidNumber(c.close) && isValidNumber(c.open) ? Math.abs(c.close - c.open) / c.close : 0,
      c.timestamp
    ]);
    const embeddings = features.map(f => this.#autoencoder.encode(f));
    this.#longTermBuffer.push(...embeddings.slice(-this.#maxCandles));
    this.#longTermBuffer = this.#longTermBuffer.slice(-this.#maxCandles);
    this.#candles = this.#candles.slice(-50);
    for (const candle of toCompress) {
      this.#candlesTimestamps.delete(candle.timestamp);
    }
    this.#stateChanged.candleEmbeddings = true;
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
    if (this.#patternScoreCache.has(key)) {
      return this.#patternScoreCache.get(key);
    }
    const stmt = this.#db.prepare(`
      SELECT score, features FROM patterns WHERE bucket_key = ?
    `);
    const patterns = stmt.all(key);
    if (!patterns || patterns.length === 0) return 0;
    let totalScore = 0;
    let matchCount = 0;
    for (const pattern of patterns) {
      const patternFeatures = JSON.parse(pattern.features);
      if (features.every((f, i) => isValidNumber(f) && isValidNumber(patternFeatures[i]) && Math.abs(f - patternFeatures[i]) < 0.1)) {
        totalScore += pattern.score;
        matchCount++;
      }
    }
    const score = matchCount > 0 ? totalScore / matchCount : 0;
    this.#patternScoreCache.set(key, score);
    if (this.#patternScoreCache.size > 10000) {
      this.#patternScoreCache.delete(this.#patternScoreCache.keys().next().value);
    }
    return score;
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
    const quantized = features.map(f => isValidNumber(f) ? Math.round(f * 1000000000000000) / 1000000000000000 : 0);
    return quantized.join('|');
  }

  #updateOpenTrades(candles) {
    if (!Array.isArray(candles) || candles.length === 0) return;

    const newCandles = candles.filter(c => 
      isValidNumber(c.timestamp) &&
      isValidNumber(c.open) &&
      isValidNumber(c.high) &&
      isValidNumber(c.low) &&
      isValidNumber(c.close) &&
      isValidNumber(c.volume) &&
      c.volume >= 0 &&
      !this.#candlesTimestamps.has(c.timestamp)
    );

    if (newCandles.length === 0) return;

    const minLow = Math.min(...newCandles.map(c => isValidNumber(c.low) ? c.low : Infinity));
    const maxHigh = Math.max(...newCandles.map(c => isValidNumber(c.high) ? c.high : -Infinity));
    if (!isValidNumber(minLow) || !isValidNumber(maxHigh)) return;

    const closedTrades = [];
    const keysToDelete = new Set();

    const sellBucketStart = Math.floor(minLow / this.#bucketPriceRange);
    const stopBucketEnd = Math.ceil(maxHigh / this.#bucketPriceRange);
    const tradeKeys = new Set();
    for (let i = sellBucketStart; i <= stopBucketEnd; i++) {
      if (this.#tradeBuckets.sell[i]) this.#tradeBuckets.sell[i].forEach(key => tradeKeys.add(key));
      if (this.#tradeBuckets.stop[i]) this.#tradeBuckets.stop[i].forEach(key => tradeKeys.add(key));
    }

    for (const key of tradeKeys) {
      const trade = this.#openTrades.get(key);
      if (!trade) continue;
      const { sellPrice, stopLoss, entryPrice, confidence, candlesHeld } = trade;
      const metadata = this.#tradeMetadata.get(key);

      for (const candle of newCandles) {
        if (!candle || !isValidNumber(candle.high) || !isValidNumber(candle.low)) continue;

        if (candle.high >= sellPrice) {
          const outcome = (sellPrice - entryPrice) / entryPrice;
          closedTrades.push({
            timestamp: Date.now(),
            entryPrice,
            exitPrice: sellPrice,
            confidence,
            outcome: Math.min(Math.max(outcome, -1), 1),
            reward: outcome * (confidence / 100),
            strategy: metadata.strategy,
            patternScore: metadata.patternScore,
            candlesHeld: candlesHeld + newCandles.length,
            features: metadata.features,
            stateKey: metadata.stateKey,
            dynamicThreshold: metadata.dynamicThreshold
          });
          keysToDelete.add(key);
          break;
        } else if (candle.low <= stopLoss) {
          const outcome = (stopLoss - entryPrice) / entryPrice;
          closedTrades.push({
            timestamp: Date.now(),
            entryPrice,
            exitPrice: stopLoss,
            confidence,
            outcome: Math.min(Math.max(outcome, -1), 1),
            reward: outcome * (confidence / 100),
            strategy: metadata.strategy,
            patternScore: metadata.patternScore,
            candlesHeld: candlesHeld + newCandles.length,
            features: metadata.features,
            stateKey: metadata.stateKey,
            dynamicThreshold: metadata.dynamicThreshold
          });
          keysToDelete.add(key);
          break;
        }
      }
    }

    for (const key of keysToDelete) {
      const trade = this.#openTrades.get(key);
      if (trade) {
        const sellBucket = Math.floor(trade.sellPrice / this.#bucketPriceRange);
        const stopBucket = Math.floor(trade.stopLoss / this.#bucketPriceRange);
        if (this.#tradeBuckets.sell[sellBucket]) this.#tradeBuckets.sell[sellBucket].delete(key);
        if (this.#tradeBuckets.stop[stopBucket]) this.#tradeBuckets.stop[stopBucket].delete(key);
        this.#openTrades.delete(key);
        this.#tradeMetadata.delete(key);
        this.#stateChanged.openTrades = true;
      }
    }

    if (closedTrades.length > 0) {
      this.#trades.push(...closedTrades);
      this.#trades = this.#trades.slice(-this.#maxTrades);

      const insertPatternStmt = this.#db.prepare(`
        INSERT OR REPLACE INTO patterns (bucket_key, features, score)
        VALUES (?, ?, ?)
      `);
      const updateQTableStmt = this.#db.prepare(`
        INSERT OR REPLACE INTO qtable (state_key, buy, hold)
        VALUES (?, ?, COALESCE((SELECT hold FROM qtable WHERE state_key = ?), 0))
      `);

      for (const trade of closedTrades) {
        const pattern = { features: trade.features, score: trade.reward };
        const key = this.#generateFeatureKey(trade.features);
        insertPatternStmt.run(key, JSON.stringify(trade.features), trade.reward);

        const existingQ = this.#db.prepare('SELECT buy, hold FROM qtable WHERE state_key = ?').get(trade.stateKey) || { buy: 0, hold: 0 };
        updateQTableStmt.run(trade.stateKey, existingQ.buy + this.#config.learningRate * (trade.reward - existingQ.buy), trade.stateKey);
        
        this.#transformer.train(trade.features, trade.outcome > 0 ? 1 : 0);
        this.#autoencoder.train(trade.features);
        this.#stateChanged.neuralState = true;
      }

      this.#state.tradeCount += closedTrades.length;
      const wins = closedTrades.filter(t => t.outcome > 0).length;
      const totalReward = closedTrades.reduce((sum, t) => sum + t.reward, 0);
      const reliableSignals = closedTrades.filter(t => t.confidence > t.dynamicThreshold && t.outcome > 0).length;
      this.#state.winRate = (this.#state.winRate * this.#trades.length + wins) / (this.#trades.length + closedTrades.length) || 0.5;
      this.#state.avgReward = (this.#state.avgReward * this.#trades.length + totalReward) / (this.#trades.length + closedTrades.length) || 0;
      this.#state.signalReliability = (this.#state.signalReliability * this.#trades.length + reliableSignals) / (this.#trades.length + closedTrades.length) || 0.5;
      this.#lifetimeState.totalTrades += closedTrades.length;
      this.#lifetimeState.totalWins += wins;
      this.#lifetimeState.totalAccuracy = this.#lifetimeState.totalTrades > 0 ? (this.#lifetimeState.totalWins / this.#lifetimeState.totalTrades) * 100 : 50;
      this.#performanceBaseline.avgWinRate = (this.#performanceBaseline.avgWinRate * 0.9 + this.#state.winRate * 0.1);
      this.#performanceBaseline.avgReward = (this.#performanceBaseline.avgReward * 0.9 + this.#state.avgReward * 0.1);
      this.#performanceBaseline.avgConfidence = (this.#performanceBaseline.avgConfidence * 0.9 + (closedTrades.reduce((sum, t) => sum + t.confidence, 0) / closedTrades.length || 0) * 0.1);
      this.#stateChanged.lifetimeState = true;
      this.#stateChanged.performanceBaseline = true;
    }
  }

  getSignalsAndHealth(candles, saveState = true) {
    if (!Array.isArray(candles) || candles.length === 0) {
      return {
        error: 'Invalid candle array type or length'
      };
    }

    this.#updateOpenTrades(candles);

    const newCandles = candles.filter(c =>
      isValidNumber(c.timestamp) &&
      isValidNumber(c.open) &&
      isValidNumber(c.high) &&
      isValidNumber(c.low) &&
      isValidNumber(c.close) &&
      isValidNumber(c.volume) &&
      c.volume >= 0 &&
      !this.#candlesTimestamps.has(c.timestamp)
    );

    if (newCandles.length === 0 && this.#candles.length === 0) {
      return { error: 'Invalid candle array type or length' };
    }

    this.#candles.push(...newCandles);
    for (const candle of newCandles) {
      this.#candlesTimestamps.add(candle.timestamp);
      this.#stateChanged.candleEmbeddings = true;
    }
    this.#candles = this.#candles.slice(-this.#maxCandles);

    this.#compressCandles();
    const indicators = this.#indicators.compute(this.#candles.slice(-100));
    if (indicators.error) {
      return {
        error: 'Indicators error'
      };
    }

    const features = this.#extractFeatures(indicators);
    const patternScore = this.#scorePattern(features);
    const confidence = this.#transformer.forward(features)[0] * 100 * (1 + patternScore);
    const dynamicThreshold = this.#computeDynamicThreshold(indicators, confidence, this.#state.tradeCount < 100 ? 50 : this.#config.baseConfidenceThreshold);
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
    const qTableStmt = this.#db.prepare('SELECT buy, hold FROM qtable WHERE state_key = ?');
    let qValues = qTableStmt.get(stateKey);
    if (!qValues) {
      this.#db.prepare('INSERT OR IGNORE INTO qtable (state_key, buy, hold) VALUES (?, 0, 0)').run(stateKey);
      qValues = { buy: 0, hold: 0 };
    }

    const action = qValues.buy > qValues.hold ? 'buy' : 'hold';

    const entryPrice = indicators.lastClose;
    const key = Date.now().toString();
    const trade = {
      sellPrice: truncateToDecimals(sellPrice, 2),
      stopLoss: truncateToDecimals(stopLoss, 2),
      entryPrice,
      confidence,
      candlesHeld: 0
    };
    this.#openTrades.set(key, trade);
    this.#tradeMetadata.set(key, {
      strategy: action,
      patternScore,
      features,
      stateKey,
      dynamicThreshold
    });

    const sellBucket = Math.floor(sellPrice / this.#bucketPriceRange);
    const stopBucket = Math.floor(stopLoss / this.#bucketPriceRange);
    this.#tradeBuckets.sell[sellBucket] = this.#tradeBuckets.sell[sellBucket] || new Set();
    this.#tradeBuckets.stop[stopBucket] = this.#tradeBuckets.stop[stopBucket] || new Set();
    this.#tradeBuckets.sell[sellBucket].add(key);
    this.#tradeBuckets.stop[stopBucket].add(key);
    this.#stateChanged.openTrades = true;

    this.#state.lastUpdate = Date.now();

    if (saveState) {
      this.#saveState();
      this.#stateChanged = {
        lifetimeState: false,
        performanceBaseline: false,
        neuralState: false,
        learningState: false,
        openTrades: false,
        candleEmbeddings: false
      };
    }

    const totalPatternsStmt = this.#db.prepare('SELECT COUNT(*) as count FROM patterns');
    const totalPatterns = totalPatternsStmt.get().count;

    return {
      currentConfidence: isValidNumber(confidence) ? truncateToDecimals(confidence, 3) : 0,
      suggestedConfidence: isValidNumber(dynamicThreshold) ? truncateToDecimals(dynamicThreshold, 3) : 0,
      multiplier: isValidNumber(multiplier) ? truncateToDecimals(multiplier, 3) : this.#config.minMultiplier,
      sellPrice: isValidNumber(sellPrice) ? truncateToDecimals(sellPrice, 2) : 0,
      stopLoss: isValidNumber(stopLoss) ? truncateToDecimals(stopLoss, 2) : 0,
      expectedReward: truncateToDecimals(expectedReward, 8),
      suggestedAction: action,
      totalTrades: this.#lifetimeState.totalTrades,
      totalAccuracy: truncateToDecimals(this.#lifetimeState.totalAccuracy, 4),
      performanceWinRate: truncateToDecimals(this.#performanceBaseline.avgWinRate * 100, 4),
      performanceAvgReward: truncateToDecimals(this.#performanceBaseline.avgReward, 8),
      lastUpdate: this.#state.lastUpdate ? new Date(this.#state.lastUpdate).toLocaleString() : 'N/A',
      signalReliability: truncateToDecimals(this.#state.signalReliability, 4),
      activePatterns: totalPatterns
    };
  }

  dumpState() {
    this.#saveState(true);
  }
}

export default NeuralSignalEngine;