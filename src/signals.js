import fs from 'fs';
import path from 'path';

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
  #candles = [];
  #trades = [];
  #openTrades = [];
  #qTable = {};
  #maxCandles = 1000;
  #maxTrades = 1000;
  #patternBuckets = {};
  #bucketSize = 100; // Max patterns per bucket
  #maxBuckets = 20000; // Adjusted for 1M patterns (optional, was 100000)
  #totalPatterns = 0;
  #maxPatterns = 2000000; // Target capacity
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
    learningRate: 0.25,
    explorationRate: 0.25
  };
  #longTermBuffer = [];

  constructor() {
    this.#loadState();
  }

  #loadState() {
    try {
      const state = JSON.parse(fs.readFileSync(path.join(directoryPath, 'lifetime_state.json'), 'utf8'));
      if (
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
    } catch {}
    try {
      const baseline = JSON.parse(fs.readFileSync(path.join(directoryPath, 'performance_summary.json'), 'utf8'));
      if (
        isValidNumber(baseline.avgWinRate) &&
        isValidNumber(baseline.avgReward) &&
        isValidNumber(baseline.avgConfidence)
      ) {
        this.#performanceBaseline.avgWinRate = baseline.avgWinRate;
        this.#performanceBaseline.avgReward = baseline.avgReward;
        this.#performanceBaseline.avgConfidence = baseline.avgConfidence;
      }
    } catch {}
    this.#loadNeuralState();
    this.#loadLearningState();
    this.#loadOpenTrades();
    this.#loadCandleEmbeddings();
  }

  #saveState() {
    try {
      fs.mkdirSync(directoryPath, { recursive: true });
      fs.writeFileSync(path.join(directoryPath, 'lifetime_state.json'), JSON.stringify(this.#lifetimeState), 'utf8');
      fs.writeFileSync(path.join(directoryPath, 'performance_summary.json'), JSON.stringify(this.#performanceBaseline), 'utf8');
      this.#saveNeuralState();
      this.#saveLearningState();
      this.#saveOpenTrades();
      this.#saveCandleEmbeddings();
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
    try {
      fs.writeFileSync(path.join(directoryPath, 'neural_state.json'), JSON.stringify(state), 'utf8');
    } catch {}
  }

  #loadNeuralState() {
    try {
      const state = JSON.parse(fs.readFileSync(path.join(directoryPath, 'neural_state.json'), 'utf8'));
      this.#autoencoder.setWeightsEncode(state.autoEncoder.weightsEncode);
      this.#autoencoder.setWeightsDecode(state.autoEncoder.weightsDecode);
      this.#autoencoder.setBiasEncode(state.autoEncoder.biasEncode);
      this.#autoencoder.setBiasDecode(state.autoEncoder.biasDecode);
      this.#transformer.setWeightsQ(state.transformer.weightsQ);
      this.#transformer.setWeightsK(state.transformer.weightsK);
      this.#transformer.setWeightsV(state.transformer.weightsV);
      this.#transformer.setWeightsOut(state.transformer.weightsOut);
      this.#transformer.setBiasQ(state.transformer.biasQ);
      this.#transformer.setBiasK(state.transformer.biasK);
      this.#transformer.setBiasV(state.transformer.biasV);
      this.#transformer.setBiasOut(state.transformer.biasOut);
    } catch {}
  }

  #saveLearningState() {
    const state = {
      qTable: this.#qTable,
      patternBuckets: this.#patternBuckets,
      totalPatterns: this.#totalPatterns
    };
    try {
      fs.writeFileSync(path.join(directoryPath, 'learning_state.json'), JSON.stringify(state), 'utf8');
    } catch {}
  }

  #loadLearningState() {
    try {
      const state = JSON.parse(fs.readFileSync(path.join(directoryPath, 'learning_state.json'), 'utf8'));
      if (typeof state.qTable === 'object' && typeof state.patternBuckets === 'object' && isValidNumber(state.totalPatterns)) {
        this.#qTable = {};
        for (const [key, value] of Object.entries(state.qTable)) {
          if (
            value &&
            isValidNumber(value.buy) &&
            isValidNumber(value.hold)
          ) {
            this.#qTable[key] = { buy: value.buy, hold: value.hold };
          }
        }
        this.#patternBuckets = {};
        this.#totalPatterns = 0;
        for (const [key, bucket] of Object.entries(state.patternBuckets)) {
          if (Array.isArray(bucket)) {
            this.#patternBuckets[key] = bucket
              .filter(p => Array.isArray(p.features) && p.features.every(isValidNumber) && isValidNumber(p.score))
              .slice(0, this.#bucketSize);
            this.#totalPatterns += this.#patternBuckets[key].length;
          }
        }
        // Ensure totalPatterns doesn't exceed maxPatterns
        if (this.#totalPatterns > this.#maxPatterns) {
          const keys = Object.keys(this.#patternBuckets);
          while (this.#totalPatterns > this.#maxPatterns && keys.length > 0) {
            const key = keys.pop();
            this.#totalPatterns -= this.#patternBuckets[key].length;
            delete this.#patternBuckets[key];
          }
        }
      }
    } catch {}
  }

  #saveOpenTrades() {
    try {
      fs.writeFileSync(path.join(directoryPath, 'open_trades.json'), JSON.stringify(this.#openTrades), 'utf8');
    } catch {}
  }

  #loadOpenTrades() {
    try {
      const openTrades = JSON.parse(fs.readFileSync(path.join(directoryPath, 'open_trades.json'), 'utf8'));
      if (Array.isArray(openTrades)) {
        this.#openTrades = openTrades
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
        this.#openTrades.forEach(trade => {
          this.#qTable[trade.stateKey] = this.#qTable[trade.stateKey] || { buy: 0, hold: 0 };
        });
      }
    } catch {}
  }

  #saveCandleEmbeddings() {
    const state = { longTermBuffer: this.#longTermBuffer.slice(-this.#maxCandles / 2) };
    try {
      fs.writeFileSync(path.join(directoryPath, 'candle_embeddings.json'), JSON.stringify(state), 'utf8');
    } catch {}
  }

  #loadCandleEmbeddings() {
    try {
      const state = JSON.parse(fs.readFileSync(path.join(directoryPath, 'candle_embeddings.json'), 'utf8'));
      if (Array.isArray(state.longTermBuffer)) {
        this.#longTermBuffer = state.longTermBuffer
          .filter(embedding => Array.isArray(embedding) && embedding.every(isValidNumber))
          .slice(-this.#maxCandles);
      }
    } catch {}
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
    const bucket = this.#patternBuckets[key] || [];
    if (bucket.length === 0) return 0;
    let totalScore = 0;
    let matchCount = 0;
    for (const pattern of bucket) {
      if (features.every((f, i) => isValidNumber(f) && isValidNumber(pattern.features[i]) && Math.abs(f - pattern.features[i]) < 0.1)) {
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
    const quantized = features.map(f => isValidNumber(f) ? Math.round(f * 10) / 10 : 0);
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
      !this.#candles.some(existing => existing.timestamp === c.timestamp)
    );

    if (newCandles.length === 0) return;

    this.#openTrades = this.#openTrades.map(trade => ({
      ...trade,
      candlesHeld: trade.candlesHeld + newCandles.length
    }));

    const closedTrades = [];
    this.#openTrades = this.#openTrades.filter(trade => {
      let isOpen = true;
      for (const candle of newCandles) {
        if (!candle || !isValidNumber(candle.high) || !isValidNumber(candle.low)) continue;

        const entryPrice = trade.entryPrice;
        const sellPrice = trade.sellPrice;
        const stopLoss = trade.stopLoss;

        if (candle.high >= sellPrice) {
          const outcome = (sellPrice - entryPrice) / entryPrice;
          closedTrades.push({
            timestamp: Date.now(),
            entryPrice,
            exitPrice: sellPrice,
            confidence: trade.confidence,
            outcome: Math.min(Math.max(outcome, -1), 1),
            reward: outcome * (trade.confidence / 100),
            strategy: trade.strategy,
            patternScore: trade.patternScore,
            candlesHeld: trade.candlesHeld,
            features: trade.features,
            stateKey: trade.stateKey,
            dynamicThreshold: trade.dynamicThreshold
          });
          isOpen = false;
          break;
        } else if (candle.low <= stopLoss) {
          const outcome = (stopLoss - entryPrice) / entryPrice;
          closedTrades.push({
            timestamp: Date.now(),
            entryPrice,
            exitPrice: stopLoss,
            confidence: trade.confidence,
            outcome: Math.min(Math.max(outcome, -1), 1),
            reward: outcome * (trade.confidence / 100),
            strategy: trade.strategy,
            patternScore: trade.patternScore,
            candlesHeld: trade.candlesHeld,
            features: trade.features,
            stateKey: trade.stateKey,
            dynamicThreshold: trade.dynamicThreshold
          });
          isOpen = false;
          break;
        }
      }
      return isOpen;
    });

    for (const trade of closedTrades) {
      this.#trades.push(trade);
      this.#trades = this.#trades.slice(-this.#maxTrades);

      const pattern = { features: trade.features, score: trade.reward };
      const key = this.#generateFeatureKey(trade.features);
      this.#patternBuckets[key] = this.#patternBuckets[key] || [];

      if (this.#patternBuckets[key].length < this.#bucketSize) {
        this.#patternBuckets[key].push(pattern);
        this.#totalPatterns++;
      } else {
        // Replace the lowest-scoring pattern in the bucket if new score is higher
        const minIndex = this.#patternBuckets[key].reduce((minIdx, p, idx) => p.score < this.#patternBuckets[key][minIdx].score ? idx : minIdx, 0);
        if (pattern.score > this.#patternBuckets[key][minIndex].score) {
          this.#patternBuckets[key][minIndex] = pattern;
        }
      }

      // Evict buckets if total patterns exceed maxPatterns
      // In #updateOpenTrades, replace the eviction logic (lines around 614-620 in your file)
      // Evict buckets if total patterns exceed maxPatterns
      if (this.#totalPatterns > this.#maxPatterns) {
        const keys = Object.keys(this.#patternBuckets);
        if (keys.length > this.#maxBuckets) {
          // Find the bucket with the lowest average score
          let lowestAvgScore = Infinity;
          let keyToRemove = null;
          for (const key of keys) {
            const bucket = this.#patternBuckets[key];
            const avgScore = bucket.length > 0 ? bucket.reduce((sum, p) => sum + p.score, 0) / bucket.length : 0;
            if (avgScore < lowestAvgScore) {
              lowestAvgScore = avgScore;
              keyToRemove = key;
            }
          }
          if (keyToRemove) {
            this.#totalPatterns -= this.#patternBuckets[keyToRemove].length;
            delete this.#patternBuckets[keyToRemove];
          }
        }
      }

      this.#state.tradeCount++;
      this.#state.winRate = this.#trades.length > 0 ? this.#trades.filter(t => t.outcome > 0).length / this.#trades.length : 0.5;
      this.#state.avgReward = this.#trades.length > 0 ? this.#trades.reduce((sum, t) => sum + t.reward, 0) / this.#trades.length : 0;
      this.#state.signalReliability = this.#trades.length > 0 ? this.#trades.filter(t => t.confidence > trade.dynamicThreshold && t.outcome > 0).length / this.#trades.length : 0.5;
      this.#lifetimeState.totalTrades++;
      if (trade.outcome > 0) this.#lifetimeState.totalWins++;
      this.#lifetimeState.totalAccuracy = this.#lifetimeState.totalTrades > 0 ? (this.#lifetimeState.totalWins / this.#lifetimeState.totalTrades) * 100 : 50;
      this.#performanceBaseline.avgWinRate = (this.#performanceBaseline.avgWinRate * 0.9 + this.#state.winRate * 0.1);
      this.#performanceBaseline.avgReward = (this.#performanceBaseline.avgReward * 0.9 + this.#state.avgReward * 0.1);
      this.#performanceBaseline.avgConfidence = (this.#performanceBaseline.avgConfidence * 0.9 + trade.confidence * 0.1);
      
      this.#qTable[trade.stateKey] = this.#qTable[trade.stateKey] || { buy: 0, hold: 0 };
      this.#qTable[trade.stateKey].buy += this.#config.learningRate * (trade.reward - this.#qTable[trade.stateKey].buy);
      
      this.#transformer.train(trade.features, trade.outcome > 0 ? 1 : 0);
      this.#autoencoder.train(trade.features);
    }
  }

  getSignalsAndHealth(candles, saveState = true) {
    if (!Array.isArray(candles) || candles.length === 0) {
      return {
        error : 'Invalid candle array type or length'
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
      !this.#candles.some(existing => existing.timestamp === c.timestamp)
    );
    if (newCandles.length === 0 && this.#candles.length === 0) {
      return {
        error : 'Invalid candle array type or length'
      };
    }

    this.#candles.push(...newCandles);
    this.#candles = this.#candles.slice(-this.#maxCandles);
    this.#compressCandles();
    const indicators = this.#indicators.compute(this.#candles);
    if (indicators.error) {
      return {
        error : 'Indicators error'
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

    const stateKey = JSON.stringify(features.map(f => isValidNumber(f) ? Math.round(f * 10) / 10 : 0));
    this.#qTable[stateKey] = this.#qTable[stateKey] || { buy: 0, hold: 0 };

    const baseRate = this.#config.explorationRate;
    const decayFactor = 1 - Math.min(this.#lifetimeState.totalTrades / 25000, 1);
    const effectiveExplorationRate = Math.max(baseRate * decayFactor, baseRate * 0.1);

    const action = Math.random() < effectiveExplorationRate
      ? (Math.random() < 0.5 ? 'buy' : 'hold')
      : this.#qTable[stateKey].buy > this.#qTable[stateKey].hold ? 'buy' : 'hold';

    if (action === 'buy' && confidence >= dynamicThreshold && this.#candles.length > 1) {
      const entryPrice = indicators.lastClose;
      const trade = {
        timestamp: Date.now(),
        entryPrice,
        sellPrice: truncateToDecimals(sellPrice, 2),
        stopLoss: truncateToDecimals(stopLoss, 2),
        confidence,
        strategy: action,
        patternScore,
        features,
        stateKey,
        dynamicThreshold,
        candlesHeld: 0
      };
      this.#openTrades.push(trade);
    }

    this.#state.lastUpdate = Date.now();

    if (saveState) {
      this.#saveState();
    }

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
      signalReliability: truncateToDecimals(this.#state.signalReliability, 4)
    };
  }

  dumpState () {
    this.#saveState()
  }
}

export default NeuralSignalEngine;