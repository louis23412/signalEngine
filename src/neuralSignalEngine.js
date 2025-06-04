import fs from 'fs';
import path from 'path';
import Database from 'better-sqlite3';

import HiveMind from './hiveMind.js';
import IndicatorProcessor from './indicatorProcessor.js';

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

class NeuralSignalEngine {
  #transformer = new HiveMind();
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
        usage_count INTEGER NOT NULL DEFAULT 0,
        win_count INTEGER NOT NULL DEFAULT 0,
        UNIQUE(bucket_key, features)
      );
      CREATE TABLE IF NOT EXISTS open_trades (
        timestamp TEXT PRIMARY KEY,
        sellPrice REAL NOT NULL,
        stopLoss REAL NOT NULL,
        entryPrice REAL NOT NULL,
        confidence REAL NOT NULL,
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

  #getRecentCandles(candles) {
    if (!Array.isArray(candles) || candles.length === 0) {
      return { error: 'Invalid candle array type or length', candles: [] };
    }

    const newCandles = candles.filter(c =>
      isValidNumber(c.timestamp) &&
      isValidNumber(c.open) &&
      isValidNumber(c.high) &&
      isValidNumber(c.low) &&
      isValidNumber(c.close) &&
      isValidNumber(c.volume) &&
      c.volume >= 0
    );

    let recentCandles = [];
    const transaction = this.#db.transaction(() => {
      if (newCandles.length > 0) {
        const insertCandleStmt = this.#db.prepare(`
          INSERT OR IGNORE INTO candles (timestamp, open, high, low, close, volume)
          VALUES (?, ?, ?, ?, ?, ?)
        `);
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
      }

      const fetchCandlesStmt = this.#db.prepare(`SELECT * FROM candles ORDER BY timestamp ASC LIMIT 1000`);
      recentCandles = fetchCandlesStmt.all();

      const cleanupStmt = this.#db.prepare(`DELETE FROM candles WHERE timestamp NOT IN (SELECT timestamp FROM candles ORDER BY timestamp DESC LIMIT 1000)`);
      cleanupStmt.run();
    });
    transaction();

    if (recentCandles.length === 0) {
      return { error: 'No valid candles available', candles: [] };
    }

    return { error: null, candles: recentCandles };
  }

  #extractFeatures(data) {
    const normalize = (value, min, max) => {
      if (!isValidNumber(value) || !isValidNumber(min) || !isValidNumber(max) || max === min) return 0;
      return Math.min(1, Math.max(0, (value - min) / (max - min)));
    };
    return [
      truncateToDecimals(normalize(data.rsi[data.rsi.length - 1], data.rsiMin, data.rsiMax), 6),
      truncateToDecimals(normalize(data.macd[data.macd.length - 1].MACD - data.macd[data.macd.length - 1].signal, data.macdMin, data.macdMax), 6),
      truncateToDecimals(normalize(data.atr[data.atr.length - 1], data.atrMin, data.atrMax), 6),
      truncateToDecimals(Math.min(1, Math.max(-1, data.volumeZScore / 3)), 6),
      data.isTrending,
      data.isRanging
    ];
  }

  #generateFeatureKey(features) {
    if (!Array.isArray(features) || features.length !== 6) return 'default';
    const quantized = features.map(f => isValidNumber(f) ? Math.round(f * 10000) / 10000 : 0);
    return quantized.join('|');
  }

  #scorePattern(features) {
    const key = this.#generateFeatureKey(features);
    const stmt = this.#db.prepare(`SELECT score, features, usage_count, win_count FROM patterns WHERE bucket_key = ?`);
    const patterns = stmt.all(key);
    if (!patterns || patterns.length === 0) return 0;
    let totalScore = 0, matchCount = 0;
    for (const pattern of patterns) {
      const patternFeatures = JSON.parse(pattern.features);
      if (features.every((f, i) => isValidNumber(f) && isValidNumber(patternFeatures[i]) && Math.abs(f - patternFeatures[i]) < 0.1)) {
        const pseudoWins = 1;
        const pseudoUses = 2;
        const winRate = isValidNumber(pattern.usage_count) && pattern.usage_count > 0
          ? (pattern.win_count + pseudoWins) / (pattern.usage_count + pseudoUses)
          : 0.5;
        totalScore += pattern.score * (0.5 + 0.5 * winRate);
        matchCount++;
      }
    }
    return matchCount > 0 ? totalScore / matchCount : 0;
  }

  #computeDynamicThreshold(data, confidence, baseThreshold = this.#config.baseConfidenceThreshold, winRate = 0.5) {
    const normalize = (value, min, max) => {
      if (!isValidNumber(value) || !isValidNumber(min) || !isValidNumber(max) || max === min) return 0.5;
      return Math.min(1, Math.max(0, (value - min) / (max - min)));
    };
    const atrNorm = normalize(data.atr[data.atr.length - 1], data.atrMin, data.atrMax);
    const rsiNorm = normalize(data.rsi[data.rsi.length - 1], data.rsiMin, data.rsiMax);
    const volumeNorm = isValidNumber(data.volumeZScore) ? Math.abs(data.volumeZScore) / 3 : 0;
    const volatilityScore = (atrNorm + volumeNorm + rsiNorm * 0.5) / 2.5;
    const marketCondition = data.isTrending ? 0.8 : data.isRanging ? 1.2 : 1.0;
    let dynamicThreshold = baseThreshold * volatilityScore * marketCondition * (1 - 0.2 * winRate);
    dynamicThreshold = Math.max(40, Math.min(80, isValidNumber(dynamicThreshold) ? dynamicThreshold : 60));
    if (!isValidNumber(confidence)) return parseFloat(dynamicThreshold.toFixed(3));
    const confidenceProximity = Math.abs(confidence - dynamicThreshold) / 100;
    return (dynamicThreshold * (1 - 0.1 * confidenceProximity));
  }

  #updateOpenTrades(candles) {
    if (!Array.isArray(candles) || candles.length === 0) return;

    const minLow = Math.min(...candles.map(c => isValidNumber(c.low) ? c.low : Infinity));
    const maxHigh = Math.max(...candles.map(c => isValidNumber(c.high) ? c.high : -Infinity));
    if (!isValidNumber(minLow) || !isValidNumber(maxHigh)) return;

    const tradesStmt = this.#db.prepare(`
      SELECT timestamp, sellPrice, stopLoss, entryPrice, confidence, patternScore, features, stateKey, dynamicThreshold
      FROM open_trades
      WHERE sellPrice BETWEEN ? AND ? OR stopLoss BETWEEN ? AND ?
    `);
    const trades = tradesStmt.all(minLow, maxHigh, minLow, maxHigh);

    const closedTrades = [];
    const keysToDelete = new Set();

    for (const trade of trades) {
      const features = JSON.parse(trade.features);
      for (const candle of candles) {
        if (!candle || !isValidNumber(candle.high) || !isValidNumber(candle.low)) continue;

        if (candle.high >= trade.sellPrice || candle.low <= trade.stopLoss) {
          const exitPrice = candle.high >= trade.sellPrice ? trade.sellPrice : trade.stopLoss;
          const outcome = (exitPrice - trade.entryPrice) / trade.entryPrice;
          closedTrades.push({
            timestamp: Date.now(),
            entryPrice: trade.entryPrice,
            exitPrice,
            confidence: trade.confidence,
            outcome: Math.min(Math.max(outcome, -1), 1),
            reward: outcome * (trade.confidence / 100),
            patternScore: trade.patternScore,
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
        const patternStmt = this.#db.prepare(`SELECT score, usage_count, win_count FROM patterns WHERE bucket_key = ? AND features = ?`);
        const insertPatternStmt = this.#db.prepare(`
          INSERT OR REPLACE INTO patterns (bucket_key, features, score, usage_count, win_count)
          VALUES (?, ?, ?, ?, ?)
        `);
        const updateQTableStmt = this.#db.prepare(`
          INSERT OR REPLACE INTO qtable (state_key, buy, hold)
          VALUES (?, ?, COALESCE((SELECT hold FROM qtable WHERE state_key = ?), 0))
        `);
        const deleteTradeStmt = this.#db.prepare(`DELETE FROM open_trades WHERE timestamp = ?`);

        let tempCounter = 0
        for (const trade of closedTrades) {
          tempCounter++
          const key = this.#generateFeatureKey(trade.features);
          const pattern = patternStmt.get(key, JSON.stringify(trade.features));
          const isWin = trade.outcome > 0 ? 1 : 0;
          const usageCount = pattern ? pattern.usage_count + 1 : 1;
          const winCount = pattern ? pattern.win_count + isWin : isWin;
          insertPatternStmt.run(key, JSON.stringify(trade.features), trade.reward, usageCount, winCount);

          const existingQ = this.#db.prepare(`SELECT buy, hold FROM qtable WHERE state_key = ?`).get(trade.stateKey) || { buy: 0, hold: 0 };
          updateQTableStmt.run(trade.stateKey, existingQ.buy + this.#config.learningRate * (trade.reward - existingQ.buy), trade.stateKey);

          const winRate = pattern && pattern.usage_count > 0 ? (pattern.win_count + isWin) / usageCount : isWin;
          const target = (trade.outcome + 1) / 2 * (0.8 + 0.2 * winRate);
          this.#transformer.train(trade.features, target, winRate, tempCounter === closedTrades.length);
        }

        for (const key of keysToDelete) {
          deleteTradeStmt.run(key);
        }
      });
      transaction();
    }
  }

  #computeAdvancedAction(qValues, confidence, dynamicThreshold, features, patternScore, winRate) {
    if (
      !qValues ||
      !isValidNumber(confidence) ||
      !isValidNumber(dynamicThreshold) ||
      !Array.isArray(features) ||
      features.length !== 6 ||
      !features.every(isValidNumber) ||
      !isValidNumber(patternScore) ||
      !isValidNumber(winRate)
    ) {
      return 'hold';
    }

    const qBuy = isValidNumber(qValues.buy) ? qValues.buy : 0;
    const qHold = isValidNumber(qValues.hold) ? qValues.hold : 0;
    const maxQ = Math.max(qBuy, qHold, 1e-6);
    const temperature = 0.5 + 0.5 * winRate;
    const expBuy = Math.exp((qBuy - maxQ) / temperature);
    const expHold = Math.exp((qHold - maxQ) / temperature);
    const sumExp = expBuy + expHold || 1;
    const probBuy = expBuy / sumExp;
    const probHold = expHold / sumExp;

    const renyiEntropy = -Math.log2(probBuy ** 2 + probHold ** 2 + 1e-6) / Math.log2(2);
    const normalizedEntropy = Math.min(1, renyiEntropy);

    const priorConfidence = 0.5;
    const evidenceStrength = 1 + patternScore * (0.5 + 0.5 * winRate);
    const bayesianConfidence = (confidence * evidenceStrength + priorConfidence * 0.1) / (evidenceStrength + 0.1);
    const riskAdjustedConfidence = bayesianConfidence * (1 - 0.3 * normalizedEntropy * (1 - winRate));

    const featureNorm = Math.sqrt(features.reduce((sum, f) => sum + f ** 2, 0)) || 1;
    const normalizedFeatures = features.map(f => f / featureNorm);
    const idealBuyFeature = [1, 1, 0.5, 0, 1, 0];
    const idealHoldFeature = [0.5, 0, 0.5, 0, 0, 1];
    const buySimilarity = normalizedFeatures.reduce((sum, f, i) => sum + f * idealBuyFeature[i], 0) / (Math.sqrt(idealBuyFeature.reduce((sum, f) => sum + f ** 2, 0)) || 1);
    const holdSimilarity = normalizedFeatures.reduce((sum, f, i) => sum + f * idealHoldFeature[i], 0) / (Math.sqrt(idealHoldFeature.reduce((sum, f) => sum + f ** 2, 0)) || 1);
    const contextScore = buySimilarity / (buySimilarity + holdSimilarity + 1e-6);

    const featureVariance = features.reduce((sum, f) => sum + (f - 0.5) ** 2, 0) / features.length;
    const volatilityScore = Math.sqrt(featureVariance) * (1 + 0.2 * normalizedEntropy);
    const patternReliability = patternScore > 0 ? 1 + 0.3 * patternScore : 1;
    const marketStability = winRate > 0.5 ? 1 + 0.2 * (winRate - 0.5) : 1 - 0.2 * (0.5 - winRate);
    const riskScore = 0.4 * volatilityScore + 0.4 * (1 - patternReliability) + 0.2 * (1 - marketStability);

    const baseThreshold = dynamicThreshold * patternReliability * (1 - 0.2 * normalizedEntropy);
    const logisticAdjustment = 1 / (1 + Math.exp(-10 * (contextScore - 0.5)));
    const adaptiveThreshold = baseThreshold * (0.6 + 0.4 * logisticAdjustment) * (0.7 + 0.3 * winRate);

    const decisionScore = (
      0.5 * riskAdjustedConfidence * probBuy +
      0.3 * contextScore +
      0.2 * patternScore * winRate
    ) * (1 - 0.1 * riskScore);

    const hysteresisFactor = 1.05;
    const buyThreshold = adaptiveThreshold * (probBuy > probHold ? 1 : hysteresisFactor);

    return {
      suggestedAction : decisionScore >= buyThreshold ? 'buy' : 'hold',
      decisionScore,
      buyThreshold
    }
  }

  getSignal(candles) {
    const { error, candles: recentCandles } = this.#getRecentCandles(candles);
    if (error) {
      return { error };
    }

    this.#updateOpenTrades(recentCandles);

    const indicators = this.#indicators.compute(recentCandles);
    if (indicators.error) {
      return { error: 'Indicators error' };
    }

    const features = this.#extractFeatures(indicators);
    const patternScore = this.#scorePattern(features);
    const confidence = this.#transformer.forward(features)[0] * 100 * (1 + patternScore);

    const key = this.#generateFeatureKey(features);
    const patternStmt = this.#db.prepare(`SELECT usage_count, win_count FROM patterns WHERE bucket_key = ? AND features = ?`);
    const pattern = patternStmt.get(key, JSON.stringify(features));
    const winRate = pattern && pattern.usage_count > 0
      ? pattern.win_count / pattern.usage_count
      : 0;

    const dynamicThreshold = this.#computeDynamicThreshold(indicators, confidence, this.#config.baseConfidenceThreshold, winRate);
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

    const entryPrice = indicators.lastClose;
    const timestamp = Date.now().toString();

    if (!pattern) {
      const insertPatternStmt = this.#db.prepare(`
        INSERT OR IGNORE INTO patterns (bucket_key, features, score, usage_count, win_count)
        VALUES (?, ?, ?, 0, 0)
      `);
      insertPatternStmt.run(key, JSON.stringify(features), patternScore);
    }

    const insertTradeStmt = this.#db.prepare(`
      INSERT INTO open_trades (timestamp, sellPrice, stopLoss, entryPrice, confidence, patternScore, features, stateKey, dynamicThreshold)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);
    insertTradeStmt.run(
      timestamp,
      truncateToDecimals(sellPrice, 2),
      truncateToDecimals(stopLoss, 2),
      entryPrice,
      confidence,
      patternScore,
      JSON.stringify(features),
      stateKey,
      dynamicThreshold
    );

    const filteredDecision = this.#computeAdvancedAction(qValues, confidence, dynamicThreshold, features, patternScore, winRate);

    return {
      suggestedAction: filteredDecision.suggestedAction,
      entryPrice,
      sellPrice: isValidNumber(sellPrice) ? truncateToDecimals(sellPrice, 2) : 0,
      stopLoss: isValidNumber(stopLoss) ? truncateToDecimals(stopLoss, 2) : 0,
      multiplier: isValidNumber(multiplier) ? truncateToDecimals(multiplier, 3) : this.#config.minMultiplier,
      expectedReward: truncateToDecimals(expectedReward, 8),
      rawConfidence: confidence,
      rawThreshold: dynamicThreshold,
      filteredConfidence: filteredDecision.decisionScore,
      filteredThreshold: filteredDecision.buyThreshold
    };
  }
}

export default NeuralSignalEngine;