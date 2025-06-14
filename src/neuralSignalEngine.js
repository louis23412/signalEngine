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
        Threshold REAL NOT NULL
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

  #robustNormalize(data, count = 1, lowerPercentile = 0.05, upperPercentile = 0.95) {
    if (!Array.isArray(data) || data.length < 2) return Array(count).fill(0);
    
    const actualCount = Math.min(count, data.length);
    const valuesToNormalize = data.slice(-actualCount);
    
    if (!valuesToNormalize.every(isValidNumber)) return Array(actualCount).fill(0);
    
    const sortedData = [...data].sort((a, b) => a - b);
    const lowerIdx = Math.floor(lowerPercentile * sortedData.length);
    const upperIdx = Math.ceil(upperPercentile * sortedData.length) - 1;
    const min = sortedData[lowerIdx];
    const max = sortedData[upperIdx];
    
    if (!isValidNumber(min) || !isValidNumber(max) || max === min) return Array(actualCount).fill(0);
    
    return valuesToNormalize.map(value => truncateToDecimals(Math.min(1, Math.max(0, (value - min) / (max - min))), 4));
  };

  #extractFeatures(data) {
    const normalizedRsi = this.#robustNormalize(data.rsi, 10);
    const normalizedMacdDiff = this.#robustNormalize(data.macdDiff, 10);
    const normalizedEma100 = this.#robustNormalize(data.ema100, 10);
    const normalizedStochasticDiff = this.#robustNormalize(data.stochasticDiff, 10);
    const normalizedBollingerPercentB = this.#robustNormalize(data.bollingerPercentB, 10);
    const normalizedObv = this.#robustNormalize(data.obv, 10);
    const normalizedAdx = this.#robustNormalize(data.adx, 10);
    const normalizedCci = this.#robustNormalize(data.cci, 10);
    const normalizedWilliamsR = this.#robustNormalize(data.williamsR, 10);
    const normalizedCmf = this.#robustNormalize(data.cmf, 10);

    const result = Array.from({ length: 10 }, (_, i) => [
        normalizedRsi[i],
        normalizedMacdDiff[i],
        normalizedEma100[i],
        normalizedStochasticDiff[i],
        normalizedBollingerPercentB[i],
        normalizedObv[i],
        normalizedAdx[i],
        normalizedCci[i],
        normalizedWilliamsR[i],
        normalizedCmf[i]
    ]);

    return result;
  }

  #generateFeatureKey(features) {
    if (!Array.isArray(features) || features.length !== 10) return 'default';
    const quantized = features.map(f => isValidNumber(f) ? Math.round(f * 1000) / 1000 : 0);
    return quantized.join('|');
  }

  #scorePattern(features, key) {
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

  #updateOpenTrades(candles, status) {
    if (!Array.isArray(candles) || candles.length === 0) return;

    const minLow = Math.min(...candles.map(c => isValidNumber(c.low) ? c.low : Infinity));
    const maxHigh = Math.max(...candles.map(c => isValidNumber(c.high) ? c.high : -Infinity));
    if (!isValidNumber(minLow) || !isValidNumber(maxHigh)) return;

    const tradesStmt = this.#db.prepare(`
      SELECT timestamp, sellPrice, stopLoss, entryPrice, confidence, patternScore, features, stateKey, Threshold
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
            key: trade.stateKey,
            threshold: trade.threshold
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
          const pattern = patternStmt.get(trade.key, JSON.stringify(trade.features.at(-1)));
          const isWin = trade.outcome > 0 ? 1 : 0;
          const usageCount = pattern ? pattern.usage_count + 1 : 1;
          const winCount = pattern ? pattern.win_count + isWin : isWin;
          insertPatternStmt.run(trade.key, JSON.stringify(trade.features.at(-1)), trade.reward, usageCount, winCount);

          const existingQ = this.#db.prepare(`SELECT buy, hold FROM qtable WHERE state_key = ?`).get(trade.key) || { buy: 0, hold: 0 };
          updateQTableStmt.run(trade.key, existingQ.buy + this.#config.learningRate * (trade.reward - existingQ.buy), trade.key);

          const winRate = pattern && pattern.usage_count > 0 ? (pattern.win_count + isWin) / usageCount : isWin;
          const target = (trade.outcome + 1) / 2 * (0.8 + 0.2 * winRate);

          console.log(`Training triggered with .train for open trade: ${tempCounter} / ${closedTrades.length}...`)
          const startTime = process.hrtime();

          this.#transformer.train(
            trade.features.flat(), 
            target, 
            winRate, 
            (tempCounter === closedTrades.length) && (status === 'production')
          );

          const diff = process.hrtime(startTime);
          const executionTime = (diff[0] * 1e9 + diff[1]) / 1e9;
          console.log(`Training complete! - Execution time: ${executionTime} seconds`);
        }

        for (const key of keysToDelete) {
          deleteTradeStmt.run(key);
        }
      });
      transaction();
    }
  }

  getSignal(candles, status = 'production') {
    if (status !== 'production' && status !== 'training' && status !== 'dump') {
      return { error : 'Invalid status. Valid options: production / training / dump' }
    }

    const { error, candles: recentCandles } = this.#getRecentCandles(candles);
    if (error) {
      return { error };
    }

    this.#updateOpenTrades(recentCandles, status);

    const indicators = this.#indicators.compute(recentCandles);
    if (indicators.error) {
      return { error: 'Indicators error' };
    }

    const features = this.#extractFeatures(indicators);
    const key = this.#generateFeatureKey(features.at(-1));
    const patternScore = this.#scorePattern(features.at(-1), key);

    const patternStmt = this.#db.prepare(`SELECT usage_count, win_count FROM patterns WHERE bucket_key = ? AND features = ?`);
    const pattern = patternStmt.get(key, JSON.stringify(features.at(-1)));

    console.log(`Prediction triggered with .forward for key: ${key}...`)
    const startTime = process.hrtime();

    const confidence = this.#transformer.forward(features.flat())[0] * 100 * (1 + patternScore);

    const diff = process.hrtime(startTime);
    const executionTime = (diff[0] * 1e9 + diff[1]) / 1e9;
    console.log(`Prediction complete! (${confidence} %) Execution time: ${executionTime} seconds`);

    const scaleFactor = Math.max(0, (confidence - this.#config.baseConfidenceThreshold) / (100 - this.#config.baseConfidenceThreshold));
    const multiplier = Math.min(Math.max(this.#config.minMultiplier + (this.#config.maxMultiplier - this.#config.minMultiplier) * scaleFactor, this.#config.minMultiplier), this.#config.maxMultiplier);
    
    let sellPrice = indicators.lastClose + this.#config.atrFactor * indicators.lastAtr;
    let stopLoss = indicators.lastClose - this.#config.stopFactor * indicators.lastAtr;
    sellPrice = isValidNumber(sellPrice) && sellPrice > indicators.lastClose && sellPrice <= indicators.lastClose * 1.3 ? sellPrice : indicators.lastClose * 1.001;
    stopLoss = isValidNumber(stopLoss) && stopLoss < indicators.lastClose && stopLoss > 0 && stopLoss >= indicators.lastClose * 0.7 ? stopLoss : indicators.lastClose * 0.999;

    const fee = 0.0021;
    const adjustedSellPrice = sellPrice * (1 - fee);
    const expectedReward = isValidNumber(adjustedSellPrice) && isValidNumber(indicators.lastClose) && indicators.lastClose !== 0
      ? (adjustedSellPrice - indicators.lastClose) / indicators.lastClose
      : 0;

    const qTableStmt = this.#db.prepare(`SELECT buy, hold FROM qtable WHERE state_key = ?`);
    let qValues = qTableStmt.get(key);
    if (!qValues) {
      this.#db.prepare(`INSERT OR IGNORE INTO qtable (state_key, buy, hold) VALUES (?, 0, 0)`).run(key);
      qValues = { buy: 0, hold: 0 };
    }

    const entryPrice = indicators.lastClose;
    const timestamp = Date.now().toString();

    if (!pattern) {
      const insertPatternStmt = this.#db.prepare(`
        INSERT OR IGNORE INTO patterns (bucket_key, features, score, usage_count, win_count)
        VALUES (?, ?, ?, 0, 0)
      `);
      insertPatternStmt.run(key, JSON.stringify(features.at(-1)), patternScore);
    }

    const insertTradeStmt = this.#db.prepare(`
      INSERT INTO open_trades (timestamp, sellPrice, stopLoss, entryPrice, confidence, patternScore, features, stateKey, threshold)
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
      key,
      this.#config.baseConfidenceThreshold
    );

    if (status === 'dump') {
      this.#transformer.dumpState()
    }

    return {
      entryPrice,
      sellPrice: isValidNumber(sellPrice) ? truncateToDecimals(sellPrice, 2) : 0,
      stopLoss: isValidNumber(stopLoss) ? truncateToDecimals(stopLoss, 2) : 0,
      multiplier: isValidNumber(multiplier) ? truncateToDecimals(multiplier, 3) : this.#config.minMultiplier,
      expectedReward: truncateToDecimals(expectedReward, 8),
      confidence: confidence,
      threshold: this.#config.baseConfidenceThreshold
    }
  }
}

export default NeuralSignalEngine;