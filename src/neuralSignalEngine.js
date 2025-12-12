import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import Database from 'better-sqlite3';

import HiveMind from './hiveMind.js';
import IndicatorProcessor from './indicatorProcessor.js';

import { truncateToDecimals, isValidNumber } from './utils.js';

const directoryPath = path.join(import.meta.dirname, '..', 'state');

class NeuralSignalEngine {
    #hivemind;
    #indicators;
    #db;
    #config = {
        minMultiplier: 1,
        maxMultiplier: 2.5,
        baseConfidenceThreshold: 50,
        atrFactor: 2.5,
        stopFactor: 1,
        minPriceMovement : 0.0021,
        maxPriceMovement : 0.05
    };
    #trainingStep = 0;

    #resetData = {
        gradientResetFreq : null,
        gradientResetStep : null,
        regulateFreq : null,
        regulateStep : null
    };

    #globalAccuracy = {
        total : 0,
        correct : 0,
        currentPoints : 0,
        maxPoints : 0
    }

    #openSimulations = 0;

    constructor() {
        fs.mkdirSync(directoryPath, { recursive: true });

        this.#hivemind = new HiveMind(directoryPath);

        const resetFreq = this.#hivemind.getFreq();
        this.#resetData.gradientResetFreq = resetFreq.gradientResetFreq
        this.#resetData.regulateFreq = resetFreq.regulateFreq

        this.#indicators = new IndicatorProcessor();
        this.#db = new Database(path.join(directoryPath, 'neural_engine.db'), { fileMustExist: false });

        this.#initDatabase();
        this.#loadGlobalAccuracy();
    }

    #initDatabase() {
        this.#db.exec(`
            CREATE TABLE IF NOT EXISTS open_trades (
                timestamp TEXT PRIMARY KEY,
                sellPrice REAL NOT NULL,
                stopLoss REAL NOT NULL,
                entryPrice REAL NOT NULL,
                features TEXT NOT NULL,
                confidence REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS candles (
                timestamp TEXT PRIMARY KEY,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS trained_features (
                encoding TEXT PRIMARY KEY
            );
            CREATE TABLE IF NOT EXISTS global_stats (
                key TEXT PRIMARY KEY,
                value INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_open_trades_sellPrice ON open_trades(sellPrice);
            CREATE INDEX IF NOT EXISTS idx_open_trades_stopLoss ON open_trades(stopLoss);
            CREATE INDEX IF NOT EXISTS idx_candles_timestamp ON candles(timestamp);
            CREATE INDEX IF NOT EXISTS idx_trained_features_encoding ON trained_features(encoding);
        `);
    }

    #loadGlobalAccuracy() {
        const selectStmt = this.#db.prepare(`
            SELECT value FROM global_stats WHERE key = ?
        `);

        const totalRow = selectStmt.get('accuracy_total');
        const correctRow = selectStmt.get('accuracy_correct');
        const currentPointsRow = selectStmt.get('accuracy_current_points');
        const maxPointsRow = selectStmt.get('accuracy_max_points');

        if (totalRow) {
            this.#globalAccuracy.total = totalRow.value;
        }
        if (correctRow) {
            this.#globalAccuracy.correct = correctRow.value;
        }
        if (currentPointsRow) {
            this.#globalAccuracy.currentPoints = currentPointsRow.value;
        }
        if (maxPointsRow) {
            this.#globalAccuracy.maxPoints = maxPointsRow.value;
        }
    }

    #saveGlobalAccuracy() {
        const upsertStmt = this.#db.prepare(`
            INSERT INTO global_stats (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
        `);

        const transaction = this.#db.transaction(() => {
            upsertStmt.run('accuracy_total', this.#globalAccuracy.total);
            upsertStmt.run('accuracy_correct', this.#globalAccuracy.correct);
            upsertStmt.run('accuracy_current_points', this.#globalAccuracy.currentPoints);
            upsertStmt.run('accuracy_max_points', this.#globalAccuracy.maxPoints);
        });
        transaction();
    }

    #getRecentCandles(candles) {
        if (!Array.isArray(candles) || candles.length === 0) {
            return { error: 'Invalid candle array type or length', recentCandles: [], fullCandles: [] };
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
        let fullCandles = [];
        const transaction = this.#db.transaction(() => {
            if (newCandles.length > 0) {
                const insertCandleStmt = this.#db.prepare(`
                    INSERT OR IGNORE INTO candles (timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?)
                `);
                const insertedTimestamps = [];
                for (const candle of newCandles) {
                    const result = insertCandleStmt.run(
                        candle.timestamp,
                        candle.open,
                        candle.high,
                        candle.low,
                        candle.close,
                        candle.volume
                    );

                    if (result.changes > 0) {
                        insertedTimestamps.push(candle.timestamp);
                    }
                }

                if (insertedTimestamps.length > 0) {
                    const placeholders = insertedTimestamps.map(() => '?').join(',');
                    const fetchRecentStmt = this.#db.prepare(`
                        SELECT * FROM candles WHERE timestamp IN (${placeholders}) ORDER BY timestamp ASC
                    `);
                    recentCandles = fetchRecentStmt.all(...insertedTimestamps);
                }
            }

            const fetchCandlesStmt = this.#db.prepare(`
                SELECT * FROM candles ORDER BY timestamp ASC LIMIT 1000
            `);
            fullCandles = fetchCandlesStmt.all();

            const cleanupStmt = this.#db.prepare(`
                DELETE FROM candles WHERE timestamp NOT IN (
                    SELECT timestamp FROM candles ORDER BY timestamp DESC LIMIT 1000
                )
            `);
            cleanupStmt.run();
        });
        transaction();

        if (fullCandles.length === 0) {
            return { error: 'No valid candles available', recentCandles: [], fullCandles: [] };
        }

        return { error: null, recentCandles, fullCandles };
    }

    #robustNormalize(data, count = 1, lowerPercentile = 0.05, upperPercentile = 0.95) {
        if (!Array.isArray(data) || data.length < 2) return Array(count).fill(0);
        
        const actualCount = Math.min(count, data.length);
        const valuesToNormalize = data.slice(-actualCount);
        
        if (!valuesToNormalize.every(isValidNumber)) return Array(actualCount).fill(0);
        
        const sortedData = [...data].sort((a, b) => a - b);
        const lowerIdx = Math.floor(lowerPercentile * sortedData.length);
        const upperIdx = Math.ceil(upperPercentile * sortedData.length) - 1;
        let min_h = sortedData[lowerIdx];
        let max_h = sortedData[upperIdx];
        
        const recentMin = Math.min(...valuesToNormalize);
        const recentMax = Math.max(...valuesToNormalize);
        let min = Math.min(min_h, recentMin);
        let max = Math.max(max_h, recentMax);
        
        if (max === min) {
            const median = sortedData[Math.floor(sortedData.length / 2)];
            const mad = sortedData.reduce((sum, val) => {
                return sum + Math.abs(val - median);
            }, 0) / sortedData.length;
            
            const scale = mad > 0 ? mad : Number.EPSILON * 1e6;
            min = median - scale;
            max = median + scale;
        }
        
        const range = max - min;
        const epsilon = Number.EPSILON * Math.max(Math.abs(min), Math.abs(max));
        if (range < epsilon) {
            min -= epsilon;
            max += epsilon;
        }
        
        return valuesToNormalize.map(value => {
            const normalized = (value - min) / (max - min);
            return truncateToDecimals(Math.min(1, Math.max(0, normalized)), 2);
        });
    }

    #extractFeatures(data, candleCount) {
        const normalizedRsi = this.#robustNormalize(data.rsi, candleCount);
        const normalizedMacdDiff = this.#robustNormalize(data.macdDiff, candleCount);
        const normalizedAtr = this.#robustNormalize(data.atr, candleCount);
        const normalizedEma100 = this.#robustNormalize(data.ema100, candleCount);
        const normalizedStochasticDiff = this.#robustNormalize(data.stochasticDiff, candleCount);
        const normalizedBollingerPercentB = this.#robustNormalize(data.bollingerPercentB, candleCount);
        const normalizedObv = this.#robustNormalize(data.obv, candleCount);
        const normalizedAdx = this.#robustNormalize(data.adx, candleCount);
        const normalizedCci = this.#robustNormalize(data.cci, candleCount);
        const normalizedWilliamsR = this.#robustNormalize(data.williamsR, candleCount);

        const result = Array.from({ length: candleCount }, (_, i) => [
            normalizedRsi[i],
            normalizedMacdDiff[i],
            normalizedAtr[i],
            normalizedEma100[i],
            normalizedStochasticDiff[i],
            normalizedBollingerPercentB[i],
            normalizedObv[i],
            normalizedAdx[i],
            normalizedCci[i],
            normalizedWilliamsR[i]
        ]);

        return result;
    }

    #updateOpenTrades(candles, shouldSave, cutoff) {
        if (!Array.isArray(candles) || candles.length === 0) return;

        const tradesStmt = this.#db.prepare(`
            SELECT timestamp, sellPrice, stopLoss, entryPrice, features, confidence
            FROM open_trades
        `);
        const trades = tradesStmt.all();

        const closedTrades = [];
        const keysToDelete = new Set();

        for (const trade of trades) {
            const features = JSON.parse(trade.features);
            for (const candle of candles) {
                if (!candle || !isValidNumber(candle.high) || !isValidNumber(candle.low)) continue;

                if (candle.low <= trade.stopLoss || candle.high >= trade.sellPrice) {
                    const exitPrice = candle.low <= trade.stopLoss ? trade.stopLoss : trade.sellPrice;
                    const outcome = candle.low <= trade.stopLoss ? 0 : 1;
                    
                    closedTrades.push({
                        timestamp: trade.timestamp,
                        entryPrice: trade.entryPrice,
                        exitPrice,
                        outcome,
                        features,
                        confidence : trade.confidence
                    });
                    break;
                }
            }
        }

        if (closedTrades.length > 0) {
            const transaction = this.#db.transaction(() => {
                const deleteTradeStmt = this.#db.prepare(`DELETE FROM open_trades WHERE timestamp = ?`);
                const checkEncodingStmt = this.#db.prepare(`SELECT encoding FROM trained_features WHERE encoding = ?`);
                const insertEncodingStmt = this.#db.prepare(`INSERT INTO trained_features (encoding) VALUES (?)`);

                let tradeCounter = 0;
                for (const trade of closedTrades) {
                    tradeCounter++;

                    if (trade.confidence !== -1) {
                        this.#globalAccuracy.total++
                        this.#globalAccuracy.maxPoints += 100

                        if (trade.outcome === 1) {
                            if (trade.confidence >= 50) {
                                this.#globalAccuracy.correct++;
                            }

                            this.#globalAccuracy.currentPoints += trade.confidence
                        }

                        if (trade.outcome === 0) {
                            if (trade.confidence < 50) {
                               this.#globalAccuracy.correct++; 
                            }

                            this.#globalAccuracy.currentPoints += Math.abs(trade.confidence - 100)
                        }
                    }

                    const flatFeatures = trade.features.flat()
                    const encodingString = `${flatFeatures.join(',')}|${trade.outcome}`;
                    const encodingHash = crypto.createHash('sha256').update(encodingString).digest('hex');

                    const existingEncoding = checkEncodingStmt.get(encodingHash);
                    if (existingEncoding) {
                        console.log(`Skipping training for closed trade (${trade.outcome ? 'win' : 'loss'}): ${tradeCounter} / ${closedTrades.length} (duplicate encoding)`);
                        keysToDelete.add(trade.timestamp);
                        continue;
                    }

                    console.log(`Training triggered for closed trade (${trade.outcome ? 'win' : 'loss'}): ${tradeCounter} / ${closedTrades.length}`);
                    const startTime = process.hrtime();

                    const trainingData = this.#hivemind.train(flatFeatures, trade.outcome)

                    const diff = process.hrtime(startTime);
                    const executionTime = truncateToDecimals((diff[0] * 1e9 + diff[1]) / 1e9, 4);
                    console.log(`Training complete (${trainingData.step})! - Execution time: ${executionTime} seconds`);

                    this.#trainingStep = trainingData.step;
                    this.#resetData.gradientResetStep = trainingData.lastGradientResetStep
                    this.#resetData.regulateStep = trainingData.lastRegulateStep

                    insertEncodingStmt.run(encodingHash);
                    keysToDelete.add(trade.timestamp);

                    if (cutoff !== null && this.#trainingStep === cutoff) { break }
                }

                if (shouldSave || (cutoff !== null && this.#trainingStep === cutoff)) {
                    const saveStatus = this.#hivemind.dumpState()

                    if (!saveStatus.status) {
                        console.log(`Hivemind state save failed! Error: ${saveStatus.error} Trace: ${saveStatus.trace}`)
                    } else {
                        console.log('Hivemind state saved!')
                    }
                }

                for (const key of keysToDelete) {
                    deleteTradeStmt.run(key);
                }

                this.#saveGlobalAccuracy();
            });

            transaction();
        }
    }

    getFreq () {
        return {
            gradientResetFreq : this.#resetData.gradientResetFreq,
            regulateFreq : this.#resetData.regulateFreq,
        }
    }

    getSignal(candles, shouldPredict = true, shouldSave = true, cutoff = null) {
        const { error, recentCandles, fullCandles } = this.#getRecentCandles(candles);

        if (error) return { error };

        // console.log(`Processed candles - Total: ${fullCandles.length} | Newly added: ${recentCandles.length}`);

        this.#updateOpenTrades(recentCandles, shouldSave, cutoff);

        const indicators = this.#indicators.compute(fullCandles);

        if (indicators.error) return { error: 'Indicators error' };

        const features = this.#extractFeatures(indicators, 1);

        let confidence = 'disabled';
        let multiplier = 'disabled';

        if (shouldPredict) {
            confidence = truncateToDecimals(this.#hivemind.predict(features.flat()) * 100, 4);

            const scaleFactor = Math.max(0, (confidence - this.#config.baseConfidenceThreshold) / (100 - this.#config.baseConfidenceThreshold));

            multiplier = truncateToDecimals(
                Math.min(
                    Math.max(
                        this.#config.minMultiplier + (this.#config.maxMultiplier - this.#config.minMultiplier) * scaleFactor, this.#config.minMultiplier
                    ), 
                    this.#config.maxMultiplier
                ),
                4
            );
        }
        
        const entryPrice = indicators.lastClose;
        const atrBasedSellPrice = indicators.lastClose + this.#config.atrFactor * indicators.lastAtr;
        const atrBasedStopLoss = indicators.lastClose - this.#config.stopFactor * indicators.lastAtr;

        const minPriceDelta = entryPrice * this.#config.minPriceMovement;
        const maxPriceDelta = entryPrice * this.#config.maxPriceMovement;

        const sellPriceDelta = Math.min(
            Math.max(atrBasedSellPrice - entryPrice, minPriceDelta),
            maxPriceDelta
        );
        const sellPrice = truncateToDecimals(entryPrice + sellPriceDelta, 2);

        const stopLossDelta = Math.min(
            Math.max(entryPrice - atrBasedStopLoss, minPriceDelta),
            maxPriceDelta
        );
        const stopLoss = truncateToDecimals(entryPrice - stopLossDelta, 2);

        const confidenceInsert = confidence !== 'disabled' ? confidence : -1

        const insertTradeStmt = this.#db.prepare(`
            INSERT INTO open_trades (timestamp, sellPrice, stopLoss, entryPrice, features, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        `);
        
        insertTradeStmt.run(
            recentCandles.at(-1).timestamp,
            sellPrice,
            stopLoss,
            entryPrice,
            JSON.stringify(features),
            confidenceInsert
        );

        let currentAcc;
        let trueAcc;
        if (confidenceInsert !== -1) {
            currentAcc = this.#globalAccuracy.total > 0 
                ? truncateToDecimals((this.#globalAccuracy.correct / this.#globalAccuracy.total) * 100, 3)
                : 'N/A'

            trueAcc = this.#globalAccuracy.maxPoints > 0 
                ? truncateToDecimals((this.#globalAccuracy.currentPoints / this.#globalAccuracy.maxPoints) * 100, 3)
                : 'N/A'
        } else {
            currentAcc = 'disabled';
            trueAcc = 'disabled'
        }

        const tradesStmt = this.#db.prepare(`SELECT timestamp FROM open_trades`);
        this.#openSimulations = tradesStmt.all().length;

        return {
            entryPrice,
            sellPrice,
            stopLoss,
            multiplier,
            confidence,
            threshold: this.#config.baseConfidenceThreshold,
            lastTrainingStep : this.#trainingStep,
            lastGradientResetStep : this.#resetData.gradientResetStep,
            lastRegulateStep : this.#resetData.regulateStep,
            countAccuracy : currentAcc,
            trueAccuracy : trueAcc,
            openSimulations : this.#openSimulations
        };
    }
}

export default NeuralSignalEngine;