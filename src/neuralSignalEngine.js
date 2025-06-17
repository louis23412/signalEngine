import fs from 'fs';
import path from 'path';
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

    constructor() {
        fs.mkdirSync(directoryPath, { recursive: true });

        this.#hivemind = new HiveMind(directoryPath);
        this.#indicators = new IndicatorProcessor();
        this.#db = new Database(path.join(directoryPath, 'neural_engine.db'), { fileMustExist: false });

        this.#initDatabase();
    }

    #initDatabase() {
        this.#db.exec(`
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                features TEXT NOT NULL,
                score REAL NOT NULL,
                usage_count INTEGER NOT NULL DEFAULT 0,
                win_count INTEGER NOT NULL DEFAULT 0,
                UNIQUE(features)
            );
            CREATE TABLE IF NOT EXISTS open_trades (
                timestamp TEXT PRIMARY KEY,
                sellPrice REAL NOT NULL,
                stopLoss REAL NOT NULL,
                entryPrice REAL NOT NULL,
                patternScore REAL NOT NULL,
                features TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS candles (
                timestamp TEXT PRIMARY KEY,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_patterns_features ON patterns(features);
            CREATE INDEX IF NOT EXISTS idx_open_trades_sellPrice ON open_trades(sellPrice);
            CREATE INDEX IF NOT EXISTS idx_open_trades_stopLoss ON open_trades(stopLoss);
            CREATE INDEX IF NOT EXISTS idx_candles_timestamp ON candles(timestamp);
        `);
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
        let min = sortedData[lowerIdx];
        let max = sortedData[upperIdx];
        
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

    #scorePattern(features) {
        if (!Array.isArray(features) || features.length !== 10 || !features.every(isValidNumber)) {
            return 0;
        }

        const stmt = this.#db.prepare(`SELECT score, usage_count, win_count FROM patterns WHERE features = ?`);
        const pattern = stmt.get(JSON.stringify(features));
        if (!pattern) {
            return 0;
        }

        const winRate = pattern.usage_count > 0 ? pattern.win_count / pattern.usage_count : 0;
        return truncateToDecimals(winRate, 4);
    }

    #updateOpenTrades(candles, status) {
        if (!Array.isArray(candles) || candles.length === 0) return;

        const tradesStmt = this.#db.prepare(`
            SELECT timestamp, sellPrice, stopLoss, entryPrice, patternScore, features
            FROM open_trades
        `);
        const trades = tradesStmt.all();

        console.log(`Current open trade simulations: ${trades.length}`);
        if (trades.length === 0) return;

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
                        timestamp: Date.now(),
                        entryPrice: trade.entryPrice,
                        exitPrice,
                        outcome,
                        patternScore: trade.patternScore,
                        features
                    });
                    keysToDelete.add(trade.timestamp);
                    break;
                }
            }
        }

        if (closedTrades.length > 0) {
            const transaction = this.#db.transaction(() => {
                const patternStmt = this.#db.prepare(`SELECT score, usage_count, win_count FROM patterns WHERE features = ?`);
                const insertPatternStmt = this.#db.prepare(`
                    INSERT OR REPLACE INTO patterns (features, score, usage_count, win_count)
                    VALUES (?, ?, ?, ?)
                `);
                const deleteTradeStmt = this.#db.prepare(`DELETE FROM open_trades WHERE timestamp = ?`);

                let tempCounter = 0;
                for (const trade of closedTrades) {
                    tempCounter++;
                    const pattern = patternStmt.get(JSON.stringify(trade.features.at(-1)));
                    const isWin = trade.outcome;
                    const usageCount = pattern ? pattern.usage_count + 1 : 1;
                    const winCount = pattern ? pattern.win_count + isWin : isWin;
                    const winRate = usageCount > 0 ? winCount / usageCount : 0;

                    insertPatternStmt.run(
                        JSON.stringify(trade.features.at(-1)),
                        truncateToDecimals(winRate, 4),
                        usageCount,
                        winCount
                    );

                    console.log(`Training triggered for closed trade (${isWin ? 'win' : 'loss'}): ${tempCounter} / ${closedTrades.length} - WinRate: ${winRate * 100}%`);
                    const startTime = process.hrtime();

                    this.#hivemind.train(
                        trade.features.flat(), 
                        isWin,
                        (tempCounter === closedTrades.length) && (status === 'production')
                    );

                    const diff = process.hrtime(startTime);
                    const executionTime = truncateToDecimals((diff[0] * 1e9 + diff[1]) / 1e9, 4);
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
            return { error: 'Invalid status. Valid options: production / training / dump' };
        }

        const { error, recentCandles, fullCandles } = this.#getRecentCandles(candles);

        if (error) {
            return { error };
        }

        console.log(`Processed candles - Total: ${fullCandles.length} | Newly added: ${recentCandles.length}`);

        this.#updateOpenTrades(recentCandles, status);

        const indicators = this.#indicators.compute(fullCandles);

        if (indicators.error) {
            return { error: 'Indicators error' };
        }

        const features = this.#extractFeatures(indicators, 10);
        const patternScore = this.#scorePattern(features.at(-1));

        const patternStmt = this.#db.prepare(`SELECT usage_count, win_count FROM patterns WHERE features = ?`);
        const pattern = patternStmt.get(JSON.stringify(features.at(-1)));

        console.log(`Forward triggered for pattern with win rate: ${patternScore * 100}%`);
        const startTime = process.hrtime();

        const confidence = truncateToDecimals(this.#hivemind.forward(features.flat())[0] * 100, 4);

        const diff = process.hrtime(startTime);
        const executionTime = truncateToDecimals((diff[0] * 1e9 + diff[1]) / 1e9, 4);
        console.log(`Forward complete! (${confidence} %) Execution time: ${executionTime} seconds`);

        const scaleFactor = Math.max(0, (confidence - this.#config.baseConfidenceThreshold) / (100 - this.#config.baseConfidenceThreshold));
        const multiplier = truncateToDecimals(Math.min(Math.max(this.#config.minMultiplier + (this.#config.maxMultiplier - this.#config.minMultiplier) * scaleFactor, this.#config.minMultiplier), this.#config.maxMultiplier), 4);
        
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

        if (!pattern) {
            const insertPatternStmt = this.#db.prepare(`
                INSERT OR IGNORE INTO patterns (features, score, usage_count, win_count)
                VALUES (?, ?, 0, 0)
            `);
            const result = insertPatternStmt.run(JSON.stringify(features.at(-1)), patternScore);
            if (result.changes > 0) {
                console.log(`New pattern created for features: ${JSON.stringify(features.at(-1))}`);
            }
        }

        const insertTradeStmt = this.#db.prepare(`
            INSERT INTO open_trades (timestamp, sellPrice, stopLoss, entryPrice, patternScore, features)
            VALUES (?, ?, ?, ?, ?, ?)
        `);
        insertTradeStmt.run(
            Date.now().toString(),
            sellPrice,
            stopLoss,
            entryPrice,
            patternScore,
            JSON.stringify(features)
        );

        if (status === 'dump') {
            this.#hivemind.dumpState();
        }

        return {
            entryPrice,
            sellPrice,
            stopLoss,
            multiplier,
            confidence,
            threshold: this.#config.baseConfidenceThreshold
        };
    }
}

export default NeuralSignalEngine;