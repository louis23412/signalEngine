import fs from 'fs';
import path from 'path';
import readline from 'readline';
import NeuralSignalEngine from '../src/neuralSignalEngine.js';

const engine = new NeuralSignalEngine();

const cacheSize = 1000;
const cache = [];
const signalTimes = [];

let signal;
let totalCandles = 0;
let totalLines = 0;
let signalCount = 0;
let trainingSteps = 0;
let gradientResetFreq;
let regulateFreq;

// ──────────────────────────────
// ADVANCED CONFIDENCE ANALYTICS
// ──────────────────────────────

// Lifetime running stats (O(1) memory)
let lifetimeCount = 0;
let lifetimeSum = 0;
let lifetimeSumSq = 0;
let allTimeMinConfidence = Infinity;
let allTimeMaxConfidence = -Infinity;

// Rolling windows: 100, 500, 1000, 5000
const windows = {
    100: [],
    500: [],
    1000: [],
    5000: []
};

const windowStats = {
    100: { mean: null, std: null, size: 0, trend: '—' },
    500: { mean: null, std: null, size: 0, trend: '—' },
    1000: { mean: null, std: null, size: 0, trend: '—' },
    5000: { mean: null, std: null, size: 0, trend: '—' }
};

const lifetimeStats = {
    min: null, max: null, mean: null, std: null, count: 0
};

const updateWindowStats = (size) => {
    const arr = windows[size];
    if (arr.length === 0) {
        windowStats[size] = { mean: null, std: null, size: 0, trend: '—' };
        return;
    }

    const sum = arr.reduce((a, b) => a + b, 0);
    const mean = sum / arr.length;
    const variance = arr.reduce((s, v) => s + (v - mean) ** 2, 0) / arr.length;
    const std = Math.sqrt(variance);

    // Trend: last 20% vs first 20%
    const segment = Math.max(5, Math.floor(arr.length * 0.2));
    const recentMean = arr.slice(-segment).reduce((a, b) => a + b, 0) / segment;
    const olderMean = arr.slice(0, segment).reduce((a, b) => a + b, 0) / segment;
    const diff = recentMean - olderMean;
    const trend = diff > 0.008 ? 'rising' : diff < -0.008 ? 'falling' : 'stable';

    windowStats[size] = {
        mean: mean.toFixed(6),
        std: std.toFixed(6),
        size: arr.length,
        trend
    };
};

const updateAllStats = () => {
    for (const size of Object.keys(windows)) {
        updateWindowStats(Number(size));
    }

    if (lifetimeCount > 0) {
        const mean = lifetimeSum / lifetimeCount;
        const variance = lifetimeCount > 1 ? (lifetimeSumSq / lifetimeCount) - (mean ** 2) : 0;
        lifetimeStats.mean = mean.toFixed(6);
        lifetimeStats.std = Math.sqrt(variance).toFixed(6);
        lifetimeStats.min = allTimeMinConfidence === Infinity ? null : allTimeMinConfidence.toFixed(6);
        lifetimeStats.max = allTimeMaxConfidence === -Infinity ? null : allTimeMaxConfidence.toFixed(6);
        lifetimeStats.count = lifetimeCount;
    }
};
// ──────────────────────────────

const trainingCutoff = null;
const shouldPredict = true;

const formatTime = (seconds) => {
    if (seconds < 1) return `${(seconds * 1000).toFixed(0)}ms`;
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    const ms = Math.floor((seconds - Math.floor(seconds)) * 1000);
    return `${h ? h + 'h ' : ''}${m ? m + 'm ' : ''}${s}s${ms > 0 ? '.' + ms : ''}`;
};

const countLines = () => {
    return new Promise((resolve) => {
        let lineCount = 0;
        const lineCounter = readline.createInterface({
            input: fs.createReadStream(path.join(import.meta.dirname, 'candles.jsonl'))
        });
        lineCounter.on('line', () => lineCount++);
        lineCounter.on('close', () => resolve(lineCount));
    });
};

const formatSignal = (options = {}) => {
    const {
        totalCandles = null,
        totalLines = null,
        durationSec = null,
        avgSignalTime = null,
        estimatedTimeSec = null
    } = options;

    const ANSI_CYAN = '\x1b[36m';
    const ANSI_GREEN = '\x1b[32m';
    const ANSI_YELLOW = '\x1b[33m';
    const ANSI_RED = '\x1b[31m';
    const ANSI_MAGENTA = '\x1b[35m';
    const ANSI_RESET = '\x1b[0m';
    const BOLD = '\x1b[1m';

    let progressLine = '';
    if (totalCandles !== null && totalLines !== null && durationSec !== null && avgSignalTime !== null && estimatedTimeSec !== null) {
        const progressPercent = ((totalCandles / totalLines) * 100).toFixed(4);
        progressLine = `${BOLD}Progress:${ANSI_RESET} ${ANSI_CYAN}${totalCandles}/${totalLines}${ANSI_RESET} candles ` +
                      `(${ANSI_CYAN}${progressPercent}%${ANSI_RESET}) | ` +
                      `Time: ${ANSI_CYAN}${formatTime(durationSec)}${ANSI_RESET} | ` +
                      `Avg: ${ANSI_CYAN}${avgSignalTime.toFixed(3)}s${ANSI_RESET} | ` +
                      `ETA: ${ANSI_CYAN}${formatTime(estimatedTimeSec)}${ANSI_RESET}\n`;
    }

    if (!shouldPredict || !signal) {
        process.stdout.write(`\n${'─'.repeat(92)}\n${progressLine}${'─'.repeat(92)}\n\n`);
        return;
    }

    const currentConf = signal.confidence?.toFixed(6) || '—';
    const recent1000 = windows[1000];
    const percentile = recent1000.length > 10
        ? (recent1000.filter(c => c <= signal.confidence).length / recent1000.length * 100).toFixed(1)
        : '—';

    const trendColor = (t) => t === 'rising' ? ANSI_GREEN : t === 'falling' ? ANSI_RED : ANSI_YELLOW;

    const signalLine = 
`${BOLD}Signal @ candle ${totalCandles}${ANSI_RESET}
  Entry : ${ANSI_CYAN}${signal.entryPrice}${ANSI_RESET}   Sell : ${ANSI_CYAN}${signal.sellPrice}${ANSI_RESET}   Stop : ${ANSI_CYAN}${signal.stopLoss}${ANSI_RESET}
  Mult  : ${ANSI_CYAN}${signal.multiplier.toFixed(3)}${ANSI_RESET}   Conf : ${ANSI_GREEN}${currentConf}${ANSI_RESET} ${ANSI_MAGENTA}(${percentile}%ile)${ANSI_RESET}   Step : ${ANSI_CYAN}${signal.lastTrainingStep}${ANSI_RESET}

${BOLD}LIFETIME CONFIDENCE (${lifetimeStats.count.toLocaleString()} signals)${ANSI_RESET}
  Range  : ${ANSI_RED}${lifetimeStats.min ?? '—'}${ANSI_RESET} → ${ANSI_GREEN}${lifetimeStats.max ?? '—'}${ANSI_RESET}
  Mean   : ${ANSI_YELLOW}${lifetimeStats.mean ?? '—'}${ANSI_RESET} ± ${ANSI_YELLOW}${lifetimeStats.std ?? '—'}${ANSI_RESET}

${BOLD}ROLLING CONFIDENCE WINDOWS${ANSI_RESET}
  100   → ${ANSI_YELLOW}${windowStats[100].mean ?? '—'}${ANSI_RESET} (σ:${windowStats[100].std ?? '—'}) [${trendColor(windowStats[100].trend)}${windowStats[100].trend}${ANSI_RESET}] ${ANSI_CYAN}${windowStats[100].size}/100${ANSI_RESET}
  500   → ${ANSI_YELLOW}${windowStats[500].mean ?? '—'}${ANSI_RESET} (σ:${windowStats[500].std ?? '—'}) [${trendColor(windowStats[500].trend)}${windowStats[500].trend}${ANSI_RESET}] ${ANSI_CYAN}${windowStats[500].size}/500${ANSI_RESET}
  1000  → ${ANSI_YELLOW}${windowStats[1000].mean ?? '—'}${ANSI_RESET} (σ:${windowStats[1000].std ?? '—'}) [${trendColor(windowStats[1000].trend)}${windowStats[1000].trend}${ANSI_RESET}] ${ANSI_CYAN}${windowStats[1000].size}/1000${ANSI_RESET}
  5000  → ${ANSI_YELLOW}${windowStats[5000].mean ?? '—'}${ANSI_RESET} (σ:${windowStats[5000].std ?? '—'}) [${trendColor(windowStats[5000].trend)}${windowStats[5000].trend}${ANSI_RESET}] ${ANSI_CYAN}${windowStats[5000].size}/5000${ANSI_RESET}
`;

    process.stdout.write(`\n${'─'.repeat(92)}\n${progressLine}${signalLine}${'─'.repeat(92)}\n\n`);
};

const processCandles = () => {
    const rd = readline.createInterface({
        input: fs.createReadStream(path.join(import.meta.dirname, 'candles.jsonl'))
    });

    // Set freq here
    const resetFreq = engine.getFreq();
    gradientResetFreq = resetFreq.gradientResetFreq
    regulateFreq = resetFreq.regulateFreq

    rd.on('line', (line) => {
        const candleObj = JSON.parse(line);
        cache.push(candleObj);
        if (cache.length > cacheSize) cache.shift();
        totalCandles++;

        if (cache.length >= cacheSize) {
            try {
                const status = totalCandles === totalLines;
                const startTime = process.hrtime.bigint();
                signal = engine.getSignal(cache, shouldPredict, status, trainingCutoff);
                const endTime = process.hrtime.bigint();
                const durationSec = Number(endTime - startTime) / 1_000_000_000;

                trainingSteps = signal.lastTrainingStep;

                // ADVANCED CONFIDENCE TRACKING
                if (shouldPredict && signal.confidence != null) {
                    const conf = signal.confidence;

                    // Lifetime updates
                    lifetimeCount++;
                    lifetimeSum += conf;
                    lifetimeSumSq += conf * conf;
                    if (conf < allTimeMinConfidence) allTimeMinConfidence = conf;
                    if (conf > allTimeMaxConfidence) allTimeMaxConfidence = conf;

                    // Rolling windows
                    for (const size of Object.keys(windows)) {
                        const win = windows[size];
                        win.push(conf);
                        if (win.length > Number(size)) win.shift();
                    }

                    updateAllStats();
                    signalCount++;
                }

                signalTimes.push(durationSec);
                if (signalTimes.length > 100) signalTimes.shift();

                const avgSignalTime = signalTimes.reduce((a, b) => a + b, 0) / signalTimes.length;
                const remainingCandles = totalLines - totalCandles;
                const estimatedTimeSec = remainingCandles * avgSignalTime;

                formatSignal({ totalCandles, totalLines, durationSec, avgSignalTime, estimatedTimeSec });

                if (trainingSteps === trainingCutoff) {
                    console.log("Training complete. Exiting.");
                    process.exit();
                }
            } catch (e) {
                console.error("Error during getSignal:", e);
                process.exit(1);
            }
        }
    });

    rd.on('close', () => {
        console.log(`\nCompleted. Total Candles Processed: ${totalCandles}`);
        if (shouldPredict && signalCount > 0) {
            formatSignal();
            console.log(`Lifetime Confidence → Min: ${allTimeMinConfidence.toFixed(6)}, Max: ${allTimeMaxConfidence.toFixed(6)}`);
        } else {
            console.log("Prediction was disabled — no confidence stats to display.");
        }
    });
};

countLines().then((lineCount) => {
    totalLines = lineCount;
    console.log(`Total lines to process: ${totalLines}`);
    processCandles();
});