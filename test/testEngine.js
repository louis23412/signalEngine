import fs from 'fs';
import path from 'path';
import readline from 'readline';
import NeuralSignalEngine from '../src/neuralSignalEngine.js';

const engine = new NeuralSignalEngine();

const trainingCutoff = null;
const shouldPredict = true;

const cacheSize = 1000;
const cache = [];
const signalTimes = [];

let signal;
let totalCandles = 0;
let totalLines = 0;
let signalCount = 0;
let trainingSteps = 0;
let gradientResetFreq = null;
let gradientResetStep = null;
let regulateFreq = null;
let regulateStep = null;

let lifetimeCount = 0;
let lifetimeSum = 0;
let lifetimeSumSq = 0;
let allTimeMinConfidence = Infinity;
let allTimeMaxConfidence = -Infinity;

let previousConfidence = null;

const windows = {
    10:    [],
    50:    [],
    100:   [],
    500:   [],
    1000:  [],
    5000:  [],
    10000: [],
    25000: []
};

const windowStats = {
    10:    { mean: null, std: null, min: null, max: null, size: 0, trend: '—' },
    50:    { mean: null, std: null, min: null, max: null, size: 0, trend: '—' },
    100:   { mean: null, std: null, min: null, max: null, size: 0, trend: '—' },
    500:   { mean: null, std: null, min: null, max: null, size: 0, trend: '—' },
    1000:  { mean: null, std: null, min: null, max: null, size: 0, trend: '—' },
    5000:  { mean: null, std: null, min: null, max: null, size: 0, trend: '—' },
    10000: { mean: null, std: null, min: null, max: null, size: 0, trend: '—' },
    25000: { mean: null, std: null, min: null, max: null, size: 0, trend: '—' }
};

const lifetimeStats = {
    min: null, max: null, mean: null, std: null, count: 0
};

const updateWindowStats = (size) => {
    const arr = windows[size];
    if (arr.length === 0) {
        windowStats[size] = { mean: null, std: null, min: null, max: null, size: 0, trend: '—' };
        return;
    }

    const sum = arr.reduce((a, b) => a + b, 0);
    const mean = sum / arr.length;
    const variance = arr.reduce((s, v) => s + (v - mean) ** 2, 0) / arr.length;
    const std = Math.sqrt(variance);
    const min = Math.min(...arr);
    const max = Math.max(...arr);

    const segment = Math.max(5, Math.floor(arr.length * 0.2));
    const recentMean = arr.slice(-segment).reduce((a, b) => a + b, 0) / segment;
    const olderMean = arr.slice(0, segment).reduce((a, b) => a + b, 0) / segment;
    const diff = recentMean - olderMean;
    const trend = diff > 0.008 ? 'rising' : diff < -0.008 ? 'falling' : 'stable';

    windowStats[size] = {
        mean: mean.toFixed(6),
        std: std.toFixed(6),
        min: min.toFixed(6),
        max: max.toFixed(6),
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

    const CYAN = '\x1b[36m';
    const GREEN = '\x1b[32m';
    const YELLOW = '\x1b[33m';
    const RED = '\x1b[31m';
    const MAGENTA = '\x1b[35m';
    const RESET = '\x1b[0m';
    const BOLD = '\x1b[1m';

    let progressLine = '';
    if (totalCandles !== null && totalLines !== null && durationSec !== null && avgSignalTime !== null && estimatedTimeSec !== null) {
        const progressPercent = ((totalCandles / totalLines) * 100).toFixed(4);
        progressLine = `${BOLD}Progress:${RESET} ${CYAN}${totalCandles.toLocaleString()}/${totalLines.toLocaleString()}${RESET} candles ` +
                      `(${CYAN}${progressPercent}%${RESET}) | ` +
                      `Time: ${CYAN}${formatTime(durationSec)}${RESET} | ` +
                      `Avg: ${CYAN}${avgSignalTime.toFixed(3)}s${RESET} | ` +
                      `ETA: ${CYAN}${formatTime(estimatedTimeSec)}${RESET}\n`;
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

    const delta = previousConfidence !== null ? signal.confidence - previousConfidence : null;
    const deltaStr = delta === null ? '—' : delta > 0 ? `+${delta.toFixed(6)}` : delta.toFixed(6);
    const deltaColor = delta === null ? '' : delta > 0 ? GREEN : RED;

    previousConfidence = signal.confidence;

    const trendColor = (t) => t === 'rising' ? GREEN : t === 'falling' ? RED : YELLOW;

    const stepsToNextRegulate = (regulateFreq > 0)
        ? regulateFreq - (trainingSteps % regulateFreq)
        : null;

    const stepsToNextGradientReset = (gradientResetFreq > 0)
        ? gradientResetFreq - (trainingSteps % gradientResetFreq)
        : null;

    const signalLine = 
`
${BOLD}Signal @ candle ${totalCandles.toLocaleString()}${RESET}
  Entry : ${CYAN}${signal.entryPrice}${RESET}   Sell : ${CYAN}${signal.sellPrice}${RESET}   Stop : ${CYAN}${signal.stopLoss}${RESET}
  Mult  : ${CYAN}${signal.multiplier.toFixed(3)}${RESET}   Conf : ${GREEN}${currentConf}${RESET} ${deltaColor}(${deltaStr})${RESET} ${MAGENTA}(${percentile}%ile)${RESET}   Step : ${CYAN}${signal.lastTrainingStep.toLocaleString()}${RESET}

  Regulate every ${YELLOW}${regulateFreq ?? '—'}${RESET} → last: ${CYAN}${regulateStep ?? '—'}${RESET}   next in ${MAGENTA}${stepsToNextRegulate ?? '—'}${RESET} steps
  Gradient reset ${YELLOW}${gradientResetFreq ?? '—'}${RESET} → last: ${CYAN}${gradientResetStep ?? '—'}${RESET}   next in ${MAGENTA}${stepsToNextGradientReset ?? '—'}${RESET} steps

${BOLD}LIFETIME CONFIDENCE (${lifetimeStats.count.toLocaleString()} signals)${RESET}
  Range : ${RED}${lifetimeStats.min ?? '—'}${RESET} → ${GREEN}${lifetimeStats.max ?? '—'}${RESET}   Mean  : ${YELLOW}${lifetimeStats.mean ?? '—'}${RESET} ± ${YELLOW}${lifetimeStats.std ?? '—'}${RESET}

${BOLD}ROLLING CONFIDENCE WINDOWS${RESET}
   10   → ${YELLOW}${windowStats[10].mean ?? '—'}${RESET} ±${windowStats[10].std ?? '—'}  [${RED}${windowStats[10].min ?? '—'}${RESET} → ${GREEN}${windowStats[10].max ?? '—'}${RESET}] (${CYAN}${windowStats[10].size}/10${RESET})    [${trendColor(windowStats[10].trend)}${windowStats[10].trend}${RESET}]
   50   → ${YELLOW}${windowStats[50].mean ?? '—'}${RESET} ±${windowStats[50].std ?? '—'}  [${RED}${windowStats[50].min ?? '—'}${RESET} → ${GREEN}${windowStats[50].max ?? '—'}${RESET}] (${CYAN}${windowStats[50].size}/50${RESET})    [${trendColor(windowStats[50].trend)}${windowStats[50].trend}${RESET}]
  100   → ${YELLOW}${windowStats[100].mean ?? '—'}${RESET} ±${windowStats[100].std ?? '—'}  [${RED}${windowStats[100].min ?? '—'}${RESET} → ${GREEN}${windowStats[100].max ?? '—'}${RESET}] (${CYAN}${windowStats[100].size}/100${RESET})   [${trendColor(windowStats[100].trend)}${windowStats[100].trend}${RESET}]
  500   → ${YELLOW}${windowStats[500].mean ?? '—'}${RESET} ±${windowStats[500].std ?? '—'}  [${RED}${windowStats[500].min ?? '—'}${RESET} → ${GREEN}${windowStats[500].max ?? '—'}${RESET}] (${CYAN}${windowStats[500].size}/500${RESET})   [${trendColor(windowStats[500].trend)}${windowStats[500].trend}${RESET}]
 1000   → ${YELLOW}${windowStats[1000].mean ?? '—'}${RESET} ±${windowStats[1000].std ?? '—'}  [${RED}${windowStats[1000].min ?? '—'}${RESET} → ${GREEN}${windowStats[1000].max ?? '—'}${RESET}] (${CYAN}${windowStats[1000].size}/1000${RESET})  [${trendColor(windowStats[1000].trend)}${windowStats[1000].trend}${RESET}]
 5000   → ${YELLOW}${windowStats[5000].mean ?? '—'}${RESET} ±${windowStats[5000].std ?? '—'}  [${RED}${windowStats[5000].min ?? '—'}${RESET} → ${GREEN}${windowStats[5000].max ?? '—'}${RESET}] (${CYAN}${windowStats[5000].size}/5000${RESET})  [${trendColor(windowStats[5000].trend)}${windowStats[5000].trend}${RESET}]
10000   → ${YELLOW}${windowStats[10000].mean ?? '—'}${RESET} ±${windowStats[10000].std ?? '—'}  [${RED}${windowStats[10000].min ?? '—'}${RESET} → ${GREEN}${windowStats[10000].max ?? '—'}${RESET}] (${CYAN}${windowStats[10000].size}/10000${RESET}) [${trendColor(windowStats[10000].trend)}${windowStats[10000].trend}${RESET}]
25000   → ${YELLOW}${windowStats[25000].mean ?? '—'}${RESET} ±${windowStats[25000].std ?? '—'}  [${RED}${windowStats[25000].min ?? '—'}${RESET} → ${GREEN}${windowStats[25000].max ?? '—'}${RESET}] (${CYAN}${windowStats[25000].size}/25000${RESET}) [${trendColor(windowStats[25000].trend)}${windowStats[25000].trend}${RESET}]
`;

    process.stdout.write(`\n${'─'.repeat(92)}\n${progressLine}${signalLine}${'─'.repeat(92)}\n\n`);
};

const processCandles = () => {
    const rd = readline.createInterface({
        input: fs.createReadStream(path.join(import.meta.dirname, 'candles.jsonl'))
    });

    const freqs = engine.getFreq();
    gradientResetFreq = freqs.gradientResetFreq;
    regulateFreq = freqs.regulateFreq;

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
                gradientResetStep = signal.lastGradientResetStep;
                regulateStep = signal.lastRegulateStep;

                if (shouldPredict && signal.confidence != null) {
                    const conf = signal.confidence;

                    lifetimeCount++;
                    lifetimeSum += conf;
                    lifetimeSumSq += conf * conf;
                    if (conf < allTimeMinConfidence) allTimeMinConfidence = conf;
                    if (conf > allTimeMaxConfidence) allTimeMaxConfidence = conf;

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