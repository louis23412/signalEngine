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

let currentStep = -1;
let candlesSinceStepIncrease = 0;
let minConfidenceInCurrentStep = Infinity;
let maxConfidenceInCurrentStep = -Infinity;

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
let maxRangeInStep = 0;
let maxRangeStep = null;
let maxRangeAtGradientStep = null;
let maxRangeAtRegulateStep = null;

const windows = {
    10: [], 50: [], 100: [], 500: [], 1000: [], 5000: [], 10000: [], 25000: []
};

const windowStats = {
    10:   { mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: '—' },
    50:   { mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: '—' },
    100:  { mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: '—' },
    500:  { mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: '—' },
    1000: { mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: '—' },
    5000: { mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: '—' },
    10000:{ mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: '—' },
    25000:{ mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: '—' }
};

const lifetimeStats = { min: null, max: null, mean: null, std: null, count: 0 };

const formatWindow = (lastResetStep, freq) => {
    if (freq === null || freq <= 0) return '—';
    const start = (lastResetStep === null || lastResetStep <=  0) ? 0 : lastResetStep;
    const end = start + freq;
    return `${start.toLocaleString()} – ${end.toLocaleString()}`;
};

const updateWindowStats = (size) => {
    const arr = windows[size];
    if (arr.length === 0) {
        windowStats[size] = { mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: '—' };
        return;
    }
    const sum = arr.reduce((a, b) => a + b, 0);
    const mean = sum / arr.length;
    const variance = arr.reduce((s, v) => s + (v - mean) ** 2, 0) / arr.length;
    const std = Math.sqrt(variance);
    const min = Math.min(...arr);
    const max = Math.max(...arr);
    const diff = max - min;

    const segment = Math.max(5, Math.floor(arr.length * 0.2));
    const recentMean = arr.slice(-segment).reduce((a, b) => a + b, 0) / segment;
    const olderMean = arr.slice(0, segment).reduce((a, b) => a + b, 0) / segment;
    const trendDiff = recentMean - olderMean;
    const trend = trendDiff > 0.008 ? 'rising' : trendDiff < -0.008 ? 'falling' : 'stable';

    windowStats[size] = {
        mean: mean.toFixed(6),
        std: std.toFixed(6),
        min: min.toFixed(6),
        max: max.toFixed(6),
        diff: diff.toFixed(6),
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

const formatSignal = ({ totalCandles, totalLines, durationSec, avgSignalTime, estimatedTimeSec }) => {
    const C = '\x1b[36m';
    const G = '\x1b[32m';
    const Y = '\x1b[33m';
    const R = '\x1b[31m';
    const M = '\x1b[35m';
    const B = '\x1b[1m';
    const X = '\x1b[0m';

    let progressLine = '';
    if (totalCandles != null && totalLines != null) {
        const pct = ((totalCandles / totalLines) * 100).toFixed(3);
        progressLine = `${B}Progress:${X} ${C}${totalCandles.toLocaleString()}${X}/${C}${totalLines.toLocaleString()}${X} candles (${C}${pct}%${X}) | ` +
                       `Time: ${C}${formatTime(durationSec)}${X} | Avg: ${C}${avgSignalTime.toFixed(3)}s${X} | ETA: ${C}${formatTime(estimatedTimeSec)}${X}\n`;
    }

    if (!shouldPredict || !signal) {
        process.stdout.write(`\n${'─'.repeat(100)}\n${progressLine}${'─'.repeat(100)}\n\n`);
        return;
    }

    const conf = signal.confidence?.toFixed(6) || '—';
    const percentile = windows[1000].length > 10
        ? (windows[1000].filter(c => c <= signal.confidence).length / windows[1000].length * 100).toFixed(1)
        : '—';

    const delta = previousConfidence !== null ? signal.confidence - previousConfidence : null;
    const deltaStr = delta === null ? '—' : delta > 0 ? `+${delta.toFixed(6)}` : delta.toFixed(6);
    const deltaCol = delta === null ? '' : delta > 0 ? G : R;
    previousConfidence = signal.confidence;

    const trendCol = (t) => t === 'rising' ? G : t === 'falling' ? R : Y;

    const stepsToNextRegulate = regulateFreq > 0 ? regulateFreq - (trainingSteps % regulateFreq) : null;
    const stepsToNextGradientReset = gradientResetFreq > 0 ? gradientResetFreq - (trainingSteps % gradientResetFreq) : null;

    const stepDiff = (maxConfidenceInCurrentStep - minConfidenceInCurrentStep).toFixed(6);

    const signalLine = `
${B}Signal @ candle ${totalCandles.toLocaleString()}${X}
  Entry : ${C}${signal.entryPrice}${X}   Sell : ${C}${signal.sellPrice}${X}   Stop : ${C}${signal.stopLoss}${X}
  Mult  : ${C}${signal.multiplier.toFixed(3)}${X}   Conf : ${G}${conf}${X} ${deltaCol}(${deltaStr})${X} ${M}(${percentile}%ile)${X}   Step : ${C}${signal.lastTrainingStep.toLocaleString()}${X}
  Regulate every ${Y}${regulateFreq ?? '—'}${X} → last: ${C}${regulateStep ?? '—'}${X} next in ${M}${stepsToNextRegulate ?? '—'}${X} steps
  Gradient reset ${Y}${gradientResetFreq ?? '—'}${X} → last: ${C}${gradientResetStep ?? '—'}${X} next in ${M}${stepsToNextGradientReset ?? '—'}${X} steps

${B}STEP STABILITY WINDOW${X}
  Candles since last step increase : ${M}${candlesSinceStepIncrease.toLocaleString()}${X}
  Confidence range in current step : ${R}${minConfidenceInCurrentStep === Infinity ? '—' : minConfidenceInCurrentStep.toFixed(6)}${X} → ${G}${maxConfidenceInCurrentStep === -Infinity ? '—' : maxConfidenceInCurrentStep.toFixed(6)}${X}  ${M}Δ${stepDiff}${X}

  Largest range ever in a stable window : ${G}${maxRangeInStep === 0 ? '—' : maxRangeInStep.toFixed(6)}${X}
        └─ at training step ${C}${maxRangeStep ?? '—'}${X}
           in gradient reset window ${Y}${formatWindow(maxRangeAtGradientStep, gradientResetFreq)}${X} / regulate window ${Y}${formatWindow(maxRangeAtRegulateStep, regulateFreq)}${X}

${B}LIFETIME CONFIDENCE${X} (${C}${lifetimeStats.count.toLocaleString()}${X} signals)
  Range : ${R}${lifetimeStats.min ?? '—'}${X} → ${G}${lifetimeStats.max ?? '—'}${X}   Mean : ${Y}${lifetimeStats.mean ?? '—'}${X} ± ${Y}${lifetimeStats.std ?? '—'}${X}

${B}ROLLING CONFIDENCE WINDOWS${X}
   10 → ${Y}${windowStats[10].mean ?? '—'}${X} ±${Y}${windowStats[10].std ?? '—'}${X} [${R}${windowStats[10].min ?? '—'}${X} → ${G}${windowStats[10].max ?? '—'}${X}] ${M}Δ${windowStats[10].diff ?? '—'}${X} (${C}${windowStats[10].size}/10${X}) [${trendCol(windowStats[10].trend)}${windowStats[10].trend}${X}]
   50 → ${Y}${windowStats[50].mean ?? '—'}${X} ±${Y}${windowStats[50].std ?? '—'}${X} [${R}${windowStats[50].min ?? '—'}${X} → ${G}${windowStats[50].max ?? '—'}${X}] ${M}Δ${windowStats[50].diff ?? '—'}${X} (${C}${windowStats[50].size}/50${X}) [${trendCol(windowStats[50].trend)}${windowStats[50].trend}${X}]
  100 → ${Y}${windowStats[100].mean ?? '—'}${X} ±${Y}${windowStats[100].std ?? '—'}${X} [${R}${windowStats[100].min ?? '—'}${X} → ${G}${windowStats[100].max ?? '—'}${X}] ${M}Δ${windowStats[100].diff ?? '—'}${X} (${C}${windowStats[100].size}/100${X}) [${trendCol(windowStats[100].trend)}${windowStats[100].trend}${X}]
  500 → ${Y}${windowStats[500].mean ?? '—'}${X} ±${Y}${windowStats[500].std ?? '—'}${X} [${R}${windowStats[500].min ?? '—'}${X} → ${G}${windowStats[500].max ?? '—'}${X}] ${M}Δ${windowStats[500].diff ?? '—'}${X} (${C}${windowStats[500].size}/500${X}) [${trendCol(windowStats[500].trend)}${windowStats[500].trend}${X}]
 1000 → ${Y}${windowStats[1000].mean ?? '—'}${X} ±${Y}${windowStats[1000].std ?? '—'}${X} [${R}${windowStats[1000].min ?? '—'}${X} → ${G}${windowStats[1000].max ?? '—'}${X}] ${M}Δ${windowStats[1000].diff ?? '—'}${X} (${C}${windowStats[1000].size}/1000${X}) [${trendCol(windowStats[1000].trend)}${windowStats[1000].trend}${X}]
 5000 → ${Y}${windowStats[5000].mean ?? '—'}${X} ±${Y}${windowStats[5000].std ?? '—'}${X} [${R}${windowStats[5000].min ?? '—'}${X} → ${G}${windowStats[5000].max ?? '—'}${X}] ${M}Δ${windowStats[5000].diff ?? '—'}${X} (${C}${windowStats[5000].size}/5000${X}) [${trendCol(windowStats[5000].trend)}${windowStats[5000].trend}${X}]
10000 → ${Y}${windowStats[10000].mean ?? '—'}${X} ±${Y}${windowStats[10000].std ?? '—'}${X} [${R}${windowStats[10000].min ?? '—'}${X} → ${G}${windowStats[10000].max ?? '—'}${X}] ${M}Δ${windowStats[10000].diff ?? '—'}${X} (${C}${windowStats[10000].size}/10000${X}) [${trendCol(windowStats[10000].trend)}${windowStats[10000].trend}${X}]
25000 → ${Y}${windowStats[25000].mean ?? '—'}${X} ±${Y}${windowStats[25000].std ?? '—'}${X} [${R}${windowStats[25000].min ?? '—'}${X} → ${G}${windowStats[25000].max ?? '—'}${X}] ${M}Δ${windowStats[25000].diff ?? '—'}${X} (${C}${windowStats[25000].size}/25000${X}) [${trendCol(windowStats[25000].trend)}${windowStats[25000].trend}${X}]
`;

    process.stdout.write(`\n${'─'.repeat(100)}\n${progressLine}${signalLine}${'─'.repeat(100)}\n\n`);
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

                    if (currentStep === -1) currentStep = trainingSteps;

                    if (trainingSteps > currentStep) {
                        candlesSinceStepIncrease = 0;
                        minConfidenceInCurrentStep = conf;
                        maxConfidenceInCurrentStep = conf;
                        currentStep = trainingSteps;
                    } else {
                        candlesSinceStepIncrease++;
                        if (conf < minConfidenceInCurrentStep) minConfidenceInCurrentStep = conf;
                        if (conf > maxConfidenceInCurrentStep) maxConfidenceInCurrentStep = conf;
                    }

                    const rangeInCurrentStep = maxConfidenceInCurrentStep - minConfidenceInCurrentStep;

                    if (rangeInCurrentStep > maxRangeInStep) {
                        maxRangeInStep = rangeInCurrentStep;
                        maxRangeStep = currentStep;
                        maxRangeAtGradientStep = gradientResetStep;
                        maxRangeAtRegulateStep = regulateStep;
                    }

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
                    process.exit(0);
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
            formatSignal({ totalCandles, totalLines });
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