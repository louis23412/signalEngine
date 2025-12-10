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
let confidencePathInStep = [];

let maxRangeInStep = 0;
let maxRangeStep = null;
let maxRangeStableSteps = 0;
let maxRangeAtGradientStep = null;
let maxRangeAtRegulateStep = null;

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
    10: [], 50: [], 100: [], 500: [], 1000: [], 5000: [], 10000: [], 25000: []
};

const windowStats = {
    10:   { mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: 0, healthScore: '—' },
    50:   { mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: 0, healthScore: '—' },
    100:  { mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: 0, healthScore: '—' },
    500:  { mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: 0, healthScore: '—' },
    1000: { mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: 0, healthScore: '—' },
    5000: { mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: 0, healthScore: '—' },
    10000:{ mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: 0, healthScore: '—' },
    25000:{ mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: 0, healthScore: '—' }
};

const lifetimeStats = { min: null, max: null, mean: null, std: null, count: 0 };

const formatWindow = (lastResetStep, freq) => {
    if (freq === null || freq <= 0) return '—';
    const start = lastResetStep ?? 0;
    const end = start + freq;
    return `${start.toLocaleString()} – ${end.toLocaleString()}`;
};

const computeHealthScore = (arr, windowSize = null) => {
    if (!arr || arr.length < 3) return '—';

    let totalAbsChange = 0;
    let oscillations = 0;
    let prevDiff = null;

    for (let i = 1; i < arr.length; i++) {
        const diff = arr[i] - arr[i - 1];
        totalAbsChange += Math.abs(diff);
        if (prevDiff !== null && Math.sign(diff) !== Math.sign(prevDiff) && Math.abs(diff) > 1e-8) {
            oscillations++;
        }
        prevDiff = diff;
    }

    const n = arr.length - 1;
    const avgAbsChange = totalAbsChange / n;
    const meanConfidence = arr.reduce((a, b) => a + b, 0) / arr.length;

    const relativeChange = meanConfidence > 0 ? avgAbsChange / meanConfidence : avgAbsChange;
    const smoothness = Math.max(0, 1 - relativeChange * 9);

    const maxReasonableOsc = Math.max(1, n * 0.15);
    const oscillationPenalty = 1 - Math.min(oscillations / maxReasonableOsc, 0.9);

    const variance = arr.reduce((a, v) => a + (v - meanConfidence) ** 2, 0) / arr.length;
    const stdRel = meanConfidence > 0 ? Math.sqrt(variance) / meanConfidence : 1;
    const stabilityBonus = meanConfidence > 0.3 ? Math.max(0, 1 - stdRel * 5) : 0;

    let score = smoothness * 0.55 + oscillationPenalty * 0.30 + stabilityBonus * 0.15;

    if (windowSize && windowSize <= 50) score = Math.min(1, score * 1.08);

    return Math.max(0, Math.min(1, score)).toFixed(3);
};

const colorHealthPct = (score) => {
    const G = '\x1b[32m';
    const Y = '\x1b[33m';
    const R = '\x1b[31m';
    const X = '\x1b[0m';
    if (score === '—') return `${Y}—${X}`;
    const pct = (parseFloat(score) * 100).toFixed(1);
    const col = parseFloat(score) >= 0.80 ? G : parseFloat(score) >= 0.60 ? Y : R;
    return `${col}${pct}%${X}`;
};

const colorMeanByTrend = (mean, trend) => {
    const G = '\x1b[32m';
    const R = '\x1b[31m';
    const Y = '\x1b[33m';
    const X = '\x1b[0m';
    if (mean === null) return `${Y}—${X}`;
    const col = trend > 0 ? G : trend < 0 ? R : Y;
    return `${col}${mean}${X}`;
};

const updateWindowStats = (size) => {
    const arr = windows[size];
    if (arr.length === 0) {
        windowStats[size] = { mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: 0, healthScore: '—' };
        return;
    }

    const sum = arr.reduce((a, b) => a + b, 0);
    const mean = sum / arr.length;
    const variance = arr.reduce((a, v) => a + (v - mean) ** 2, 0) / arr.length;
    const std = Math.sqrt(variance);
    const min = Math.min(...arr);
    const max = Math.max(...arr);
    const diff = max - min;

    const segment = Math.max(5, Math.floor(arr.length * 0.2));
    const recent = arr.slice(-segment).reduce((a, b) => a + b, 0) / segment;
    const older = arr.slice(0, segment).reduce((a, b) => a + b, 0) / segment;
    const trendDiff = recent - older;
    const trend = trendDiff > 0.008 ? 1 : trendDiff < -0.008 ? -1 : 0;

    const healthScore = computeHealthScore(arr, Number(size));

    windowStats[size] = {
        mean: mean.toFixed(6),
        std: std.toFixed(6),
        min: min.toFixed(6),
        max: max.toFixed(6),
        diff: diff.toFixed(6),
        size: arr.length,
        trend,
        healthScore
    };
};

const updateAllStats = () => {
    for (const size of Object.keys(windows)) updateWindowStats(Number(size));

    if (lifetimeCount > 0) {
        const mean = lifetimeSum / lifetimeCount;
        const variance = lifetimeCount > 1 ? (lifetimeSumSq / lifetimeCount) - mean ** 2 : 0;
        lifetimeStats.mean = mean.toFixed(6);
        lifetimeStats.std = Math.sqrt(variance).toFixed(6);
        lifetimeStats.min = allTimeMinConfidence === Infinity ? null : allTimeMinConfidence.toFixed(6);
        lifetimeStats.max = allTimeMaxConfidence === -Infinity ? null : allTimeMaxConfidence.toFixed(6);
        lifetimeStats.count = lifetimeCount;
    }
};

const formatTime = (sec) => {
    if (sec < 1) return `${(sec * 1000).toFixed(0)}ms`;
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = Math.floor(sec % 60);
    return `${h ? h + 'h ' : ''}${m ? m + 'm ' : ''}${s}s`;
};

const countLines = () => new Promise(resolve => {
    let count = 0;
    const rl = readline.createInterface({
        input: fs.createReadStream(path.join(import.meta.dirname, 'candles.jsonl'))
    });
    rl.on('line', () => count++);
    rl.on('close', () => resolve(count));
});

const formatSignal = ({ totalCandles, totalLines, durationSec, avgSignalTime, estimatedTimeSec }) => {
    const C = '\x1b[36m';
    const G = '\x1b[32m';
    const Y = '\x1b[33m';
    const R = '\x1b[31m';
    const M = '\x1b[35m';
    const B = '\x1b[1m';
    const X = '\x1b[0m';

    process.stdout.write('\x1b[2J\x1b[0f');

    const pct = totalLines ? ((totalCandles / totalLines) * 100).toFixed(3) : '0';
    const progressLine = `${B}Progress:${X} ${C}${totalCandles.toLocaleString()}${X}/${C}${totalLines.toLocaleString()}${X} (${C}${pct}%${X}) | Time: ${C}${formatTime(durationSec)}${X} | Avg: ${C}${avgSignalTime.toFixed(3)}s${X} | ETA: ${C}${formatTime(estimatedTimeSec)}${X}\n`;

    if (!shouldPredict || !signal) {
        process.stdout.write(`\n${'─'.repeat(100)}\n${progressLine}${'─'.repeat(100)}\n\n`);
        return;
    }

    const conf = signal.confidence?.toFixed(6) ?? '—';
    const percentile = windows[1000].length > 10
        ? (windows[1000].filter(c => c <= signal.confidence).length / windows[1000].length * 100).toFixed(1)
        : '—';

    const delta = previousConfidence !== null ? signal.confidence - previousConfidence : null;
    const deltaStr = delta === null ? '—' : delta > 0 ? `+${delta.toFixed(6)}` : delta.toFixed(6);
    const deltaCol = delta === null ? '' : delta > 0 ? G : R;
    previousConfidence = signal.confidence;

    const stepDiff = (maxConfidenceInCurrentStep - minConfidenceInCurrentStep).toFixed(6);
    const currentHealthScore = computeHealthScore(confidencePathInStep, confidencePathInStep.length);

    const signalLine = `
${B}Signal${X}
  Training Step : ${C}${signal.lastTrainingStep.toLocaleString()}${X}
  Entry : ${C}${signal.entryPrice}${X}   Sell : ${C}${signal.sellPrice}${X}   Stop : ${C}${signal.stopLoss}${X}
  Mult  : ${C}${signal.multiplier.toFixed(3)}${X}   Conf : ${C}${conf}${X} ${deltaCol}(${deltaStr})${X} ${M}(${percentile}%ile)${X}
  Prediction Accuracy : ${C}${signal.globalAccuracy}${X}${signal.globalAccuracy !== 'disabled' ? '%' : ''} Open simulations : ${C}${signal.openSimulations}${X}

  Regulate every ${Y}${regulateFreq ?? '—'}${X} steps → last: ${C}${regulateStep ?? '—'}${X} next in ${M}${regulateFreq ? regulateFreq - (trainingSteps % regulateFreq) : '—'}${X}
  Gradient reset ${Y}${gradientResetFreq ?? '—'}${X} steps → last: ${C}${gradientResetStep ?? '—'}${X} next in ${M}${gradientResetFreq ? gradientResetFreq - (trainingSteps % gradientResetFreq) : '—'}${X}

${B}STEP STABILITY WINDOW${X}
  Candles in current step    : ${M}${candlesSinceStepIncrease.toLocaleString()}${X}
  Confidence range (current) : ${R}${minConfidenceInCurrentStep.toFixed(6)}${X} → ${G}${maxConfidenceInCurrentStep.toFixed(6)}${X}   ${M}Δ${stepDiff}${X}
  Health                     : ${colorHealthPct(currentHealthScore)}

  Largest range in stable window  : ${G}${maxRangeInStep === 0 ? '—' : maxRangeInStep.toFixed(6)}${X}
    └─ over ${C}${maxRangeStableSteps.toLocaleString()}${X} candles at step ${C}${maxRangeStep ?? '—'}${X} (gradient ${Y}${formatWindow(maxRangeAtGradientStep, gradientResetFreq)}${X} / regulate ${Y}${formatWindow(maxRangeAtRegulateStep, regulateFreq)}${X})

${B}LIFETIME CONFIDENCE${X}
  Range : ${R}${lifetimeStats.min ?? '—'}${X} → ${G}${lifetimeStats.max ?? '—'}${X}   Mean : ${Y}${lifetimeStats.mean ?? '—'}${X} ± ${Y}${lifetimeStats.std ?? '—'}${X}

${B}ROLLING CONFIDENCE WINDOWS${X}
  10  : ${colorMeanByTrend(windowStats[10].mean, windowStats[10].trend)} ± ${Y}${windowStats[10].std ?? '—'}${X} [${R}${windowStats[10].min ?? '—'}${X} → ${G}${windowStats[10].max ?? '—'}${X}] ${M}Δ${windowStats[10].diff ?? '—'}${X} H: ${colorHealthPct(windowStats[10].healthScore)}
  50  : ${colorMeanByTrend(windowStats[50].mean, windowStats[50].trend)} ± ${Y}${windowStats[50].std ?? '—'}${X} [${R}${windowStats[50].min ?? '—'}${X} → ${G}${windowStats[50].max ?? '—'}${X}] ${M}Δ${windowStats[50].diff ?? '—'}${X} H: ${colorHealthPct(windowStats[50].healthScore)}
  100 : ${colorMeanByTrend(windowStats[100].mean, windowStats[100].trend)} ± ${Y}${windowStats[100].std ?? '—'}${X} [${R}${windowStats[100].min ?? '—'}${X} → ${G}${windowStats[100].max ?? '—'}${X}] ${M}Δ${windowStats[100].diff ?? '—'}${X} H: ${colorHealthPct(windowStats[100].healthScore)}
  500 : ${colorMeanByTrend(windowStats[500].mean, windowStats[500].trend)} ± ${Y}${windowStats[500].std ?? '—'}${X} [${R}${windowStats[500].min ?? '—'}${X} → ${G}${windowStats[500].max ?? '—'}${X}] ${M}Δ${windowStats[500].diff ?? '—'}${X} H: ${colorHealthPct(windowStats[500].healthScore)}
  1K  : ${colorMeanByTrend(windowStats[1000].mean, windowStats[1000].trend)} ± ${Y}${windowStats[1000].std ?? '—'}${X} [${R}${windowStats[1000].min ?? '—'}${X} → ${G}${windowStats[1000].max ?? '—'}${X}] ${M}Δ${windowStats[1000].diff ?? '—'}${X} H: ${colorHealthPct(windowStats[1000].healthScore)}
  5K  : ${colorMeanByTrend(windowStats[5000].mean, windowStats[5000].trend)} ± ${Y}${windowStats[5000].std ?? '—'}${X} [${R}${windowStats[5000].min ?? '—'}${X} → ${G}${windowStats[5000].max ?? '—'}${X}] ${M}Δ${windowStats[5000].diff ?? '—'}${X} H: ${colorHealthPct(windowStats[5000].healthScore)}
  10K : ${colorMeanByTrend(windowStats[10000].mean, windowStats[10000].trend)} ± ${Y}${windowStats[10000].std ?? '—'}${X} [${R}${windowStats[10000].min ?? '—'}${X} → ${G}${windowStats[10000].max ?? '—'}${X}] ${M}Δ${windowStats[10000].diff ?? '—'}${X} H: ${colorHealthPct(windowStats[10000].healthScore)}
  25K : ${colorMeanByTrend(windowStats[25000].mean, windowStats[25000].trend)} ± ${Y}${windowStats[25000].std ?? '—'}${X} [${R}${windowStats[25000].min ?? '—'}${X} → ${G}${windowStats[25000].max ?? '—'}${X}] ${M}Δ${windowStats[25000].diff ?? '—'}${X} H: ${colorHealthPct(windowStats[25000].healthScore)}
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

    rd.on('line', line => {
        const candle = JSON.parse(line);
        cache.push(candle);
        if (cache.length > cacheSize) cache.shift();
        totalCandles++;

        if (cache.length >= cacheSize) {
            try {
                const isLast = totalCandles === totalLines;
                const start = process.hrtime.bigint();
                signal = engine.getSignal(cache, shouldPredict, isLast, trainingCutoff);
                const end = process.hrtime.bigint();
                const durationSec = Number(end - start) / 1e9;

                trainingSteps = signal.lastTrainingStep;
                gradientResetStep = signal.lastGradientResetStep;
                regulateStep = signal.lastRegulateStep;

                if (shouldPredict && signal.confidence != null) {
                    const conf = signal.confidence;

                    if (currentStep === -1) currentStep = trainingSteps;

                    if (trainingSteps > currentStep) {
                        const range = maxConfidenceInCurrentStep - minConfidenceInCurrentStep;
                        if (range > maxRangeInStep) {
                            maxRangeInStep = range;
                            maxRangeStep = currentStep;
                            maxRangeStableSteps = candlesSinceStepIncrease;
                            maxRangeAtGradientStep = gradientResetStep;
                            maxRangeAtRegulateStep = regulateStep;
                        }

                        candlesSinceStepIncrease = 0;
                        minConfidenceInCurrentStep = conf;
                        maxConfidenceInCurrentStep = conf;
                        confidencePathInStep = [conf];
                        currentStep = trainingSteps;
                    } else {
                        candlesSinceStepIncrease++;
                        minConfidenceInCurrentStep = Math.min(minConfidenceInCurrentStep, conf);
                        maxConfidenceInCurrentStep = Math.max(maxConfidenceInCurrentStep, conf);
                        confidencePathInStep.push(conf);
                    }

                    lifetimeCount++;
                    lifetimeSum += conf;
                    lifetimeSumSq += conf * conf;
                    allTimeMinConfidence = Math.min(allTimeMinConfidence, conf);
                    allTimeMaxConfidence = Math.max(allTimeMaxConfidence, conf);

                    for (const size of Object.keys(windows)) {
                        windows[size].push(conf);
                        if (windows[size].length > Number(size)) windows[size].shift();
                    }

                    updateAllStats();
                }

                signalTimes.push(durationSec);
                if (signalTimes.length > 100) signalTimes.shift();

                const avg = signalTimes.reduce((a, b) => a + b, 0) / signalTimes.length;
                const eta = (totalLines - totalCandles) * avg;

                signalCount++;
                formatSignal({ totalCandles, totalLines, durationSec, avgSignalTime: avg, estimatedTimeSec: eta });

                if (trainingSteps === trainingCutoff) {
                    console.log("Training cutoff reached. Exiting.");
                    process.exit(0);
                }
            } catch (err) {
                console.error("Error in getSignal:", err);
                process.exit(1);
            }
        }
    });

    rd.on('close', () => {
        console.log(`\nCompleted. Processed ${totalCandles.toLocaleString()} candles.`);
        if (shouldPredict && signalCount > 0) {
            formatSignal({ totalCandles, totalLines, durationSec: 0, avgSignalTime: 0, estimatedTimeSec: 0 });
        }
    });
};

countLines().then(lines => {
    totalLines = lines;
    console.log(`Total candles to process: ${totalLines.toLocaleString()}`);
    processCandles();
});