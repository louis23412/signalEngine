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

let lifetimeConfCount = 0;
let lifetimeConfSum = 0;
let lifetimeConfSumSq = 0;
let allTimeMinConf = Infinity;
let allTimeMaxConf = -Infinity;

let lifetimeAccCount = 0;
let lifetimeAccSum = 0;
let lifetimeAccSumSq = 0;
let allTimeMinAcc = Infinity;
let allTimeMaxAcc = -Infinity;

let corrSumXY = 0;
let corrSumX = 0;
let corrSumY = 0;
let corrSumX2 = 0;
let corrSumY2 = 0;
let corrN = 0;

let previousConfidence = null;
let previousOverallHealth = null;

const confidenceWindows = { 10: [], 50: [], 100: [], 500: [], 1000: [], 5000: [], 10000: [], 25000: [] };
const accuracyWindows   = { 10: [], 50: [], 100: [], 500: [], 1000: [], 5000: [], 10000: [], 25000: [] };

const windowStats = {
    conf: {
        10:   { mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: 0, healthScore: '—' },
        50:   { mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: 0, healthScore: '—' },
        100:  { mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: 0, healthScore: '—' },
        500:  { mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: 0, healthScore: '—' },
        1000: { mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: 0, healthScore: '—' },
        5000: { mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: 0, healthScore: '—' },
        10000:{ mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: 0, healthScore: '—' },
        25000:{ mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: 0, healthScore: '—' }
    },
    acc: {
        10:   { mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: 0, healthScore: '—' },
        50:   { mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: 0, healthScore: '—' },
        100:  { mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: 0, healthScore: '—' },
        500:  { mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: 0, healthScore: '—' },
        1000: { mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: 0, healthScore: '—' },
        5000: { mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: 0, healthScore: '—' },
        10000:{ mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: 0, healthScore: '—' },
        25000:{ mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: 0, healthScore: '—' }
    }
};

const lifetimeStats = {
    conf: { min: null, max: null, mean: null, std: null, count: 0 },
    acc:  { min: null, max: null, mean: null, std: null, count: 0 }
};

const formatWindow = (lastResetStep, freq) => {
    if (lastResetStep === null || freq === null || freq <= 0) return '—';
    const start = lastResetStep;
    const end = start + freq;
    return `${start.toLocaleString()} - ${end.toLocaleString()}`;
};

const computeHealthScore = (arr, windowSize = null) => {
    if (!arr || arr.length < 3) return '—';
    let totalAbsChange = 0;
    let oscillations = 0;
    let prevDiff = null;
    for (let i = 1; i < arr.length; i++) {
        const diff = arr[i] - arr[i - 1];
        totalAbsChange += Math.abs(diff);
        if (prevDiff !== null && Math.sign(diff) !== Math.sign(prevDiff) && Math.abs(diff) > 1e-8) oscillations++;
        prevDiff = diff;
    }
    const n = arr.length - 1;
    const avgAbsChange = totalAbsChange / n;
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    const relativeChange = mean > 0 ? avgAbsChange / mean : avgAbsChange;
    const smoothness = Math.max(0, 1 - relativeChange * 9);
    const maxReasonableOsc = Math.max(1, n * 0.15);
    const oscillationPenalty = 1 - Math.min(oscillations / maxReasonableOsc, 0.9);
    const variance = arr.reduce((a, v) => a + (v - mean) ** 2, 0) / arr.length;
    const stdRel = mean > 0 ? Math.sqrt(variance) / mean : 1;
    const stabilityBonus = mean > 0.3 ? Math.max(0, 1 - stdRel * 5) : 0;
    let score = smoothness * 0.55 + oscillationPenalty * 0.30 + stabilityBonus * 0.15;
    if (windowSize && windowSize <= 50) score = Math.min(1, score * 1.08);
    return Math.max(0, Math.min(1, score)).toFixed(3);
};

const formatNum = (val) => {
    if (val === null || val === undefined || val === '—') return '       —';
    const num = parseFloat(val);
    if (isNaN(num)) return '       —';
    return num.toFixed(6).padStart(10, ' ');
};

const stripAnsi = (str) => str.replace(/\x1b\[[0-9;]*m/g, '');
const visibleLength = (str) => stripAnsi(str.toString()).length;

const padVisible = (str, width, align = 'left') => {
    if (str === null || str === undefined || str === '—') str = '—';
    str = str.toString();
    const visibleLen = visibleLength(str);
    const paddingNeeded = Math.max(0, width - visibleLen);
    const padding = ' '.repeat(paddingNeeded);
    return align === 'right' ? padding + str : str + padding;
};

const padLabel = (label) => padVisible(label, 4);
const padHealth = (score) => padVisible(colorHealthPct(score), 8, 'left');

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

const updateWindowStats = (type, size) => {
    const windows = type === 'conf' ? confidenceWindows : accuracyWindows;
    const stats = windowStats[type][size];
    const arr = windows[size];

    if (arr.length === 0) {
        stats.mean = stats.std = stats.min = stats.max = stats.diff = null;
        stats.size = 0; stats.trend = 0; stats.healthScore = '—';
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

    stats.mean = mean.toFixed(6);
    stats.std = std.toFixed(6);
    stats.min = min.toFixed(6);
    stats.max = max.toFixed(6);
    stats.diff = diff.toFixed(6);
    stats.size = arr.length;
    stats.trend = trend;
    stats.healthScore = computeHealthScore(arr, Number(size));
};

const updateAllStats = () => {
    for (const size of Object.keys(confidenceWindows)) {
        updateWindowStats('conf', Number(size));
        updateWindowStats('acc', Number(size));
    }

    if (lifetimeConfCount > 0) {
        const mean = lifetimeConfSum / lifetimeConfCount;
        const variance = lifetimeConfCount > 1 ? (lifetimeConfSumSq / lifetimeConfCount) - mean ** 2 : 0;
        lifetimeStats.conf.mean = mean.toFixed(6);
        lifetimeStats.conf.std = Math.sqrt(variance).toFixed(6);
        lifetimeStats.conf.min = allTimeMinConf === Infinity ? null : allTimeMinConf.toFixed(6);
        lifetimeStats.conf.max = allTimeMaxConf === -Infinity ? null : allTimeMaxConf.toFixed(6);
        lifetimeStats.conf.count = lifetimeConfCount;
    }

    if (lifetimeAccCount > 0) {
        const mean = lifetimeAccSum / lifetimeAccCount;
        const variance = lifetimeAccCount > 1 ? (lifetimeAccSumSq / lifetimeAccCount) - mean ** 2 : 0;
        lifetimeStats.acc.mean = mean.toFixed(6);
        lifetimeStats.acc.std = Math.sqrt(variance).toFixed(6);
        lifetimeStats.acc.min = allTimeMinAcc === Infinity ? null : allTimeMinAcc.toFixed(6);
        lifetimeStats.acc.max = allTimeMaxAcc === -Infinity ? null : allTimeMaxAcc.toFixed(6);
        lifetimeStats.acc.count = lifetimeAccCount;
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
    const rl = readline.createInterface({ input: fs.createReadStream(path.join(import.meta.dirname, 'candles.jsonl')) });
    rl.on('line', () => count++);
    rl.on('close', () => resolve(count));
});

const buildWindowRow = (label, stats) => {
    const meanRaw = stats.mean ?? '—';
    const stdRaw = stats.std ?? '—';
    const minRaw = stats.min ?? '—';
    const maxRaw = stats.max ?? '—';
    const diffRaw = stats.diff ?? '—';

    const mean = colorMeanByTrend(formatNum(meanRaw), stats.trend);
    const std = '\x1b[36m' + formatNum(stdRaw) + '\x1b[0m';
    const minVal = '\x1b[31m' + formatNum(minRaw) + '\x1b[0m';
    const maxVal = '\x1b[32m' + formatNum(maxRaw) + '\x1b[0m';
    const diff = '\x1b[35mΔ' + formatNum(diffRaw) + '\x1b[0m';

    return `  ${padLabel(label)}: ${mean} ± ${std} [${minVal} → ${maxVal}] ${diff} H: ${padHealth(stats.healthScore)}`;
};

const getTrendArrow = (current, previous) => {
    if (previous === null || current === '—') return ' ';
    const diff = parseFloat(current) - parseFloat(previous);
    if (Math.abs(diff) < 0.005) return '→';
    return diff > 0 ? '↑' : '↓';
};

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
    const progressLine = `Progress : ${C}${totalCandles.toLocaleString()}${X} / ${C}${totalLines.toLocaleString()}${X} (${C}${pct}%${X}) | Time : ${C}${formatTime(durationSec)}${X} | Avg : ${C}${avgSignalTime.toFixed(3)}s${X} | ETA : ${C}${formatTime(estimatedTimeSec)}${X}\n`;

    if (!shouldPredict || !signal) {
        process.stdout.write(`\n${'─'.repeat(100)}\n${progressLine}${'─'.repeat(100)}\n\n`);
        return;
    }

    const conf = signal.confidence?.toFixed(6) ?? '—';
    const acc = signal.globalAccuracy !== 'disabled' ? (typeof signal.globalAccuracy === 'number' ? signal.globalAccuracy.toFixed(3) : signal.globalAccuracy) : '—';
    const percentile = confidenceWindows[1000].length > 10
        ? (confidenceWindows[1000].filter(c => c <= signal.confidence).length / confidenceWindows[1000].length * 100).toFixed(1)
        : '—';

    const delta = previousConfidence !== null ? signal.confidence - previousConfidence : null;
    const deltaStr = delta === null ? '—' : delta > 0 ? `+${delta.toFixed(6)}` : delta.toFixed(6);
    const deltaCol = delta === null ? '' : delta > 0 ? G : R;
    previousConfidence = signal.confidence;

    const stepDiff = (maxConfidenceInCurrentStep - minConfidenceInCurrentStep).toFixed(6);
    const currentHealthScore = computeHealthScore(confidencePathInStep, confidencePathInStep.length);

    const corrConfidence = Math.min(1, corrN / 500);
    const corrNum = corrN > 10 ? (corrN * corrSumXY - corrSumX * corrSumY) / Math.sqrt((corrN * corrSumX2 - corrSumX ** 2) * (corrN * corrSumY2 - corrSumY ** 2) || 1) : 0;
    const corrBonus = corrN > 10 ? Math.max(0, corrNum * 0.4) * corrConfidence : 0;
    const corrPenalty = corrN > 10 && corrNum < 0 ? Math.abs(corrNum) * 0.5 * corrConfidence : 0;

    const sizes = [500, 1000, 5000, 10000];
    const weights = [0.2, 0.4, 0.3, 0.1];
    let weightedConf = 0;
    let weightedAcc = 0;
    let totalWeight = 0;

    const healthLines = sizes.map((size, i) => {
        const ch = parseFloat(windowStats.conf[size].healthScore || '0');
        const ah = parseFloat(windowStats.acc[size].healthScore || '0');
        const blended = (ch * 0.55 + ah * 0.45) * (1 + corrBonus) * (1 - corrPenalty);
        const score = Math.max(0, Math.min(1, blended)).toFixed(3);
        const pct = (parseFloat(score) * 100).toFixed(1);
        const col = parseFloat(score) >= 0.80 ? G : parseFloat(score) >= 0.60 ? Y : R;
        weightedConf += ch * weights[i];
        weightedAcc += ah * weights[i];
        totalWeight += weights[i];
        const label = size === 1000 ? '1K ' : size === 5000 ? '5K ' : size === 10000 ? '10K' : '500';
        return `  ${padVisible(label, 4)} : ${col}${pct}%${X}`;
    });

    const finalBlended = totalWeight > 0 ? (weightedConf / totalWeight * 0.55 + weightedAcc / totalWeight * 0.45) * (1 + corrBonus) * (1 - corrPenalty) : 0;
    const overallHealth = Math.max(0, Math.min(1, finalBlended)).toFixed(3);
    const overallPct = (parseFloat(overallHealth) * 100).toFixed(1);
    const overallColor = parseFloat(overallHealth) >= 0.80 ? G : parseFloat(overallHealth) >= 0.60 ? Y : R;
    const trendArrow = getTrendArrow(overallHealth, previousOverallHealth);
    previousOverallHealth = overallHealth;

    const isCurrentRecord = maxRangeStep === currentStep;
    const recordLabel = isCurrentRecord ? `${Y}(current step)${X}` : `(step ${C}${maxRangeStep}${X})`;

    const signalLine = `
${B}Signal${X}
  Training Step : ${C}${signal.lastTrainingStep.toLocaleString()}${X}
  Entry : ${C}${signal.entryPrice}${X}   Sell : ${C}${signal.sellPrice}${X}   Stop : ${C}${signal.stopLoss}${X}
  Mult  : ${C}${signal.multiplier.toFixed(3)}${X}   Conf : ${C}${conf}${X} ${deltaCol}(${deltaStr})${X} ${M}(${percentile}%ile)${X}
  Prediction Accuracy : ${C}${acc}${X}${signal.globalAccuracy !== 'disabled' ? '%' : ''} Open simulations : ${C}${signal.openSimulations}${X}

  Regulate every ${C}${regulateFreq ?? '—'}${X} steps → last : ${C}${regulateStep ?? '—'}${X} next in ${C}${regulateFreq ? regulateFreq - (trainingSteps % regulateFreq) : '—'}${X}
  Gradient reset ${C}${gradientResetFreq ?? '—'}${X} steps → last : ${C}${gradientResetStep ?? '—'}${X} next in ${C}${gradientResetFreq ? gradientResetFreq - (trainingSteps % gradientResetFreq) : '—'}${X}

${B}STEP STABILITY WINDOW${X}
  Candles in current step : ${C}${candlesSinceStepIncrease.toLocaleString()}${X}
  Confidence range : ${R}${minConfidenceInCurrentStep.toFixed(6)}${X} → ${G}${maxConfidenceInCurrentStep.toFixed(6)}${X} ${M}Δ${stepDiff}${X}
  Health : ${colorHealthPct(currentHealthScore)}

  Largest range in stable window  : ${C}${maxRangeInStep === 0 ? '—' : maxRangeInStep.toFixed(6)}${X}
    └─ over ${C}${maxRangeStableSteps.toLocaleString()}${X} candles ${recordLabel} (gradient ${C}${formatWindow(maxRangeAtGradientStep, gradientResetFreq)}${X} / regulate ${C}${formatWindow(maxRangeAtRegulateStep, regulateFreq)}${X})

${B}LIFETIME CONFIDENCE${X}
  Range : ${R}${lifetimeStats.conf.min ?? '—'}${X} → ${G}${lifetimeStats.conf.max ?? '—'}${X}   Mean : ${C}${lifetimeStats.conf.mean ?? '—'}${X} ± ${C}${lifetimeStats.conf.std ?? '—'}${X}

${B}LIFETIME ACCURACY${X}
  Range : ${R}${lifetimeStats.acc.min ?? '—'}${X} → ${G}${lifetimeStats.acc.max ?? '—'}${X}   Mean : ${C}${lifetimeStats.acc.mean ?? '—'}${X} ± ${C}${lifetimeStats.acc.std ?? '—'}${X}

${B}OVERALL MODEL HEALTH${X}
${healthLines.join(' ')}
  Final Blended : ${overallColor}${overallPct}%${X} ${M}${trendArrow}${X}

${B}ROLLING CONFIDENCE WINDOWS${X}
${buildWindowRow('10 ', windowStats.conf[10])}
${buildWindowRow('50 ', windowStats.conf[50])}
${buildWindowRow('100', windowStats.conf[100])}
${buildWindowRow('500', windowStats.conf[500])}
${buildWindowRow('1K ', windowStats.conf[1000])}
${buildWindowRow('5K ', windowStats.conf[5000])}
${buildWindowRow('10K', windowStats.conf[10000])}
${buildWindowRow('25K', windowStats.conf[25000])}

${B}ROLLING ACCURACY WINDOWS${X}
${buildWindowRow('10 ', windowStats.acc[10])}
${buildWindowRow('50 ', windowStats.acc[50])}
${buildWindowRow('100', windowStats.acc[100])}
${buildWindowRow('500', windowStats.acc[500])}
${buildWindowRow('1K ', windowStats.acc[1000])}
${buildWindowRow('5K ', windowStats.acc[5000])}
${buildWindowRow('10K', windowStats.acc[10000])}
${buildWindowRow('25K', windowStats.acc[25000])}
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
                    const acc = signal.globalAccuracy !== 'disabled' && typeof signal.globalAccuracy === 'number' ? signal.globalAccuracy : null;

                    if (currentStep === -1) currentStep = trainingSteps;

                    if (trainingSteps > currentStep) {
                        const completedRange = maxConfidenceInCurrentStep - minConfidenceInCurrentStep;
                        if (completedRange > maxRangeInStep) {
                            maxRangeInStep = completedRange;
                            maxRangeStep = currentStep;
                            maxRangeStableSteps = candlesSinceStepIncrease;
                            maxRangeAtGradientStep = gradientResetFreq ? currentStep - (currentStep % gradientResetFreq) : null;
                            maxRangeAtRegulateStep = regulateFreq ? currentStep - (currentStep % regulateFreq) : null;
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

                        const currentRange = maxConfidenceInCurrentStep - minConfidenceInCurrentStep;
                        if (currentRange > maxRangeInStep) {
                            maxRangeInStep = currentRange;
                            maxRangeStep = currentStep;
                            maxRangeStableSteps = candlesSinceStepIncrease;
                            maxRangeAtGradientStep = gradientResetFreq ? currentStep - (currentStep % gradientResetFreq) : null;
                            maxRangeAtRegulateStep = regulateFreq ? currentStep - (currentStep % regulateFreq) : null;
                        }
                    }

                    lifetimeConfCount++;
                    lifetimeConfSum += conf;
                    lifetimeConfSumSq += conf * conf;
                    allTimeMinConf = Math.min(allTimeMinConf, conf);
                    allTimeMaxConf = Math.max(allTimeMaxConf, conf);

                    for (const size of Object.keys(confidenceWindows)) {
                        confidenceWindows[size].push(conf);
                        if (confidenceWindows[size].length > Number(size)) confidenceWindows[size].shift();
                    }

                    if (acc !== null) {
                        lifetimeAccCount++;
                        lifetimeAccSum += acc;
                        lifetimeAccSumSq += acc * acc;
                        allTimeMinAcc = Math.min(allTimeMinAcc, acc);
                        allTimeMaxAcc = Math.max(allTimeMaxAcc, acc);

                        for (const size of Object.keys(accuracyWindows)) {
                            accuracyWindows[size].push(acc);
                            if (accuracyWindows[size].length > Number(size)) accuracyWindows[size].shift();
                        }

                        corrN++;
                        corrSumX += conf;
                        corrSumY += acc;
                        corrSumXY += conf * acc;
                        corrSumX2 += conf * conf;
                        corrSumY2 += acc * acc;
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