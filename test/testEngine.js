import fs from 'fs';
import path from 'path';
import readline from 'readline';
import NeuralSignalEngine from '../src/neuralSignalEngine.js';

const engine = new NeuralSignalEngine();

const trainingCutoff = 1001;
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

let previousConfidence = null;

const confidenceWindows = { 10: [], 50: [], 100: [], 500: [], 1000: [], 5000: [], 10000: [], 25000: [] };
const accuracyWindows   = { 10: [], 50: [], 100: [], 500: [], 1000: [], 5000: [], 10000: [], 25000: [] };

const windowStats = {
    conf: {
        10:   { mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: 0 },
        50:   { mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: 0 },
        100:  { mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: 0 },
        500:  { mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: 0 },
        1000: { mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: 0 },
        5000: { mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: 0 },
        10000:{ mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: 0 },
        25000:{ mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: 0 }
    },
    acc: {
        10:   { mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: 0 },
        50:   { mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: 0 },
        100:  { mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: 0 },
        500:  { mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: 0 },
        1000: { mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: 0 },
        5000: { mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: 0 },
        10000:{ mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: 0 },
        25000:{ mean: null, std: null, min: null, max: null, diff: null, size: 0, trend: 0 }
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
        stats.size = 0; stats.trend = 0;
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

const formatNum = (width, val) => {
    if (val === null || val === undefined || val === '—') {
        return '—'.padStart(width, ' ');
    }
    const num = parseFloat(val);
    if (isNaN(num)) return '—'.padStart(width, ' ');
    return num.toFixed(6).padStart(width, ' ');
};

const getMaxWidth = (values) => {
    let max = '—'.length;
    for (const val of values) {
        if (val !== null && val !== undefined) {
            const str = parseFloat(val).toFixed(6);
            max = Math.max(max, str.length);
        }
    }
    return max;
};

const dedent = (str) => {
    const lines = str.split('\n');
    const minIndent = lines
        .filter(line => line.trim())
        .reduce((min, line) => {
            const indent = line.match(/^\s*/)[0].length;
            return indent < min ? indent : min;
        }, Infinity);
    return lines
        .map(line => line.slice(minIndent))
        .join('\n')
        .trim();
};

const formatSignal = ({ totalCandles, totalLines, durationSec, avgSignalTime, estimatedTimeSec }) => {
    const C = '\x1b[36m';
    const G = '\x1b[32m';
    const R = '\x1b[31m';
    const M = '\x1b[35m';
    const B = '\x1b[1m';
    const X = '\x1b[0m';

    process.stdout.write('\x1b[2J\x1b[0f');

    const pct = totalLines ? ((totalCandles / totalLines) * 100).toFixed(3) : '0';
    const progressLine = `Progress : ${C}${totalCandles.toLocaleString()}${X} / ${C}${totalLines.toLocaleString()}${X} (${C}${pct}%${X}) | Time : ${C}${formatTime(durationSec)}${X} | Avg : ${C}${avgSignalTime.toFixed(3)}s${X} | ETA : ${C}${formatTime(estimatedTimeSec)}${X}\n\n`;

    if (!shouldPredict || !signal) {
        process.stdout.write(`\n${'─'.repeat(100)}\n${progressLine}${'─'.repeat(100)}\n\n`);
        return;
    }

    const conf = signal.confidence?.toFixed(6) ?? '—';

    const countAcc = signal.countAccuracy !== 'disabled' 
        ? (typeof signal.countAccuracy === 'number' ? signal.countAccuracy.toFixed(3) : signal.countAccuracy) 
        : '—';

    const trueAcc = signal.trueAccuracy !== 'disabled' 
        ? (typeof signal.trueAccuracy === 'number' ? signal.trueAccuracy.toFixed(3) : signal.trueAccuracy) 
        : '—';

    const percentile = confidenceWindows[1000].length > 10
        ? (confidenceWindows[1000].filter(c => c <= signal.confidence).length / confidenceWindows[1000].length * 100).toFixed(1)
        : '—';

    const delta = previousConfidence !== null ? signal.confidence - previousConfidence : null;
    const deltaStr = delta === null ? '—' : delta > 0 ? `+${delta.toFixed(6)}` : delta.toFixed(6);
    const deltaCol = delta === null ? '' : delta > 0 ? G : R;
    previousConfidence = signal.confidence;

    const stepDiff = (maxConfidenceInCurrentStep - minConfidenceInCurrentStep).toFixed(6);

    const isCurrentRecord = maxRangeStep === currentStep;
    const recordLabel = isCurrentRecord ? `${C}(current step)${X}` : `(step ${C}${maxRangeStep}${X})`;

    const allMeans = [];
    const allStds = [];
    const allMinMax = [];
    const allDiffs = [];

    Object.values(windowStats.conf).concat(Object.values(windowStats.acc)).forEach(s => {
        allMeans.push(s.mean);
        allStds.push(s.std);
        allMinMax.push(s.min, s.max);
        allDiffs.push(s.diff);
    });

    const wMean   = getMaxWidth(allMeans);
    const wStd    = getMaxWidth(allStds);
    const wMinMax = getMaxWidth(allMinMax);
    const wDiff   = getMaxWidth(allDiffs);

    const fmtMean   = (v) => formatNum(wMean, v);
    const fmtStd    = (v) => formatNum(wStd, v);
    const fmtMinMax = (v) => formatNum(wMinMax, v);
    const fmtDiff   = (v) => formatNum(wDiff, v);

    const signalLine = dedent(`
        Regulate every ${C}${regulateFreq ?? '—'}${X} steps → last : ${C}${regulateStep ?? '—'}${X} next in ${C}${regulateFreq ? regulateFreq - (trainingSteps % regulateFreq) : '—'}${X}
        Gradient reset ${C}${gradientResetFreq ?? '—'}${X} steps → last : ${C}${gradientResetStep ?? '—'}${X} next in ${C}${gradientResetFreq ? gradientResetFreq - (trainingSteps % gradientResetFreq) : '—'}${X}

        ${B}Current Signal${X}
        Training Step : ${C}${signal.lastTrainingStep.toLocaleString()}${X}
        Open simulations : ${C}${signal.openSimulations}${X}
        Entry : ${C}${signal.entryPrice}${X} Sell : ${C}${signal.sellPrice}${X} Stop : ${C}${signal.stopLoss}${X} Mult : x${C}${signal.multiplier.toFixed(3)}${X}
        Conf : ${C}${conf}${X} ${deltaCol}(${deltaStr})${X} ${M}(${percentile}%ile)${X}

        ${B}Model Accuracies${X}
        Count Accuracy : ${C}${countAcc}${X}${signal.countAccuracy !== 'disabled' ? '%' : ''}  True Accuracy  : ${C}${trueAcc}${X}${signal.trueAccuracy !== 'disabled' ? '%' : ''}

        ${B}Step Stability Window${X}
        Candles in current step : ${C}${candlesSinceStepIncrease.toLocaleString()}${X}
        Confidence range : ${R}${minConfidenceInCurrentStep.toFixed(6)}${X} → ${G}${maxConfidenceInCurrentStep.toFixed(6)}${X} ${M}Δ${stepDiff}${X}

        Largest range in stable window  : ${C}${maxRangeInStep === 0 ? '—' : maxRangeInStep.toFixed(6)}${X}
            └─ over ${C}${maxRangeStableSteps.toLocaleString()}${X} candles ${recordLabel} (gradient ${C}${formatWindow(maxRangeAtGradientStep, gradientResetFreq)}${X} / regulate ${C}${formatWindow(maxRangeAtRegulateStep, regulateFreq)}${X})

        ${B}Lifetime Confidence${X}
        Range : ${R}${lifetimeStats.conf.min ?? '—'}${X} → ${G}${lifetimeStats.conf.max ?? '—'}${X}   Mean : ${C}${lifetimeStats.conf.mean ?? '—'}${X} ± ${C}${lifetimeStats.conf.std ?? '—'}${X}

        ${B}Lifetime True Accuracy${X}
        Range : ${R}${lifetimeStats.acc.min ?? '—'}${X} → ${G}${lifetimeStats.acc.max ?? '—'}${X}   Mean : ${C}${lifetimeStats.acc.mean ?? '—'}${X} ± ${C}${lifetimeStats.acc.std ?? '—'}${X}

        ${B}Rolling Windows${X}
        ${buildWindowRow('10 ', windowStats.conf[10], windowStats.acc[10], fmtMean, fmtStd, fmtMinMax, fmtDiff)}
        ${buildWindowRow('50 ', windowStats.conf[50], windowStats.acc[50], fmtMean, fmtStd, fmtMinMax, fmtDiff)}
        ${buildWindowRow('100', windowStats.conf[100], windowStats.acc[100], fmtMean, fmtStd, fmtMinMax, fmtDiff)}
        ${buildWindowRow('500', windowStats.conf[500], windowStats.acc[500], fmtMean, fmtStd, fmtMinMax, fmtDiff)}
        ${buildWindowRow('1K ', windowStats.conf[1000], windowStats.acc[1000], fmtMean, fmtStd, fmtMinMax, fmtDiff)}
        ${buildWindowRow('5K ', windowStats.conf[5000], windowStats.acc[5000], fmtMean, fmtStd, fmtMinMax, fmtDiff)}
        ${buildWindowRow('10K', windowStats.conf[10000], windowStats.acc[10000], fmtMean, fmtStd, fmtMinMax, fmtDiff)}
        ${buildWindowRow('25K', windowStats.conf[25000], windowStats.acc[25000], fmtMean, fmtStd, fmtMinMax, fmtDiff)}
    `);

    process.stdout.write(`\n${'─'.repeat(100)}\n${progressLine}${signalLine}\n${'─'.repeat(100)}\n`);
};

const buildWindowRow = (label, confStats, accStats, fmtMean, fmtStd, fmtMinMax, fmtDiff) => {
    const meanConf = colorMeanByTrend(fmtMean(confStats.mean), confStats.trend);
    const stdConf  = '\x1b[36m' + fmtStd(confStats.std) + '\x1b[0m';
    const minConf  = '\x1b[31m' + fmtMinMax(confStats.min) + '\x1b[0m';
    const maxConf  = '\x1b[32m' + fmtMinMax(confStats.max) + '\x1b[0m';
    const diffConf = '\x1b[35mΔ' + fmtDiff(confStats.diff) + '\x1b[0m';

    const meanAcc = colorMeanByTrend(fmtMean(accStats.mean), accStats.trend);
    const stdAcc  = '\x1b[36m' + fmtStd(accStats.std) + '\x1b[0m';
    const minAcc  = '\x1b[31m' + fmtMinMax(accStats.min) + '\x1b[0m';
    const maxAcc  = '\x1b[32m' + fmtMinMax(accStats.max) + '\x1b[0m';
    const diffAcc = '\x1b[35mΔ' + fmtDiff(accStats.diff) + '\x1b[0m';

    return `${padLabel(label)}Confidence : ${meanConf} ± ${stdConf} [${minConf} → ${maxConf}] ${diffConf}
            └─True Acc : ${meanAcc} ± ${stdAcc} [${minAcc} → ${maxAcc}] ${diffAcc}`;
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
                    const acc = signal.trueAccuracy !== 'disabled' && typeof signal.trueAccuracy === 'number' ? signal.trueAccuracy : null;

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

                        currentStep = trainingSteps;
                        candlesSinceStepIncrease = 1;
                        minConfidenceInCurrentStep = conf;
                        maxConfidenceInCurrentStep = conf;
                        confidencePathInStep = [conf];
                    } else {
                        candlesSinceStepIncrease++;
                        minConfidenceInCurrentStep = Math.min(minConfidenceInCurrentStep, conf);
                        maxConfidenceInCurrentStep = Math.max(maxConfidenceInCurrentStep, conf);
                        confidencePathInStep.push(conf);

                        const currentRange = maxConfidenceInCurrentStep - minConfidenceInCurrentStep;
                        if (currentRange > maxRangeInStep || maxRangeStep === currentStep) {
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
                    console.log(`Training cutoff reached (${trainingCutoff}). Exiting.`);
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