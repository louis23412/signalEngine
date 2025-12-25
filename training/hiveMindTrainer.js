import fs from 'fs';
import path from 'path';
import readline from 'readline';
import HiveMindController from '../src/hiveMindController.js';

const trainingCutoff = null;
const shouldPredict = true;
const cacheSize = 1000;
const ensembleSize = 6;
const candlesUsed = 10;
const directoryPath = path.join(import.meta.dirname, '..', 'state');

const scriptStartTime = process.hrtime.bigint();
const controller = new HiveMindController(directoryPath, cacheSize, ensembleSize, candlesUsed);

let currentStep = -1;
let candlesSinceStepIncrease = 0;
let minConfidenceInCurrentStep = Infinity;
let maxConfidenceInCurrentStep = -Infinity;

let maxRangeInStep = 0;
let maxRangeStep = null;
let maxRangeStableSteps = 0;

let signal;
let totalCandles = 0;
let totalLines = 0;
let signalCount = 0;
let trainingSteps = 0;

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

let lifetimeHealthCount = 0;
let lifetimeHealthSum = 0;
let lifetimeHealthSumSq = 0;
let allTimeMinHealth = Infinity;
let allTimeMaxHealth = -Infinity;

let previousConfidence = null;

let peakTrueAccuracy = 0;
let secondPeakTrueAccuracy = 0;
let currentTrueAccuracy = null;
let modelHealthScore = null;

const G = '\x1b[32m';
const Y = '\x1b[33m';
const R = '\x1b[31m';
const X = '\x1b[0m';
const C = '\x1b[36m';
const M = '\x1b[35m';

const cache = [];
const signalTimes = [];

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
    acc:  { min: null, max: null, mean: null, std: null, count: 0 },
    health: { min: null, max: null, mean: null, std: null, count: 0 }
};

const getHealthColor = (score) => {
    if (score === null) return `${Y}—${X}`;

    if (score >= 75) {
        return `${G}${score.toFixed(1)}${X}`;
    } else if (score >= 50) {
        return `${Y}${score.toFixed(1)}${X}`;
    } else {
        return `${R}${score.toFixed(1)}${X}`;
    }
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

    stats.mean = mean.toFixed(4);
    stats.std = std.toFixed(4);
    stats.min = min.toFixed(4);
    stats.max = max.toFixed(4);
    stats.diff = diff.toFixed(4);
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
        lifetimeStats.conf.mean = mean.toFixed(4);
        lifetimeStats.conf.std = Math.sqrt(variance).toFixed(4);
        lifetimeStats.conf.min = allTimeMinConf === Infinity ? null : allTimeMinConf.toFixed(4);
        lifetimeStats.conf.max = allTimeMaxConf === -Infinity ? null : allTimeMaxConf.toFixed(4);
        lifetimeStats.conf.count = lifetimeConfCount;
    }

    if (lifetimeAccCount > 0) {
        const mean = lifetimeAccSum / lifetimeAccCount;
        const variance = lifetimeAccCount > 1 ? (lifetimeAccSumSq / lifetimeAccCount) - mean ** 2 : 0;
        lifetimeStats.acc.mean = mean.toFixed(4);
        lifetimeStats.acc.std = Math.sqrt(variance).toFixed(4);
        lifetimeStats.acc.min = allTimeMinAcc === Infinity ? null : allTimeMinAcc.toFixed(4);
        lifetimeStats.acc.max = allTimeMaxAcc === -Infinity ? null : allTimeMaxAcc.toFixed(4);
        lifetimeStats.acc.count = lifetimeAccCount;
    }

    if (lifetimeHealthCount > 0) {
        const mean = lifetimeHealthSum / lifetimeHealthCount;
        const variance = lifetimeHealthCount > 1 ? (lifetimeHealthSumSq / lifetimeHealthCount) - mean ** 2 : 0;
        lifetimeStats.health.mean = mean.toFixed(4);
        lifetimeStats.health.std = Math.sqrt(variance).toFixed(4);
        lifetimeStats.health.min = allTimeMinHealth === Infinity ? null : allTimeMinHealth.toFixed(4);
        lifetimeStats.health.max = allTimeMaxHealth === -Infinity ? null : allTimeMaxHealth.toFixed(4);
        lifetimeStats.health.count = lifetimeHealthCount;
    }
};

const computeModelHealth = () => {
    if (lifetimeAccCount < 100) {
        modelHealthScore = null;
        return;
    }

    let score = 0.0;

    const getMean = (size) => parseFloat(windowStats.acc[size]?.mean || 0);
    const getStd  = (size) => parseFloat(windowStats.acc[size]?.std  || 100);
    const getTrend = (size) => windowStats.acc[size]?.trend || 0;

    const mean10   = getMean(10);
    const mean50   = getMean(50);
    const mean100  = getMean(100);
    const mean500  = getMean(500);
    const mean1000 = getMean(1000);

    const std100   = getStd(100);
    const std500   = getStd(500);
    const std1000  = getStd(1000);

    const longTermMean = parseFloat(lifetimeStats.acc.mean || 0);
    const longTermStd  = parseFloat(lifetimeStats.acc.std  || 100);

    const lifetimeMinAcc = allTimeMinAcc === Infinity ? longTermMean : allTimeMinAcc;
    const lifetimeMaxAcc = allTimeMaxAcc === -Infinity ? longTermMean : allTimeMaxAcc;
    const lifetimeRange = Math.max(lifetimeMaxAcc - lifetimeMinAcc, 1);

    const currentAcc = currentTrueAccuracy ?? mean100;
    const recentAcc = Math.max(currentAcc, mean100);

    const recentSharpe100  = std100  < 0.1 ? 100 : mean100  / (std100  + 0.5);
    const recentSharpe500  = std500  < 0.1 ? 100 : mean500  / (std500  + 0.5);
    const recentSharpe1000 = std1000 < 0.1 ? 100 : mean1000 / (std1000 + 0.5);

    const sharpeScore = Math.min(18, recentSharpe100 * 0.4) + Math.min(12, recentSharpe500 * 0.3) + Math.min(8,  recentSharpe1000 * 0.2);
    score += sharpeScore;

    const weightedRecentMean = (mean10 * 0.1 + mean50 * 0.2 + mean100 * 0.3 + mean500 * 0.3 + mean1000 * 0.1);
    score += Math.min(20, weightedRecentMean * 0.35);

    let momentumBonus = 0;
    const positiveTrends = [10, 50, 100, 500, 1000].filter(s => getTrend(s) > 0).length;
    momentumBonus += positiveTrends * 2.5;

    if (mean100 > longTermMean + 2 && getTrend(100) > 0 && getTrend(500) > 0) {
        momentumBonus += 6;
    }
    score += Math.min(15, momentumBonus);

    let lifetimeContextScore = 0;
    const positionInLifetimeRange = (recentAcc - lifetimeMinAcc) / lifetimeRange;
    const clampedPos = Math.min(1.2, Math.max(0, positionInLifetimeRange));

    if (clampedPos >= 1.0) {
        lifetimeContextScore = 25;
    } else if (clampedPos >= 0.9) {
        lifetimeContextScore = 20 + (clampedPos - 0.9) * 50;
    } else if (clampedPos >= 0.7) {
        lifetimeContextScore = 15 + (clampedPos - 0.7) * 25;
    } else if (clampedPos >= 0.5) {
        lifetimeContextScore = 10 + (clampedPos - 0.5) * 25;
    } else if (clampedPos >= 0.3) {
        lifetimeContextScore = 5 + (clampedPos - 0.3) * 16.7;
    } else {
        lifetimeContextScore = clampedPos * 16.7;
    }
    score += lifetimeContextScore;

    const relativeStability = std500 / Math.max(longTermStd, 0.1);
    let stabilityScore = 0;
    if (relativeStability < 0.7) stabilityScore = 10;
    else if (relativeStability < 0.9) stabilityScore = 7;
    else if (relativeStability < 1.1) stabilityScore = 4;
    else if (relativeStability < 1.4) stabilityScore = 1;

    score += stabilityScore;

    const zScore = longTermStd > 0.1 ? (mean100 - longTermMean) / longTermStd : 0;
    let zBonus = 0;
    if (zScore > 2.0) zBonus = 8;
    else if (zScore > 1.0) zBonus = 5;
    else if (zScore > 0.5) zBonus = 3;
    else if (zScore < -1.0) zBonus = -6;
    else if (zScore < -0.5) zBonus = -3;
    score += zBonus;

    if (secondPeakTrueAccuracy > 0 && currentAcc >= secondPeakTrueAccuracy - 2) {
        score += currentAcc >= secondPeakTrueAccuracy ? 6 : 3;
    }

    modelHealthScore = Math.max(0, Math.min(100, score));

    if (modelHealthScore !== null) {
        lifetimeHealthCount++;
        lifetimeHealthSum += modelHealthScore;
        lifetimeHealthSumSq += modelHealthScore * modelHealthScore;
        allTimeMinHealth = Math.min(allTimeMinHealth, modelHealthScore);
        allTimeMaxHealth = Math.max(allTimeMaxHealth, modelHealthScore);
    }
};

const formatTime = (sec) => {
    if (sec < 1) return `${C}${(sec * 1000).toFixed(0)}${X}ms`;

    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = Math.floor(sec % 60);

    const remainingSec = sec - (h * 3600 + m * 60 + s);
    const ms = Math.round(remainingSec * 1000);

    const sPart = ms > 0 ? `${C}${s}${X}s ${C}${ms}${X}ms` : `${C}${s}${X}s`;

    return `${h ? `${C}${h}${X}h ` : ''}${m ? `${C}${m}${X}m ` : ''}${sPart}`;
};

const countLines = () => new Promise(resolve => {
    let count = 0;
    const rl = readline.createInterface({ input: fs.createReadStream(path.join(import.meta.dirname, 'candles.jsonl')) });
    rl.on('line', () => count++);
    rl.on('close', () => resolve(count));
});

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
    const pct = totalLines ? ((totalCandles / totalLines) * 100).toFixed(3) : '0';

    const now = process.hrtime.bigint();
    const runtimeSec = Number(now - scriptStartTime) / 1e9;

    const progressLine = `Progress : ${C}${totalCandles.toLocaleString()}${X} / ${C}${totalLines.toLocaleString()}${X} (${C}${pct}${X}%) | Time : ${formatTime(durationSec)} | Avg : ${formatTime(avgSignalTime)}\nRuntime : ${formatTime(runtimeSec)} | ETA : ${formatTime(estimatedTimeSec)}\n\n`;

    const conf = signal.confidence ?? '—';
    const countAcc = signal.countAccuracy !== 'disabled' ? signal.countAccuracy : '—';
    const trueAcc = signal.trueAccuracy !== 'disabled' ? signal.trueAccuracy : '—';

    const percentile = confidenceWindows[1000].length > 10
        ? (confidenceWindows[1000].filter(c => c <= signal.confidence).length / confidenceWindows[1000].length * 100).toFixed(1)
        : '—';

    const delta = previousConfidence !== null ? signal.confidence - previousConfidence : null;
    const deltaStr = delta === null ? '—' : delta > 0 ? `+${delta.toFixed(4)}` : delta.toFixed(4);
    const deltaCol = delta === null ? '' : delta > 0 ? G : R;
    previousConfidence = signal.confidence;

    const stepDiff = (maxConfidenceInCurrentStep - minConfidenceInCurrentStep).toFixed(4);

    const signalLine = dedent(`
        Entry Price : ${C}${signal.entryPrice}${X} | Sell Price : ${C}${signal.sellPrice}${X} | Stop Price : ${C}${signal.stopLoss}${X} | Trade Multiplier : ${C}${signal.multiplier}${X}

        Current Open Trade Simulations : ${C}${signal.openSimulations}${X}
        Training Steps Completed : ${C}${signal.lastTrainingStep.toLocaleString()}${X}
        Candles Since Last Training Step Increase : ${C}${candlesSinceStepIncrease.toLocaleString()}${X}
        ${shouldPredict ? 
        `
        Model Health : ${modelHealthScore === null ? `${C}—${X}` : getHealthColor(modelHealthScore)}%
        Lifetime Model Health Range : ${R}${lifetimeStats.health.min ?? '—'}${X} → ${G}${lifetimeStats.health.max ?? '—'}${X} Mean : ${C}${lifetimeStats.health.mean ?? '—'}${X} ± ${C}${lifetimeStats.health.std ?? '—'}${X}

        Trade Win Accuracy : ${C}${countAcc}${X}${signal.countAccuracy !== 'disabled' ? '%' : ''}
        True Model Confidence Accuracy : ${C}${trueAcc}${X}${signal.trueAccuracy !== 'disabled' ? '%' : ''}
        Lifetime True Accuracy Range : ${R}${lifetimeStats.acc.min ?? '—'}${X} → ${G}${lifetimeStats.acc.max ?? '—'}${X} Mean : ${C}${lifetimeStats.acc.mean ?? '—'}${X} ± ${C}${lifetimeStats.acc.std ?? '—'}${X}

        Current Model Confidence : ${C}${conf}${X} ${deltaCol}(${deltaStr})${X} ${C}(${percentile}%ile)${X}
        Current Confidence Range : ${R}${minConfidenceInCurrentStep.toFixed(4)}${X} → ${G}${maxConfidenceInCurrentStep.toFixed(4)}${X} ${M}Δ${stepDiff}${X}
        Lifetime Model Confidence Range : ${R}${lifetimeStats.conf.min ?? '—'}${X} → ${G}${lifetimeStats.conf.max ?? '—'}${X} Mean : ${C}${lifetimeStats.conf.mean ?? '—'}${X} ± ${C}${lifetimeStats.conf.std ?? '—'}${X}
        Largest Lifetime Confidence Range : ${C}${maxRangeInStep === 0 ? '—' : maxRangeInStep.toFixed(4)}${X} over ${C}${maxRangeStableSteps.toLocaleString()}${X} candles (step ${C}${maxRangeStep}${X})
        ` 
        : 'prediction & stats disabled'}
    `);

    process.stdout.write('\x1b[2J\x1b[0f');
    process.stdout.write(`\n${'─'.repeat(100)}\n${progressLine}${signalLine}\n${'─'.repeat(100)}\n`);
};

const processCandles = () => {
    const rd = readline.createInterface({
        input: fs.createReadStream(path.join(import.meta.dirname, 'candles.jsonl'))
    });

    rd.on('line', line => {
        const candle = JSON.parse(line);
        cache.push(candle);
        if (cache.length > cacheSize) cache.shift();
        totalCandles++;

        if (cache.length >= cacheSize) {
            try {
                const isLast = totalCandles === totalLines;
                const start = process.hrtime.bigint();
                signal = controller.getSignal(cache, shouldPredict, isLast, trainingCutoff);
                const end = process.hrtime.bigint();
                const durationSec = Number(end - start) / 1e9;

                trainingSteps = signal.lastTrainingStep;

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
                        }

                        currentStep = trainingSteps;
                        candlesSinceStepIncrease = 1;
                        minConfidenceInCurrentStep = conf;
                        maxConfidenceInCurrentStep = conf;
                    } else {
                        candlesSinceStepIncrease++;
                        minConfidenceInCurrentStep = Math.min(minConfidenceInCurrentStep, conf);
                        maxConfidenceInCurrentStep = Math.max(maxConfidenceInCurrentStep, conf);

                        const currentRange = maxConfidenceInCurrentStep - minConfidenceInCurrentStep;
                        if (currentRange > maxRangeInStep || maxRangeStep === currentStep) {
                            maxRangeInStep = currentRange;
                            maxRangeStep = currentStep;
                            maxRangeStableSteps = candlesSinceStepIncrease;
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
                        currentTrueAccuracy = acc;

                        if (acc > peakTrueAccuracy) {
                            secondPeakTrueAccuracy = peakTrueAccuracy;
                            peakTrueAccuracy = acc;
                        } else if (acc > secondPeakTrueAccuracy && acc < peakTrueAccuracy) {
                            secondPeakTrueAccuracy = acc;
                        }

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
                    computeModelHealth();
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

        const finalNow = process.hrtime.bigint();
        const finalRuntimeSec = Number(finalNow - scriptStartTime) / 1e9;

        console.log(`Total runtime: ${formatTime(finalRuntimeSec)}`);

        if (shouldPredict && signalCount > 0) {
            const avg = signalTimes.length > 0 
                ? signalTimes.reduce((a, b) => a + b, 0) / signalTimes.length 
                : 0;

            formatSignal({ 
                totalCandles, 
                totalLines, 
                durationSec: 0, 
                avgSignalTime: avg, 
                estimatedTimeSec: 0 
            });
        }
    });
};

countLines().then(lines => {
    totalLines = lines;
    console.log(`Total candles to process: ${totalLines.toLocaleString()}`);
    processCandles();
});