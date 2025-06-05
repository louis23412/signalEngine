import fs from 'fs';
import path from 'path';
import readline from 'readline';
import NeuralSignalEngine from '../src/neuralSignalEngine.js';

const engine = new NeuralSignalEngine();

const cacheSize = 100;
const cache = [];
const signalTimes = [];

let signal;
let totalCandles = 0;
let totalLines = 0;
let signalCount = 0;

const formatTime = (seconds) => {
    if (seconds < 1) return `${(seconds * 1000).toFixed(0)}ms`;
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    return `${h}h ${m}m ${s}s`;
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

    const ANSI_CYAN = '\x1B[36m';
    const ANSI_RESET = '\x1B[0m';

    let progressLine = '';
    if (totalCandles !== null && totalLines !== null && durationSec !== null && avgSignalTime !== null && estimatedTimeSec !== null) {
        progressLine = `Progress:\n${ANSI_CYAN}${totalCandles}/${totalLines}${ANSI_RESET} candles (${ANSI_CYAN}${((totalCandles / totalLines) * 100).toFixed(6)}%${ANSI_RESET}), ` +
                       `Time: ${ANSI_CYAN}${durationSec.toFixed(3)}s${ANSI_RESET}, ` +
                       `Avg Time: ${ANSI_CYAN}${avgSignalTime.toFixed(3)}s${ANSI_RESET}, ` +
                       `ETA: ${ANSI_CYAN}${formatTime(estimatedTimeSec)}${ANSI_RESET}\n`;
    }

    const signalLine = `Signal:\n` +
                      `Suggested Action: ${ANSI_CYAN}${signal.suggestedAction}${ANSI_RESET}, ` +
                      `Multiplier: ${ANSI_CYAN}${signal.multiplier}${ANSI_RESET}, ` +
                      `Expected Reward: ${ANSI_CYAN}${signal.expectedReward}${ANSI_RESET}\n` +
                      `Entry Price: ${ANSI_CYAN}${signal.entryPrice}${ANSI_RESET}, ` +
                      `Sell Price: ${ANSI_CYAN}${signal.sellPrice}${ANSI_RESET}, ` +
                      `Stop Price: ${ANSI_CYAN}${signal.stopLoss}${ANSI_RESET}\n` +
                      `Raw Confidence: ${ANSI_CYAN}${signal.rawConfidence}${ANSI_RESET}, ` +
                      `Raw Threshold: ${ANSI_CYAN}${signal.rawThreshold}${ANSI_RESET}\n` +
                      `Filtered Confidence: ${ANSI_CYAN}${signal.filteredConfidence}${ANSI_RESET}, ` +
                      `Filtered Threshold: ${ANSI_CYAN}${signal.filteredThreshold}${ANSI_RESET}`;

    process.stdout.write('\x1B[?25l');
    process.stdout.cursorTo(0, 0);
    process.stdout.write('\x1B[0J');
    process.stdout.write(`-----------------------\n${progressLine}${signalLine}\n-----------------------`);
    process.stdout.write('\x1B[?25h');
};

const processCandles = () => {
    const rd = readline.createInterface({
        input: fs.createReadStream(path.join(import.meta.dirname, 'candles.jsonl'))
    });

    rd.on('line', (line) => {
        const candleObj = JSON.parse(line);
        cache.push(candleObj);
        if (cache.length > cacheSize) {
            cache.shift();
        }
        totalCandles++;

        if (cache.length >= cacheSize) {
            try {
                const startTime = process.hrtime.bigint();
                signal = engine.getSignal(cache.slice(-cacheSize));
                const endTime = process.hrtime.bigint();
                const durationSec = Number(endTime - startTime) / 1_000_000_000;

                signalTimes.push(durationSec);
                if (signalTimes.length > 100) {
                    signalTimes.shift();
                }
                signalCount++;

                const avgSignalTime = signalTimes.reduce((sum, time) => sum + time, 0) / signalTimes.length;
                const remainingCandles = totalLines - totalCandles;
                const estimatedTimeSec = remainingCandles * avgSignalTime;

                formatSignal({ totalCandles, totalLines, durationSec, avgSignalTime, estimatedTimeSec });
            } catch (e) {
                console.log(e);
                process.exit();
            }
        }
    });

    rd.on('close', () => {
        console.log(`\nCompleted. Total Candles: ${totalCandles}`);
        formatSignal();
    });
};

countLines().then((lineCount) => {
    totalLines = lineCount;
    console.log(`Total lines to process: ${totalLines}`);
    processCandles();
});