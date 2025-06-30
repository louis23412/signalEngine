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
const trainingCutoff = null

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
                      `Entry Price: ${ANSI_CYAN}${signal.entryPrice}${ANSI_RESET}, ` +
                      `Sell Price: ${ANSI_CYAN}${signal.sellPrice}${ANSI_RESET}, ` +
                      `Stop Price: ${ANSI_CYAN}${signal.stopLoss}${ANSI_RESET}\n` +
                      `Multiplier: ${ANSI_CYAN}${signal.multiplier}${ANSI_RESET}, ` +
                      `Confidence: ${ANSI_CYAN}${signal.confidence}${ANSI_RESET}, ` +
                      `Threshold: ${ANSI_CYAN}${signal.threshold}${ANSI_RESET}, ` +
                      `Training Step: ${ANSI_CYAN}${signal.lastTrainingStep}${ANSI_RESET}`

    process.stdout.write(`-----------------------\n${progressLine}${signalLine}\n-----------------------\n`);
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
                const status = totalCandles === totalLines

                const startTime = process.hrtime.bigint();
                signal = engine.getSignal(cache.slice(-cacheSize), status, trainingCutoff);
                const endTime = process.hrtime.bigint();
                const durationSec = Number(endTime - startTime) / 1_000_000_000;

                trainingSteps = signal.lastTrainingStep

                signalTimes.push(durationSec);
                if (signalTimes.length > 100) {
                    signalTimes.shift();
                }
                signalCount++;

                const avgSignalTime = signalTimes.reduce((sum, time) => sum + time, 0) / signalTimes.length;
                const remainingCandles = totalLines - totalCandles;
                const estimatedTimeSec = remainingCandles * avgSignalTime;

                formatSignal({ totalCandles, totalLines, durationSec, avgSignalTime, estimatedTimeSec });

                if (trainingSteps === trainingCutoff) {
                    console.log("Training complete. Exiting.");
                    process.exit()
                }
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