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

const formatSignal = (signal) => {
    return `Signal => Action: ${signal.suggestedAction}, Multiplier: ${signal.multiplier.toFixed(3)}, ` +
           `Entry: ${signal.entryPrice.toFixed(2)}, Sell: ${signal.sellPrice.toFixed(2)}, ` +
           `Stop: ${signal.stopLoss.toFixed(2)}, Reward: ${signal.expectedReward}`;
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

                process.stdout.write('\x1B[?25l');
                process.stdout.cursorTo(0, 0);
                process.stdout.write('\x1B[0J');

                const progressLine = `Progress => ${totalCandles}/${totalLines} candles (${((totalCandles / totalLines) * 100).toFixed(6)}%), ` +
                                    `Time: ${durationSec.toFixed(3)}s, ` +
                                    `Avg Time: ${avgSignalTime.toFixed(3)}s, ` +
                                    `ETA: ${formatTime(estimatedTimeSec)}`;
                
                console.log(signal)
                console.log(progressLine)
                console.log('---------------------------------------------------------')
                process.stdout.write('\x1B[?25h');

            } catch (e) {
                console.log(e);
                process.exit();
            }
        }
    });

    rd.on('close', () => {
        console.log(`\nCompleted. Total Candles: ${totalCandles}`);
        console.log(formatSignal(signal));
    });
};

countLines().then((lineCount) => {
    totalLines = lineCount;
    console.log(`Total lines to process: ${totalLines}`);
    processCandles();
});