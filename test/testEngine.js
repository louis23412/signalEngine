import fs from 'fs';
import path from 'path';
import readline from 'readline';
import NeuralSignalEngine from '../src/signals.js';

const engine = new NeuralSignalEngine();

const cacheSize = 100; // Fixed cache size
const cache = []; // Simple array to store candles
const signalTimes = []; // Array to store the last 100 signal processing times

let signal;
let totalCandles = 0;
let totalLines = 0;
let signalCount = 0;

// Format seconds into hours, minutes, seconds, or milliseconds for small values
const formatTime = (seconds) => {
    if (seconds < 1) return `${(seconds * 1000).toFixed(0)}ms`;
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    return `${h}h ${m}m ${s}s`;
};

// Count total lines in the file first
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

// Format signal values into a single-line display
const formatSignal = (signal) => {
    return `Signal => Action: ${signal.suggestedAction}, Multiplier: ${signal.multiplier.toFixed(3)}, ` +
           `Entry: ${signal.entryPrice.toFixed(2)}, Sell: ${signal.sellPrice.toFixed(2)}, ` +
           `Stop: ${signal.stopLoss.toFixed(2)}, Reward: ${signal.expectedReward.toFixed(3)}`;
};

// Process candles
const processCandles = () => {
    const rd = readline.createInterface({
        input: fs.createReadStream(path.join(import.meta.dirname, 'candles.jsonl'))
    });

    rd.on('line', (line) => {
        const candleObj = JSON.parse(line);
        cache.push(candleObj);
        if (cache.length > cacheSize) {
            cache.shift(); // Remove oldest candle to maintain cache size
        }
        totalCandles++;

        if (cache.length >= cacheSize) {
            try {
                const startTime = process.hrtime.bigint(); // Start timing
                signal = engine.getSignal(cache.slice(-cacheSize)); // Use last 100 candles
                const endTime = process.hrtime.bigint(); // End timing
                const durationSec = Number(endTime - startTime) / 1_000_000_000; // Convert to seconds

                // Store the signal time
                signalTimes.push(durationSec);
                if (signalTimes.length > 100) {
                    signalTimes.shift(); // Keep only the last 100 signal times
                }
                signalCount++;

                // Calculate the average signal time from the last 100 signals (or fewer if not enough)
                const avgSignalTime = signalTimes.reduce((sum, time) => sum + time, 0) / signalTimes.length;

                // Calculate remaining signals
                const remainingCandles = totalLines - totalCandles;
                const estimatedTimeSec = remainingCandles * avgSignalTime;

                // Clear previous output (move to top of display area and clear down)
                process.stdout.write('\x1B[?25l'); // Hide cursor
                process.stdout.cursorTo(0, 0); // Move to top-left
                process.stdout.write('\x1B[0J'); // Clear from cursor to end of screen

                // Prepare progress and signal output
                const progressLine = `Progress => ${totalCandles}/${totalLines} candles (${((totalCandles / totalLines) * 100).toFixed(6)}%), ` +
                                    `Time: ${durationSec.toFixed(3)}s, ` +
                                    `Avg Time: ${(avgSignalTime * 1000).toFixed(3)}ms, ` +
                                    `ETA: ${formatTime(estimatedTimeSec)}`;
                const signalOutput = formatSignal(signal);

                // Write progress and signal
                process.stdout.write(`${progressLine}\n${signalOutput}\n`);
                process.stdout.write('\x1B[?25h'); // Show cursor
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

// Start by counting lines, then process
countLines().then((lineCount) => {
    totalLines = lineCount;
    console.log(`Total lines to process: ${totalLines}`);
    processCandles();
});