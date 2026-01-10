import fs from 'fs';
import path from 'path';
import Database from 'better-sqlite3';

import { isValidNumber } from './utils.js';

class HiveMind {
    #directoryPath;
    #ensembleSize;
    #inputSize;
    #numLayers;
    #numHeads;
    #headDim;
    #hiddenSize;
    #feedForwardSize;
    #contextWindow;
    #adaptiveWindow;
    #maxTrustHistory;
    #maxPerformanceHistory;
    #learningRate;
    #learningRateDecay;
    #swarmIntelligenceFactor;
    #gradientResetFrequency;
    #trainingStepCount = 0;
    #adaptiveLearningRate = [];
    #performanceScores = [];
    #agreementScores = [];
    #specializationScores = [];
    #historicalPerformance = [];
    #trustScoresHistory = [];
    #ensembleWeights = [];
    #attentionWeightMatrix = [];
    #attentionBias = [];
    #attentionMemory = [];
    #adaptiveContext = [];
    #specializationWeights = [];
    #transformers = [];
    #gradientAccumulation = [];

    constructor (dp, es, is) {
        this.#scaleAndSetDimensions(dp, es, is);

        const loadStatus = this.#loadState();

        if (!loadStatus.status && loadStatus.error) {
            console.log(`Load state failed! Error: ${loadStatus.error}. Trace: ${loadStatus.trace}`);
            process.exit();
        }
    }

    #loadState () {
        const dbPath = path.join(this.#directoryPath, 'hivemind_state.db');
        let db;

        try {
            if (!fs.existsSync(dbPath)) {
                return {
                    status : false
                };
            }

            db = new Database(dbPath, { readonly: true });

            const metadataStmt = db.prepare('SELECT key, value FROM metadata WHERE key = ?');
            const trainingStepCount = metadataStmt.get('trainingStepCount');
            if (trainingStepCount && isValidNumber(Number(trainingStepCount.value))) {
                this.#trainingStepCount = Number(trainingStepCount.value);
            }

            const ensembleWeightsStmt = db.prepare('SELECT idx, weight FROM ensemble_weights');
            const ensembleWeights = ensembleWeightsStmt.all();
            ensembleWeights.forEach(({ idx, weight }) => {
                if (isValidNumber(weight) && idx >= 0 && idx < this.#ensembleSize) {
                    this.#ensembleWeights[idx] = weight;
                }
            });

            const performanceScoresStmt = db.prepare('SELECT idx, score FROM performance_scores');
            const performanceScores = performanceScoresStmt.all();
            performanceScores.forEach(({ idx, score }) => {
                if (isValidNumber(score) && idx >= 0 && idx < this.#ensembleSize) {
                    this.#performanceScores[idx] = score;
                }
            });

            const agreementScoresStmt = db.prepare('SELECT idx, score FROM agreement_scores');
            const agreementScores = agreementScoresStmt.all();
            agreementScores.forEach(({ idx, score }) => {
                if (isValidNumber(score) && idx >= 0 && idx < this.#ensembleSize) {
                    this.#agreementScores[idx] = score;
                }
            });

            const specializationScoresStmt = db.prepare('SELECT idx, score FROM specialization_scores');
            const specializationScores = specializationScoresStmt.all();
            specializationScores.forEach(({ idx, score }) => {
                if (isValidNumber(score) && idx >= 0 && idx < this.#ensembleSize) {
                    this.#specializationScores[idx] = score;
                }
            });

            const historicalPerformanceStmt = db.prepare('SELECT idx, step, score FROM historical_performance ORDER BY idx, step');
            const historicalPerformance = historicalPerformanceStmt.all();
            historicalPerformance.forEach(({ idx, step, score }) => {
                if (isValidNumber(score) && idx >= 0 && idx < this.#ensembleSize && Number.isInteger(step)) {
                    this.#historicalPerformance[idx][step] = score;
                }
            });

            const trustScoresHistoryStmt = db.prepare('SELECT idx, step, score FROM trust_scores_history ORDER BY idx, step');
            const trustScoresHistory = trustScoresHistoryStmt.all();
            trustScoresHistory.forEach(({ idx, step, score }) => {
                if (isValidNumber(score) && idx >= 0 && idx < this.#ensembleSize && Number.isInteger(step)) {
                    this.#trustScoresHistory[idx][step] = score;
                }
            });

            const adaptiveLearningRateStmt = db.prepare('SELECT idx, rate FROM adaptive_learning_rate');
            const adaptiveLearningRates = adaptiveLearningRateStmt.all();
            adaptiveLearningRates.forEach(({ idx, rate }) => {
                if (isValidNumber(rate) && idx >= 0 && idx < this.#ensembleSize) {
                    this.#adaptiveLearningRate[idx] = rate;
                }
            });

            const attentionWeightMatrixStmt = db.prepare('SELECT idx, row, value FROM attention_weight_matrix');
            const attentionWeightMatrix = attentionWeightMatrixStmt.all();
            attentionWeightMatrix.forEach(({ idx, row, value }) => {
                if (isValidNumber(value) && idx >= 0 && idx < this.#ensembleSize && row >= 0 && row < this.#hiddenSize) {
                    this.#attentionWeightMatrix[idx][row] = value;
                }
            });

            const attentionBiasStmt = db.prepare('SELECT idx, row, value FROM attention_bias');
            const attentionBias = attentionBiasStmt.all();
            attentionBias.forEach(({ idx, row, value }) => {
                if (isValidNumber(value) && idx >= 0 && idx < this.#ensembleSize && row >= 0 && row < this.#hiddenSize) {
                    this.#attentionBias[idx][row] = value;
                }
            });

            const specializationWeightsStmt = db.prepare('SELECT idx, row, col, value FROM specialization_weights');
            const specializationWeights = specializationWeightsStmt.all();
            specializationWeights.forEach(({ idx, row, col, value }) => {
                if (
                    isValidNumber(value) &&
                    idx >= 0 && idx < this.#ensembleSize &&
                    row >= 0 && row < this.#hiddenSize &&
                    col >= 0 && col < this.#hiddenSize
                ) {
                    this.#specializationWeights[idx][row][col] = value;
                }
            });

            const attentionMemoryStmt = db.prepare('SELECT idx, window, seq, dim, value FROM attention_memory');
            const attentionMemory = attentionMemoryStmt.all();
            attentionMemory.forEach(({ idx, window, seq, dim, value }) => {
                if (
                    isValidNumber(value) &&
                    idx >= 0 && idx < this.#ensembleSize &&
                    window >= 0 && window < this.#contextWindow &&
                    seq >= 0 && seq < this.#inputSize &&
                    dim >= 0 && dim < this.#hiddenSize
                ) {
                    this.#attentionMemory[idx][window][seq][dim] = value;
                }
            });

            const adaptiveContextStmt = db.prepare('SELECT idx, window, seq, dim, value FROM adaptive_context');
            const adaptiveContext = adaptiveContextStmt.all();
            adaptiveContext.forEach(({ idx, window, seq, dim, value }) => {
                if (
                    isValidNumber(value) &&
                    idx >= 0 && idx < this.#ensembleSize &&
                    window >= 0 && window < this.#adaptiveWindow &&
                    seq >= 0 && seq < this.#inputSize &&
                    dim >= 0 && dim < this.#hiddenSize
                ) {
                    this.#adaptiveContext[idx][window][seq][dim] = value;
                }
            });

            const transformerWeightsStmt = db.prepare('SELECT idx, layer, weight_type, row, col, value FROM transformers');
            const transformerBiasesStmt = db.prepare('SELECT idx, layer, bias_type, row, value FROM transformer_biases');
            const transformerLayerNormStmt = db.prepare('SELECT idx, layer, norm_type, row, value FROM transformer_layer_norm');

            const transformerWeights = transformerWeightsStmt.all();
            const transformerBiases = transformerBiasesStmt.all();
            const transformerLayerNorm = transformerLayerNormStmt.all();

            transformerWeights.forEach(({ idx, layer, weight_type, row, col, value }) => {
                if (!isValidNumber(value) || idx < 0 || idx >= this.#ensembleSize) return;

                if (weight_type === 'outputWeights' && layer === -1) {
                    if (row >= 0 && row < this.#hiddenSize && col >= 0 && col < 1) {
                        this.#transformers[idx].outputWeights[row][col] = value;
                    }
                } else if (layer >= 0 && layer < this.#numLayers) {
                    if (['Wq', 'Wk', 'Wv', 'Wo'].includes(weight_type)) {
                        if (row >= 0 && row < this.#hiddenSize && col >= 0 && col < this.#hiddenSize) {
                            this.#transformers[idx].attentionWeights[layer][weight_type][row][col] = value;
                        }
                    } else if (['gate_proj', 'up_proj'].includes(weight_type)) {
                        if (row >= 0 && row < this.#hiddenSize && col >= 0 && col < this.#feedForwardSize) {
                            this.#transformers[idx].ffnWeights[layer][weight_type][row][col] = value;
                        }
                    } else if (weight_type === 'down_proj') {
                        if (row >= 0 && row < this.#feedForwardSize && col >= 0 && col < this.#hiddenSize) {
                            this.#transformers[idx].ffnWeights[layer].down_proj[row][col] = value;
                        }
                    }
                }
            });

            transformerBiases.forEach(({ idx, layer, bias_type, row, value }) => {
                if (!isValidNumber(value) || idx < 0 || idx >= this.#ensembleSize) return;

                if (bias_type === 'outputBias' && layer === -1 && row >= 0 && row < 1) {
                    this.#transformers[idx].outputBias[row] = value;
                }
            });

            transformerLayerNorm.forEach(({ idx, layer, norm_type, row, value }) => {
                if (!isValidNumber(value) || idx < 0 || idx >= this.#ensembleSize || layer < 0 || layer >= this.#numLayers || row < 0 || row >= this.#hiddenSize) return;

                if (norm_type === 'gamma1') {
                    this.#transformers[idx].layerNormWeights[layer].gamma1[row] = value;
                } else if (norm_type === 'gamma2') {
                    this.#transformers[idx].layerNormWeights[layer].gamma2[row] = value;
                }
            });

            const gradientAccumulationStmt = db.prepare('SELECT idx, layer, weight_type, row, col, value FROM gradient_accumulation');
            const gradientAccumulations = gradientAccumulationStmt.all();

            gradientAccumulations.forEach(({ idx, layer, weight_type, row, col, value }) => {
                if (!isValidNumber(value) || idx < 0 || idx >= this.#ensembleSize) return;

                const grad = this.#gradientAccumulation[idx];

                if (layer === -1) {
                    if (weight_type === 'outputWeights') {
                        if (row >= 0 && row < this.#hiddenSize && col >= 0 && col < 1) {
                            grad.outputWeights[row][col] = value;
                        }
                    } else if (weight_type === 'outputBias') {
                        if (row >= 0 && row < 1) {
                            grad.outputBias[row] = value;
                        }
                    } else if (weight_type === 'attentionBias') {
                        if (row >= 0 && row < this.#hiddenSize) {
                            grad.attentionBias[row] = value;
                        }
                    } else if (weight_type === 'attentionWeightMatrix') {
                        if (row >= 0 && row < this.#hiddenSize) {
                            grad.attentionWeightMatrix[row] = value;
                        }
                    } else if (weight_type === 'specializationWeights') {
                        if (row >= 0 && row < this.#hiddenSize && col >= 0 && col < this.#hiddenSize) {
                            grad.specializationWeights[row][col] = value;
                        }
                    }
                } else if (layer >= 0 && layer < this.#numLayers) {
                    if (['Wq', 'Wk', 'Wv', 'Wo'].includes(weight_type)) {
                        if (row >= 0 && row < this.#hiddenSize && col >= 0 && col < this.#hiddenSize) {
                            grad.attentionWeights[layer][weight_type][row][col] = value;
                        }
                    } else if (['gate_proj', 'up_proj'].includes(weight_type)) {
                        if (row >= 0 && row < this.#hiddenSize && col >= 0 && col < this.#feedForwardSize) {
                            grad.ffnWeights[layer][weight_type][row][col] = value;
                        }
                    } else if (weight_type === 'down_proj') {
                        if (row >= 0 && row < this.#feedForwardSize && col >= 0 && col < this.#hiddenSize) {
                            grad.ffnWeights[layer].down_proj[row][col] = value;
                        }
                    } else if (['gamma1', 'gamma2'].includes(weight_type)) {
                        if (row >= 0 && row < this.#hiddenSize) {
                            grad.layerNormWeights[layer][weight_type][row] = value;
                        }
                    }
                }
            });

            return { status: true };
        } catch (error) {
            return {
                status: false,
                error: error.message,
                trace: error.stack
            };
        } finally {
            if (db) {
                db.close();
            }
        }
    }

    #saveState () {
        const dbPath = path.join(this.#directoryPath, 'hivemind_state.db');
        let db;

        try {
            db = new Database(dbPath, { fileMustExist: false });
            db.pragma('journal_mode = WAL');
            db.pragma('synchronous = NORMAL');

            db.exec('BEGIN TRANSACTION');

            db.exec(`
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );
                CREATE TABLE IF NOT EXISTS ensemble_weights (
                    idx INTEGER PRIMARY KEY,
                    weight REAL
                );
                CREATE TABLE IF NOT EXISTS performance_scores (
                    idx INTEGER PRIMARY KEY,
                    score REAL
                );
                CREATE TABLE IF NOT EXISTS agreement_scores (
                    idx INTEGER PRIMARY KEY,
                    score REAL
                );
                CREATE TABLE IF NOT EXISTS specialization_scores (
                    idx INTEGER PRIMARY KEY,
                    score REAL
                );
                CREATE TABLE IF NOT EXISTS historical_performance (
                    idx INTEGER,
                    step INTEGER,
                    score REAL,
                    PRIMARY KEY (idx, step)
                );
                CREATE TABLE IF NOT EXISTS trust_scores_history (
                    idx INTEGER,
                    step INTEGER,
                    score REAL,
                    PRIMARY KEY (idx, step)
                );
                CREATE TABLE IF NOT EXISTS adaptive_learning_rate (
                    idx INTEGER PRIMARY KEY,
                    rate REAL
                );
                CREATE TABLE IF NOT EXISTS attention_weight_matrix (
                    idx INTEGER,
                    row INTEGER,
                    value REAL,
                    PRIMARY KEY (idx, row)
                );
                CREATE TABLE IF NOT EXISTS attention_bias (
                    idx INTEGER,
                    row INTEGER,
                    value REAL,
                    PRIMARY KEY (idx, row)
                );
                CREATE TABLE IF NOT EXISTS specialization_weights (
                    idx INTEGER,
                    row INTEGER,
                    col INTEGER,
                    value REAL,
                    PRIMARY KEY (idx, row, col)
                );
                CREATE TABLE IF NOT EXISTS attention_memory (
                    idx INTEGER,
                    window INTEGER,
                    seq INTEGER,
                    dim INTEGER,
                    value REAL,
                    PRIMARY KEY (idx, window, seq, dim)
                );
                CREATE TABLE IF NOT EXISTS adaptive_context (
                    idx INTEGER,
                    window INTEGER,
                    seq INTEGER,
                    dim INTEGER,
                    value REAL,
                    PRIMARY KEY (idx, window, seq, dim)
                );
                CREATE TABLE IF NOT EXISTS transformers (
                    idx INTEGER,
                    layer INTEGER,
                    weight_type TEXT,
                    row INTEGER,
                    col INTEGER,
                    value REAL,
                    PRIMARY KEY (idx, layer, weight_type, row, col)
                );
                CREATE TABLE IF NOT EXISTS transformer_biases (
                    idx INTEGER,
                    layer INTEGER,
                    bias_type TEXT,
                    row INTEGER,
                    value REAL,
                    PRIMARY KEY (idx, layer, bias_type, row)
                );
                CREATE TABLE IF NOT EXISTS transformer_layer_norm (
                    idx INTEGER,
                    layer INTEGER,
                    norm_type TEXT,
                    row INTEGER,
                    value REAL,
                    PRIMARY KEY (idx, layer, norm_type, row)
                );
                CREATE TABLE IF NOT EXISTS gradient_accumulation (
                    idx INTEGER,
                    layer INTEGER,
                    weight_type TEXT,
                    row INTEGER,
                    col INTEGER,
                    value REAL,
                    PRIMARY KEY (idx, layer, weight_type, row, col)
                );
            `);

            const insertMetadata = db.prepare('INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)');
            insertMetadata.run('trainingStepCount', (this.#trainingStepCount ?? 0).toString());

            const insertEnsembleWeights = db.prepare('INSERT OR REPLACE INTO ensemble_weights (idx, weight) VALUES (?, ?)');
            this.#ensembleWeights.forEach((weight, idx) => {
                if (isValidNumber(weight)) {
                    insertEnsembleWeights.run(idx, weight);
                }
            });

            const insertPerformanceScores = db.prepare('INSERT OR REPLACE INTO performance_scores (idx, score) VALUES (?, ?)');
            this.#performanceScores.forEach((score, idx) => {
                if (isValidNumber(score)) {
                    insertPerformanceScores.run(idx, score);
                }
            });

            const insertAgreementScores = db.prepare('INSERT OR REPLACE INTO agreement_scores (idx, score) VALUES (?, ?)');
            this.#agreementScores.forEach((score, idx) => {
                if (isValidNumber(score)) {
                    insertAgreementScores.run(idx, score);
                }
            });

            const insertSpecializationScores = db.prepare('INSERT OR REPLACE INTO specialization_scores (idx, score) VALUES (?, ?)');
            this.#specializationScores.forEach((score, idx) => {
                if (isValidNumber(score)) {
                    insertSpecializationScores.run(idx, score);
                }
            });

            const insertHistoricalPerformance = db.prepare('INSERT OR REPLACE INTO historical_performance (idx, step, score) VALUES (?, ?, ?)');
            this.#historicalPerformance.forEach((history, idx) => {
                history.forEach((score, step) => {
                    if (isValidNumber(score)) {
                        insertHistoricalPerformance.run(idx, step, score);
                    }
                });
            });

            const insertTrustScoresHistory = db.prepare('INSERT OR REPLACE INTO trust_scores_history (idx, step, score) VALUES (?, ?, ?)');
            this.#trustScoresHistory.forEach((history, idx) => {
                history.forEach((score, step) => {
                    if (isValidNumber(score)) {
                        insertTrustScoresHistory.run(idx, step, score);
                    }
                });
            });

            const insertAdaptiveLearningRate = db.prepare('INSERT OR REPLACE INTO adaptive_learning_rate (idx, rate) VALUES (?, ?)');
            this.#adaptiveLearningRate.forEach((rate, idx) => {
                if (isValidNumber(rate)) {
                    insertAdaptiveLearningRate.run(idx, rate);
                }
            });

            const insertAttentionWeightMatrix = db.prepare('INSERT OR REPLACE INTO attention_weight_matrix (idx, row, value) VALUES (?, ?, ?)');
            this.#attentionWeightMatrix.forEach((weights, idx) => {
                weights.forEach((value, row) => {
                    if (isValidNumber(value)) {
                        insertAttentionWeightMatrix.run(idx, row, value);
                    }
                });
            });

            const insertAttentionBias = db.prepare('INSERT OR REPLACE INTO attention_bias (idx, row, value) VALUES (?, ?, ?)');
            this.#attentionBias.forEach((biases, idx) => {
                biases.forEach((value, row) => {
                    if (isValidNumber(value)) {
                        insertAttentionBias.run(idx, row, value);
                    }
                });
            });

            const insertSpecializationWeights = db.prepare('INSERT OR REPLACE INTO specialization_weights (idx, row, col, value) VALUES (?, ?, ?, ?)');
            this.#specializationWeights.forEach((matrix, idx) => {
                matrix.forEach((rowArr, r) => {
                    rowArr.forEach((value, c) => {
                        if (isValidNumber(value)) {
                            insertSpecializationWeights.run(idx, r, c, value);
                        }
                    });
                });
            });

            const insertAttentionMemory = db.prepare('INSERT OR REPLACE INTO attention_memory (idx, window, seq, dim, value) VALUES (?, ?, ?, ?, ?)');
            this.#attentionMemory.forEach((memory, idx) => {
                memory.forEach((window, w) => {
                    window.forEach((seq, s) => {
                        seq.forEach((value, d) => {
                            if (isValidNumber(value)) {
                                insertAttentionMemory.run(idx, w, s, d, value);
                            }
                        });
                    });
                });
            });

            const insertAdaptiveContext = db.prepare('INSERT OR REPLACE INTO adaptive_context (idx, window, seq, dim, value) VALUES (?, ?, ?, ?, ?)');
            this.#adaptiveContext.forEach((memory, idx) => {
                memory.forEach((window, w) => {
                    window.forEach((seq, s) => {
                        seq.forEach((value, d) => {
                            if (isValidNumber(value)) {
                                insertAdaptiveContext.run(idx, w, s, d, value);
                            }
                        });
                    });
                });
            });

            const insertTransformerWeights = db.prepare('INSERT OR REPLACE INTO transformers (idx, layer, weight_type, row, col, value) VALUES (?, ?, ?, ?, ?, ?)');
            const insertTransformerLayerNorm = db.prepare('INSERT OR REPLACE INTO transformer_layer_norm (idx, layer, norm_type, row, value) VALUES (?, ?, ?, ?, ?)');
            const insertTransformerBiases = db.prepare('INSERT OR REPLACE INTO transformer_biases (idx, layer, bias_type, row, value) VALUES (?, ?, ?, ?, ?)');

            this.#transformers.forEach((transformer, idx) => {
                transformer.attentionWeights.forEach((layerWeights, layer) => {
                    ['Wq', 'Wk', 'Wv', 'Wo'].forEach(weightType => {
                        const matrix = layerWeights[weightType];
                        matrix.forEach((rowArr, r) => {
                            rowArr.forEach((value, c) => {
                                if (isValidNumber(value)) {
                                    insertTransformerWeights.run(idx, layer, weightType, r, c, value);
                                }
                            });
                        });
                    });
                });

                transformer.ffnWeights.forEach((layerFFN, layer) => {
                    ['gate_proj', 'up_proj', 'down_proj'].forEach(weightType => {
                        const matrix = layerFFN[weightType];
                        matrix.forEach((rowArr, r) => {
                            rowArr.forEach((value, c) => {
                                if (isValidNumber(value)) {
                                    insertTransformerWeights.run(idx, layer, weightType, r, c, value);
                                }
                            });
                        });
                    });
                });

                transformer.layerNormWeights.forEach((layerNorm, layer) => {
                    ['gamma1', 'gamma2'].forEach(normType => {
                        layerNorm[normType].forEach((value, r) => {
                            if (isValidNumber(value)) {
                                insertTransformerLayerNorm.run(idx, layer, normType, r, value);
                            }
                        });
                    });
                });

                transformer.outputWeights.forEach((rowArr, r) => {
                    rowArr.forEach((value, c) => {
                        if (isValidNumber(value)) {
                            insertTransformerWeights.run(idx, -1, 'outputWeights', r, c, value);
                        }
                    });
                });

                transformer.outputBias.forEach((value, r) => {
                    if (isValidNumber(value)) {
                        insertTransformerBiases.run(idx, -1, 'outputBias', r, value);
                    }
                });
            });

            const insertGradientAccumulation = db.prepare('INSERT OR REPLACE INTO gradient_accumulation (idx, layer, weight_type, row, col, value) VALUES (?, ?, ?, ?, ?, ?)');

            this.#gradientAccumulation.forEach((grad, idx) => {
                grad.outputWeights.forEach((rowArr, r) => {
                    rowArr.forEach((value, c) => {
                        if (isValidNumber(value)) {
                            insertGradientAccumulation.run(idx, -1, 'outputWeights', r, c, value);
                        }
                    });
                });
                grad.outputBias.forEach((value, r) => {
                    if (isValidNumber(value)) {
                        insertGradientAccumulation.run(idx, -1, 'outputBias', r, 0, value);
                    }
                });

                grad.attentionWeights.forEach((layerGrad, layer) => {
                    ['Wq', 'Wk', 'Wv', 'Wo'].forEach(weightType => {
                        const matrix = layerGrad[weightType];
                        matrix.forEach((rowArr, r) => {
                            rowArr.forEach((value, c) => {
                                if (isValidNumber(value)) {
                                    insertGradientAccumulation.run(idx, layer, weightType, r, c, value);
                                }
                            });
                        });
                    });
                });

                grad.ffnWeights.forEach((layerGrad, layer) => {
                    ['gate_proj', 'up_proj', 'down_proj'].forEach(weightType => {
                        const matrix = layerGrad[weightType];
                        matrix.forEach((rowArr, r) => {
                            rowArr.forEach((value, c) => {
                                if (isValidNumber(value)) {
                                    insertGradientAccumulation.run(idx, layer, weightType, r, c, value);
                                }
                            });
                        });
                    });
                });

                grad.layerNormWeights.forEach((layerGrad, layer) => {
                    ['gamma1', 'gamma2'].forEach(normType => {
                        layerGrad[normType].forEach((value, r) => {
                            if (isValidNumber(value)) {
                                insertGradientAccumulation.run(idx, layer, normType, r, 0, value);
                            }
                        });
                    });
                });

                grad.attentionBias.forEach((value, r) => {
                    if (isValidNumber(value)) {
                        insertGradientAccumulation.run(idx, -1, 'attentionBias', r, 0, value);
                    }
                });
                grad.attentionWeightMatrix.forEach((value, r) => {
                    if (isValidNumber(value)) {
                        insertGradientAccumulation.run(idx, -1, 'attentionWeightMatrix', r, 0, value);
                    }
                });
                grad.specializationWeights.forEach((rowArr, r) => {
                    rowArr.forEach((value, c) => {
                        if (isValidNumber(value)) {
                            insertGradientAccumulation.run(idx, -1, 'specializationWeights', r, c, value);
                        }
                    });
                });
            });

            db.exec('COMMIT');

            return { status: true };
        } catch (error) {
            if (db) db.exec('ROLLBACK');
            return {
                status: false,
                error: error.message,
                trace: error.stack
            };
        } finally {
            if (db) db.close();
        }
    }
    
    #scaleAndSetDimensions (dp, es, is) {
        this.#directoryPath = dp;
        this.#ensembleSize = Math.max(1, es);
        this.#inputSize = is;

        const logEs = Math.log10(this.#ensembleSize);
        const normalized = Math.min(1.0, logEs / 3.0);
        const sqrtEs = Math.sqrt(this.#ensembleSize);
        const log2SqrtEs = Math.log2(sqrtEs);

        const numLayersRaw = 6.0 - 4.0 * normalized;
        this.#numLayers = Math.max(2, Math.round(numLayersRaw));

        const numHeadsRaw = 8.0 - 6.0 * normalized;
        this.#numHeads = Math.max(2, Math.round(numHeadsRaw));

        const hsExponent = Math.max(2, Math.min(4, 4 - log2SqrtEs));
        const hs = Math.pow(2, Math.floor(hsExponent));
        this.#headDim = hs * Math.ceil(this.#numHeads / 4);
        this.#hiddenSize = this.#numHeads * this.#headDim;

        const ffnMultiplier = 8.0 - 4.0 * normalized;
        this.#feedForwardSize = Math.round(this.#hiddenSize * ffnMultiplier);

        const memoryFactor = 1 + 0.5 * logEs;
        this.#contextWindow = Math.round(this.#hiddenSize * 2 * memoryFactor);
        this.#adaptiveWindow = Math.max(1, Math.round(this.#contextWindow * 0.25));
        this.#maxTrustHistory = Math.round(this.#contextWindow * 2);
        this.#maxPerformanceHistory = Math.round(this.#contextWindow * 4);

        const baseLrUnscaled = 0.12 / Math.max(this.#inputSize, 1);
        const sizeScale = Math.pow(this.#hiddenSize, -0.18);
        this.#learningRate = Number(Math.min(0.0025, baseLrUnscaled * sizeScale).toPrecision(6));
        this.#learningRateDecay = Number((this.#learningRate / 10).toPrecision(6));

        this.#swarmIntelligenceFactor = Number((0.05 + 0.90 * normalized).toFixed(6));

        this.#gradientResetFrequency = Math.min(100, 10 + Math.ceil((this.#ensembleSize - 1) / 10));

        this.#performanceScores = Array(this.#ensembleSize).fill(0);
        this.#agreementScores = Array(this.#ensembleSize).fill(0);
        this.#specializationScores = Array(this.#ensembleSize).fill(0);
        this.#trustScoresHistory = Array(this.#ensembleSize).fill().map(() => [0]);
        this.#historicalPerformance = Array(this.#ensembleSize).fill().map(() => [0]);
        this.#adaptiveLearningRate = Array(this.#ensembleSize).fill(this.#learningRate);

        this.#ensembleWeights = Array(this.#ensembleSize).fill().map(() => 
            Math.max(0, 1 / this.#ensembleSize + (Math.random() - 0.5) * 0.1 / this.#ensembleSize)
        );
        this.#normalizeEnsembleWeights();

        this.#attentionMemory = Array(this.#ensembleSize).fill().map(() =>
            Array(this.#contextWindow).fill().map(() => Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0)))
        );

        this.#adaptiveContext = Array(this.#ensembleSize).fill().map(() =>
            Array(this.#adaptiveWindow).fill().map(() => Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0)))
        );

        this.#attentionWeightMatrix = Array(this.#ensembleSize).fill().map(() =>
            this.#dynamicInit(this.#hiddenSize, 1, 0, 1).map(row => row[0])
        );

        this.#attentionBias = Array(this.#ensembleSize).fill().map(() =>
            Array(this.#hiddenSize).fill().map(() => (Math.random() - 0.5) * Math.sqrt(4 / this.#hiddenSize))
        );

        this.#specializationWeights = Array(this.#ensembleSize).fill().map(() =>
            Array(this.#hiddenSize).fill().map((_, j) =>
                Array(this.#hiddenSize).fill().map((_, k) => {
                    const baseWeights = this.#dynamicInit(this.#hiddenSize, this.#hiddenSize, 0, 1);
                    const scale = 1 + 0.1 * (j + k / this.#hiddenSize);
                    return baseWeights[j][k] * scale;
                })
            )
        );

        this.#transformers = this.#setTransformerStructure();

        this.#gradientAccumulation = this.#setGradientStructure();
    }

    #dynamicInit (rows, cols, layerIndex, totalLayers, customK = null) {
        let baseK = customK !== null ? customK : 2.2;
        let fanInScale = Math.min(1.0, 1000 / rows);
        let depthScale = Math.pow(totalLayers, -layerIndex / totalLayers);
        let k = baseK * fanInScale * depthScale;
        k = Math.max(1.5, Math.min(k, 3.0));

        return Array(rows).fill().map(() =>
            Array(cols).fill().map(() => (Math.random() - 0.5) * Math.sqrt(k / rows))
        );
    }

    #setTransformerStructure () {
        return Array(this.#ensembleSize).fill().map(() => ({
            attentionWeights: Array(this.#numLayers).fill().map((_, layerIndex) => ({
                Wq: this.#dynamicInit(this.#hiddenSize, this.#hiddenSize, layerIndex, this.#numLayers),
                Wk: this.#dynamicInit(this.#hiddenSize, this.#hiddenSize, layerIndex, this.#numLayers),
                Wv: this.#dynamicInit(this.#hiddenSize, this.#hiddenSize, layerIndex, this.#numLayers),
                Wo: this.#dynamicInit(this.#hiddenSize, this.#hiddenSize, layerIndex, this.#numLayers)
            })),

            ffnWeights: Array(this.#numLayers).fill().map((_, layerIndex) => ({
                gate_proj: this.#dynamicInit(this.#hiddenSize, this.#feedForwardSize, layerIndex, this.#numLayers),
                up_proj: this.#dynamicInit(this.#hiddenSize, this.#feedForwardSize, layerIndex, this.#numLayers),
                down_proj: this.#dynamicInit(this.#feedForwardSize, this.#hiddenSize, layerIndex, this.#numLayers)
            })),
            
            layerNormWeights: Array(this.#numLayers).fill().map(() => ({
                gamma1: Array(this.#hiddenSize).fill(1.0),
                gamma2: Array(this.#hiddenSize).fill(1.0)
            })),

            outputWeights: this.#dynamicInit(this.#hiddenSize, 1, this.#numLayers, this.#numLayers + 1, 2.1),
            outputBias: Array(1).fill(0)
        }));
    }

    #setGradientStructure () {
        return Array(this.#ensembleSize).fill().map(() => ({
            outputWeights: Array(this.#hiddenSize).fill().map(() => Array(1).fill(0)),
            outputBias: Array(1).fill(0),

            attentionWeights: Array(this.#numLayers).fill().map(() => ({
                Wq: Array(this.#hiddenSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
                Wk: Array(this.#hiddenSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
                Wv: Array(this.#hiddenSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
                Wo: Array(this.#hiddenSize).fill().map(() => Array(this.#hiddenSize).fill(0))
            })),

            ffnWeights: Array(this.#numLayers).fill().map(() => ({
                gate_proj: Array(this.#hiddenSize).fill().map(() => Array(this.#feedForwardSize).fill(0)),
                up_proj: Array(this.#hiddenSize).fill().map(() => Array(this.#feedForwardSize).fill(0)),
                down_proj: Array(this.#feedForwardSize).fill().map(() => Array(this.#hiddenSize).fill(0))
            })),

            layerNormWeights: Array(this.#numLayers).fill().map(() => ({
                gamma1: Array(this.#hiddenSize).fill(0),
                gamma2: Array(this.#hiddenSize).fill(0)
            })),

            attentionBias: Array(this.#hiddenSize).fill(0),
            attentionWeightMatrix: Array(this.#hiddenSize).fill(0),
            specializationWeights: Array(this.#hiddenSize).fill().map(() => Array(this.#hiddenSize).fill(0))
        }));
    }

    #silu (x) {
        if (!isValidNumber(x)) return 0;
        const sig = 1 / (1 + Math.exp(-x));
        return x * sig;
    }

    #siluDerivative (x) {
        if (!isValidNumber(x)) return 0;
        const sig = 1 / (1 + Math.exp(-x));
        return sig * (1 + x * (1 - sig));
    }

    #sigmoid (x) {
        return isValidNumber(x) ? 1 / (1 + Math.exp(-Math.min(Math.max(x, -100), 100))) : 0;
    }

    #softmax (arr) {
        if (!arr.every(isValidNumber)) return arr.map(() => 1 / arr.length);
        const max = Math.max(...arr);
        const exp = arr.map(x => Math.exp(x - max));
        const sum = exp.reduce((a, b) => a + b, 0) || 1;
        return exp.map(x => x / sum);
    }

    #rmsNorm (x, gamma, eps = 1e-6) {
        if (
            !Array.isArray(x) || x.length !== this.#hiddenSize ||
            !Array.isArray(gamma) || gamma.length !== this.#hiddenSize ||
            !x.every(isValidNumber) || !gamma.every(isValidNumber)
        ) {
            return Array(this.#hiddenSize).fill(0);
        }

        const sq_sum = x.reduce((sum, val) => sum + val * val, 0);
        const rms = Math.sqrt(sq_sum / this.#hiddenSize + eps);
        return x.map((val, i) => val * gamma[i] / rms);
    }

    #applyRoPE (matrix, startPos = 0) {
        if (!Array.isArray(matrix) || matrix.length === 0 || !matrix[0] || matrix[0].length !== this.#hiddenSize) return;
        const head_dim = this.#hiddenSize / this.#numHeads;
        if (head_dim % 2 !== 0) return;
        const base = 10000.0;
        const seqLen = matrix.length;

        for (let pos = 0; pos < seqLen; pos++) {
            const absPos = startPos + pos;
            for (let h = 0; h < this.#numHeads; h++) {
                const offset = h * head_dim;
                for (let i = 0; i < head_dim / 2; i++) {
                    const exp = -2.0 * i / head_dim;
                    const theta = absPos * Math.pow(base, exp);
                    const cos = Math.cos(theta);
                    const sin = Math.sin(theta);
                    const idx1 = offset + 2 * i;
                    const idx2 = offset + 2 * i + 1;
                    const x = matrix[pos][idx1];
                    const y = matrix[pos][idx2];
                    if (!isValidNumber(x) || !isValidNumber(y)) continue;
                    matrix[pos][idx1] = x * cos - y * sin;
                    matrix[pos][idx2] = x * sin + y * cos;
                }
            }
        }
    }

    #weightedMean (protos) {
        if (!Array.isArray(protos) || protos.length === 0) {
            return Array(this.#hiddenSize).fill(0);
        }
        let totalSize = 0;
        for (const p of protos) {
            totalSize += p.size || 0;
        }
        if (totalSize === 0) {
            return Array(this.#hiddenSize).fill(0);
        }
        const rep = Array(this.#hiddenSize).fill(0);
        for (const p of protos) {
            const weight = p.size / totalSize;
            for (let j = 0; j < this.#hiddenSize; j++) {
                rep[j] += (p.mean[j] || 0) * weight;
            }
        }
        return rep;
    }

    #maxPairwiseKernel (protosA, protosB, gamma = 4.0) {
        if (protosA.length === 0 || protosB.length === 0) return 0;
        let maxSim = 0;
        for (const a of protosA) {
            for (const b of protosB) {
                const sim = this.#kernelSimilarity(a, b, gamma);
                if (sim > maxSim) maxSim = sim;
            }
        }
        return maxSim;
    }

    #cosineSimilarity (a, b) {
        let dot = 0, normA = 0, normB = 0;
        for (let i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] ** 2;
            normB += b[i] ** 2;
        }
        const denom = Math.sqrt(normA * normB) + 1e-8;
        return denom > 0 ? dot / denom : 0;
    }

    #activationEntropy (entry) {
        let entropy = 0;
        let totalMass = 0;
        for (let p = 0; p < entry.length; p++) {
            for (let j = 0; j < entry[p].length; j++) {
                const val = Math.abs(entry[p][j]);
                if (val > 1e-6) {
                    const p = val;
                    entropy -= p * Math.log(p + 1e-12);
                    totalMass += val;
                }
            }
        }
        return totalMass > 0 ? entropy / totalMass : 0;
    }

    #poolMultiPrototype (entry, maxAdditionalProtos = 8, thresholdFactor = 4.0) {
        if (
            !Array.isArray(entry) ||
            entry.length === 0 ||
            !entry[0] ||
            !Array.isArray(entry[0]) ||
            entry[0].length !== this.#hiddenSize
        ) {
            return [];
        }
        const seqLen = entry.length;
        const hiddenSize = this.#hiddenSize;

        const globalMean = new Array(hiddenSize).fill(0);
        for (let i = 0; i < seqLen; i++) {
            const row = entry[i];
            for (let j = 0; j < hiddenSize; j++) {
                globalMean[j] += row[j];
            }
        }
        for (let j = 0; j < hiddenSize; j++) globalMean[j] /= seqLen;
        const globalVariance = new Array(hiddenSize).fill(0);
        for (let i = 0; i < seqLen; i++) {
            const row = entry[i];
            for (let j = 0; j < hiddenSize; j++) {
                const diff = row[j] - globalMean[j];
                globalVariance[j] += diff * diff;
            }
        }
        for (let j = 0; j < hiddenSize; j++) globalVariance[j] /= seqLen;
        let protos = [{
            mean: globalMean.slice(),
            variance: globalVariance.map(v => Math.max(v, 1e-6)),
            size: seqLen
        }];
        if (seqLen <= 1) return protos;

        const avgVar = globalVariance.reduce((a, b) => a + b, 0) / hiddenSize;
        const thresholdSq = avgVar * hiddenSize * thresholdFactor;

        for (let i = 0; i < seqLen; i++) {
            const point = entry[i];
            let minDistSq = Infinity;
            let bestProtoIdx = 0;
            for (let c = 0; c < protos.length; c++) {
                let d = 0;
                for (let j = 0; j < hiddenSize; j++) {
                    const diff = point[j] - protos[c].mean[j];
                    d += diff * diff;
                }
                if (d < minDistSq) {
                    minDistSq = d;
                    bestProtoIdx = c;
                }
            }

            if (protos.length <= maxAdditionalProtos && minDistSq > thresholdSq) {
                protos.push({
                    mean: point.slice(),
                    variance: new Array(hiddenSize).fill(1e-6),
                    size: 1
                });
            } else {
                const proto = protos[bestProtoIdx];
                const oldMean = proto.mean.slice();
                proto.size++;
                for (let j = 0; j < hiddenSize; j++) {
                    const delta = point[j] - oldMean[j];
                    proto.mean[j] += delta / proto.size;
                    const newDiff = point[j] - proto.mean[j];
                    proto.variance[j] += delta * newDiff;
                }
            }
        }

        for (const proto of protos) {
            if (proto.size > 1) {
                for (let j = 0; j < hiddenSize; j++) {
                    proto.variance[j] /= (proto.size - 1);
                    proto.variance[j] = Math.max(proto.variance[j], 1e-6);
                }
            }
        }
        return protos;
    }

    #kernelSimilarity (protoA, protoB, gamma = 4.0) {
        let dist = 0;
        const eps = 1e-6;
        for (let j = 0; j < this.#hiddenSize; j++) {
            const diff = protoA.mean[j] - protoB.mean[j];
            const pooledVar = protoA.variance[j] + protoB.variance[j] + eps;
            dist += diff * diff / pooledVar;
        }
        dist /= this.#hiddenSize;
        return Math.exp(-gamma * dist);
    }

    #selectTopKRelevantEntries (transformerIdx, currentProtos, K = 12) {
        const memory = this.#attentionMemory[transformerIdx];
        const adaptive = this.#adaptiveContext[transformerIdx];
        const pastEntries = [...memory, ...adaptive];

        if (pastEntries.length <= K) {
            return pastEntries;
        }

        const scores = pastEntries.map(entry => {
            const entryProtos = this.#poolMultiPrototype(entry);
            return this.#maxPairwiseKernel(currentProtos, entryProtos, 4.0);
        });

        const indexedScores = scores.map((score, index) => ({ score, index }));
        indexedScores.sort((a, b) => b.score - a.score);

        const topKIndices = indexedScores.slice(0, K).map(item => item.index);
        return topKIndices.map(idx => pastEntries[idx]);
    }

    #computeMemoryScore (memoryEntry, attentionScores, transformerIdx, entryIndex, ignoreRecency = false) {
        let sum = 0, absSum = 0, sqSum = 0, cubeSum = 0, fourthSum = 0;
        let count = 0, nonZero = 0;
        for (let i = 0; i < memoryEntry.length; i++) {
            for (let j = 0; j < memoryEntry[i].length; j++) {
                const val = memoryEntry[i][j];
                if (isValidNumber(val)) {
                    const absVal = Math.abs(val);
                    sum += val;
                    absSum += absVal;
                    sqSum += val * val;
                    cubeSum += val * val * val;
                    fourthSum += val * val * val * val;
                    count++;
                    if (absVal > 1e-3) nonZero++;
                }
            }
        }

        const mean = count > 0 ? sum / count : 0;
        const variance = count > 0 ? (sqSum / count - mean * mean) : 0;
        const std = Math.sqrt(Math.max(variance, 0));
        const skew = std > 0 ? (cubeSum / count - 3 * mean * variance - mean * mean * mean) / (std * std * std) : 0;
        const kurtosis = variance > 0 ? (fourthSum / count) / (variance * variance) - 3 : 0;

        const varianceScore = Math.tanh(std);
        const skewScore = Math.tanh(Math.abs(skew));
        const kurtosisScore = Math.tanh(Math.abs(kurtosis));
        const entropyScore = Math.tanh(this.#activationEntropy(memoryEntry));

        const sparsity = count > 0 ? nonZero / count : 0;
        const sparsityScore = 1 - Math.abs(sparsity - 0.5) * 2;

        const magnitudeScore = Math.sqrt(sqSum);

        let attentionSharpness = 0;
        if (attentionScores) {
            let totalEntropy = 0, queryCount = 0;
            for (let h = 0; h < attentionScores.length; h++) {
                for (let i = 0; i < attentionScores[h].length; i++) {
                    let entropy = 0, sumP = 0;
                    for (let j = 0; j < attentionScores[h][i].length; j++) {
                        const p = attentionScores[h][i][j];
                        if (p > 0) {
                            entropy -= p * Math.log(p + 1e-12);
                            sumP += p;
                        }
                    }
                    if (sumP > 0.1) {
                        totalEntropy += entropy;
                        queryCount++;
                    }
                }
            }
            const avgEntropy = queryCount > 0 ? totalEntropy / queryCount : Math.log(this.#inputSize);
            attentionSharpness = Math.exp(-avgEntropy / Math.log(this.#inputSize));
        }

        const specScore = Math.min(Math.max(this.#specializationScores[transformerIdx] || 0.5, 0), 1);
        let perf = Math.max(0, Math.min(1, this.#performanceScores[transformerIdx] || 0.5));
        const confidenceBoost = 0.25 * perf * specScore;

        const entryProtos = this.#poolMultiPrototype(memoryEntry);

        let uniqueness = 1.0;
        if (!ignoreRecency && this.#attentionMemory[transformerIdx].length > 1) {
            if (entryProtos.length > 0) {
                const entryRep = this.#weightedMean(entryProtos);

                let centroidRep = Array(this.#hiddenSize).fill(0);
                let totalSize = 0;

                const pastMemory = this.#attentionMemory[transformerIdx];
                const maxSampled = this.#adaptiveWindow;

                let indicesToUse = [];
                if (pastMemory.length > maxSampled) {
                    for (let i = 0; i < maxSampled; i++) {
                        indicesToUse.push(Math.floor(i * pastMemory.length / maxSampled));
                    }
                } else {
                    for (let i = 0; i < pastMemory.length; i++) {
                        indicesToUse.push(i);
                    }
                }

                indicesToUse = indicesToUse.filter(idx => idx !== entryIndex);

                for (const idx of indicesToUse) {
                    const mem = pastMemory[idx];
                    const mProtos = this.#poolMultiPrototype(mem);
                    for (const pr of mProtos) {
                        for (let j = 0; j < this.#hiddenSize; j++) {
                            centroidRep[j] += pr.mean[j] * pr.size;
                        }
                        totalSize += pr.size;
                    }
                }

                if (totalSize > 0) {
                    for (let j = 0; j < this.#hiddenSize; j++) centroidRep[j] /= totalSize;
                    uniqueness = 1 - this.#cosineSimilarity(entryRep, centroidRep);
                    uniqueness = Math.max(0, uniqueness);
                } else {
                    uniqueness = 1.0;
                }
            }
        }

        let clusterDiversityScore = 0;
        if (entryProtos.length > 1) {
            let clusterEntropy = 0;
            const totalMass = entryProtos.reduce((s, p) => s + p.size, 0);
            if (totalMass > 0) {
                for (const p of entryProtos) {
                    const prob = p.size / totalMass;
                    if (prob > 1e-6) {
                        clusterEntropy -= prob * Math.log(prob);
                    }
                }
            }
            const maxEnt = Math.log(entryProtos.length);
            clusterDiversityScore = maxEnt > 0 ? clusterEntropy / maxEnt : 0;
        }

        let baseScore =
            0.2 * varianceScore +
            0.15 * skewScore +
            0.15 * kurtosisScore +
            0.1 * entropyScore +
            0.1 * sparsityScore +
            0.15 * magnitudeScore +
            0.15 * attentionSharpness +
            0.1 * clusterDiversityScore;

        baseScore = baseScore * (1 + confidenceBoost) * Math.pow(uniqueness + 0.5, 1.5);

        let diversityFactor = 1.0;
        if (!ignoreRecency && this.#attentionMemory[transformerIdx].length > 5) {
            const memoryLength = this.#attentionMemory[transformerIdx].length;
            const recentCount = Math.max(4, Math.floor(memoryLength * 0.2));

            let distSum = 0;
            for (let r = 1; r <= recentCount; r++) {
                const recentIndex = memoryLength - r;
                if (recentIndex < 0) break;
                if (recentIndex === entryIndex) continue;

                const recentEntry = this.#attentionMemory[transformerIdx][recentIndex];
                const recentProtos = this.#poolMultiPrototype(recentEntry);
                const overlapSim = this.#maxPairwiseKernel(entryProtos, recentProtos);
                distSum += 1 - overlapSim;
            }
            diversityFactor = 1 + 1.0 * (distSum / recentCount);
        }

        const age = ignoreRecency ? 0 : this.#attentionMemory[transformerIdx].length - 1 - entryIndex;
        const recencyFactor = Math.exp(-0.03 * age) * (1 + 1.0 / (1 + age / 5));

        return baseScore * diversityFactor * recencyFactor;
    }

    #pruneMemory (transformerIdx, contextWindow, latestScores) {
        const memory = this.#attentionMemory[transformerIdx];
        if (memory.length <= contextWindow) return;

        const numToKeep = contextWindow;

        const candidates = memory.map((entry, index) => ({
            index,
            entry,
            protos: this.#poolMultiPrototype(entry),
            score: this.#computeMemoryScore(
                entry,
                index === memory.length - 1 ? latestScores : null,
                transformerIdx,
                index
            )
        }));

        let selected = [];

        const numForcedRecent = Math.min(20, Math.floor(numToKeep * 0.5));
        for (let r = 0; r < numForcedRecent; r++) {
            const candIdx = candidates.length - 1 - r;
            if (candIdx >= 0) {
                selected.push(candidates[candIdx]);
            }
        }

        while (selected.length < numToKeep) {
            let bestCand = null;
            let bestMarginal = -Infinity;

            for (const cand of candidates) {
                if (selected.some(s => s.index === cand.index)) continue;

                let maxOverlap = 0;
                for (const sel of selected) {
                    const overlap = this.#maxPairwiseKernel(cand.protos, sel.protos);
                    if (overlap > maxOverlap) maxOverlap = overlap;
                }

                const dist = 1 - maxOverlap;
                const marginal = cand.score * (1 + 2.0 * dist);

                if (marginal > bestMarginal) {
                    bestMarginal = marginal;
                    bestCand = cand;
                }
            }

            if (bestCand === null) break;
            selected.push(bestCand);
        }

        if (selected.length < numToKeep) {
            candidates.sort((a, b) => b.score - a.score);
            for (const cand of candidates) {
                if (!selected.some(s => s.index === cand.index)) {
                    selected.push(cand);
                    if (selected.length >= numToKeep) break;
                }
            }
        }

        selected.sort((a, b) => a.index - b.index);
        this.#attentionMemory[transformerIdx] = selected.map(s => s.entry);
    }

    #multiHeadAttention (x, layer, transformerIdx, training = true) {
        if (
            !Array.isArray(x) ||
            x.length !== this.#inputSize ||
            !x.every(row => Array.isArray(row) && row.length === this.#hiddenSize && row.every(isValidNumber)) ||
            !Number.isInteger(transformerIdx) ||
            transformerIdx < 0 ||
            transformerIdx >= this.#ensembleSize ||
            !layer || !layer.Wq || !layer.Wk || !layer.Wv || !layer.Wo ||
            !layer.Wq.every(row => Array.isArray(row) && row.length === this.#hiddenSize && row.every(isValidNumber)) ||
            !layer.Wk.every(row => Array.isArray(row) && row.length === this.#hiddenSize && row.every(isValidNumber)) ||
            !layer.Wv.every(row => Array.isArray(row) && row.length === this.#hiddenSize && row.every(isValidNumber)) ||
            !layer.Wo.every(row => Array.isArray(row) && row.length === this.#hiddenSize && row.every(isValidNumber))
        ) {
            return {
                output: Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
                preWoOutput: Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
                Q: Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
                K: Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
                V: Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
                scores: Array(this.#numHeads).fill().map(() =>
                    Array(this.#inputSize).fill().map(() => Array(this.#inputSize).fill(0))
                ),
                probs: Array(this.#numHeads).fill().map(() =>
                    Array(this.#inputSize).fill().map(() => Array(this.#inputSize).fill(0))
                )
            };
        }

        const headSize = this.#hiddenSize / this.#numHeads;
        if (this.#hiddenSize % this.#numHeads !== 0 || headSize <= 0 || headSize % 2 !== 0) {
            console.log(`Critical error: hiddenSize (${this.#hiddenSize}) must be divisible by numHeads (${this.#numHeads}) and headSize must be even for RoPE.`);
            process.exit();
        }

        const Q = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
        const K = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
        const V = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));

        for (let i = 0; i < this.#inputSize; i++) {
            for (let j = 0; j < this.#hiddenSize; j++) {
                for (let k = 0; k < this.#hiddenSize; k++) {
                    const specWeight = isValidNumber(this.#specializationWeights[transformerIdx][k % this.#hiddenSize][j])
                        ? Math.min(Math.max(1 + this.#specializationScores[transformerIdx] * this.#specializationWeights[transformerIdx][k % this.#hiddenSize][j], 0.5), 1.5)
                        : 1;
                    Q[i][j] += isValidNumber(x[i][k]) && isValidNumber(layer.Wq[k][j])
                        ? x[i][k] * layer.Wq[k][j] * specWeight
                        : 0;
                    K[i][j] += isValidNumber(x[i][k]) && isValidNumber(layer.Wk[k][j])
                        ? x[i][k] * layer.Wk[k][j] * specWeight
                        : 0;
                    V[i][j] += isValidNumber(x[i][k]) && isValidNumber(layer.Wv[k][j])
                        ? x[i][k] * layer.Wv[k][j] * specWeight
                        : 0;
                }
            }
        }

        this.#applyRoPE(Q);
        this.#applyRoPE(K);

        const attentionScores = Array(this.#numHeads).fill().map(() =>
            Array(this.#inputSize).fill().map(() => Array(this.#inputSize).fill(0))
        );
        const attentionProbs = Array(this.#numHeads).fill().map(() =>
            Array(this.#inputSize).fill().map(() => Array(this.#inputSize).fill(0))
        );

        for (let h = 0; h < this.#numHeads; h++) {
            for (let i = 0; i < this.#inputSize; i++) {
                for (let j = 0; j < this.#inputSize; j++) {
                    let sum = 0;
                    for (let k = 0; k < headSize; k++) {
                        const idx = h * headSize + k;
                        sum += isValidNumber(Q[i][idx]) && isValidNumber(K[j][idx])
                            ? Q[i][idx] * K[j][idx]
                            : 0;
                    }
                    attentionScores[h][i][j] = isValidNumber(sum)
                        ? sum / Math.sqrt(headSize)
                        : 0;
                }
                attentionProbs[h][i] = this.#softmax(attentionScores[h][i].map(score => isValidNumber(score) ? score : 0));
            }
        }

        const preWoOutput = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
        for (let h = 0; h < this.#numHeads; h++) {
            for (let i = 0; i < this.#inputSize; i++) {
                for (let j = 0; j < this.#inputSize; j++) {
                    for (let k = 0; k < headSize; k++) {
                        const idx = h * headSize + k;
                        preWoOutput[i][idx] += isValidNumber(attentionProbs[h][i][j]) && isValidNumber(V[j][idx])
                            ? attentionProbs[h][i][j] * V[j][idx]
                            : 0;
                    }
                }
            }
        }

        const finalOutput = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
        for (let i = 0; i < this.#inputSize; i++) {
            for (let j = 0; j < this.#hiddenSize; j++) {
                for (let k = 0; k < this.#hiddenSize; k++) {
                    const specWeight = isValidNumber(this.#specializationWeights[transformerIdx][k % this.#hiddenSize][j])
                        ? Math.min(Math.max(1 + this.#specializationScores[transformerIdx] * this.#specializationWeights[transformerIdx][k % this.#hiddenSize][j], 0.5), 1.5)
                        : 1;
                    finalOutput[i][j] += isValidNumber(preWoOutput[i][k]) && isValidNumber(layer.Wo[k][j])
                        ? preWoOutput[i][k] * layer.Wo[k][j] * specWeight
                        : 0;
                }
            }
        }

        let outputToAdaptive = finalOutput;
        if (training) {
            let perf = Math.max(0, Math.min(1, this.#performanceScores[transformerIdx] || 0.5));
            const specScore = Math.min(Math.max(this.#specializationScores[transformerIdx] || 0.5, 0), 1);
            const confidence = perf * (0.7 + 0.3 * specScore);
            const maxNoiseScale = 0.03 + 0.3 * (1 - confidence);

            outputToAdaptive = finalOutput.map(tokenVec => {
                const noiseVec = tokenVec.map(() => (Math.random() - 0.5) * maxNoiseScale);
                return tokenVec.map((val, j) => val + noiseVec[j] * (1 + (this.#specializationWeights[transformerIdx][j % this.#hiddenSize][j % this.#hiddenSize] || 1)));
            });
        } else {
            let perf = Math.max(0, Math.min(1, this.#performanceScores[transformerIdx] || 0.5));
            if (perf < 0.6) {
                const scale = 0.01 * (1 - perf);
                outputToAdaptive = finalOutput.map(tokenVec =>
                    tokenVec.map((val, j) => val * (1 + scale * Math.sin(j * 0.1)))
                );
            }
        }

        this.#adaptiveContext[transformerIdx].push(outputToAdaptive);
        if (this.#adaptiveContext[transformerIdx].length > this.#adaptiveWindow) {
            this.#adaptiveContext[transformerIdx].shift();
        }

        if (training) {
            this.#attentionMemory[transformerIdx].push(finalOutput);
            this.#pruneMemory(transformerIdx, this.#contextWindow, attentionScores);
        }

        return {
            output: finalOutput,
            preWoOutput,
            Q,
            K,
            V,
            scores: attentionScores,
            probs: attentionProbs
        };
    }

    #feedForward (x, layer, transformerIdx) {
        if (
            !Array.isArray(x) || x.length !== this.#hiddenSize ||
            !x.every(isValidNumber) ||
            !Number.isInteger(transformerIdx) ||
            transformerIdx < 0 ||
            transformerIdx >= this.#ensembleSize ||
            !layer || !layer.gate_proj || !layer.up_proj || !layer.down_proj ||
            !layer.gate_proj.every(row => Array.isArray(row) && row.length === this.#feedForwardSize && row.every(isValidNumber)) ||
            !layer.up_proj.every(row => Array.isArray(row) && row.length === this.#feedForwardSize && row.every(isValidNumber)) ||
            !layer.down_proj.every(row => Array.isArray(row) && row.length === this.#hiddenSize && row.every(isValidNumber))
        ) {
            return Array(this.#hiddenSize).fill(0);
        }

        const gate = Array(this.#feedForwardSize).fill(0);
        const up = Array(this.#feedForwardSize).fill(0);

        for (let j = 0; j < this.#feedForwardSize; j++) {
            for (let i = 0; i < this.#hiddenSize; i++) {
                const specWeight = isValidNumber(this.#specializationWeights[transformerIdx][i % this.#hiddenSize][j % this.#hiddenSize])
                    ? Math.min(Math.max(1 + this.#specializationScores[transformerIdx] * this.#specializationWeights[transformerIdx][i % this.#hiddenSize][j % this.#hiddenSize], 0.5), 1.5)
                    : 1;
                gate[j] += isValidNumber(x[i]) && isValidNumber(layer.gate_proj[i][j])
                    ? x[i] * layer.gate_proj[i][j] * specWeight
                    : 0;
                up[j] += isValidNumber(x[i]) && isValidNumber(layer.up_proj[i][j])
                    ? x[i] * layer.up_proj[i][j] * specWeight
                    : 0;
            }
        }

        let activated = Array(this.#feedForwardSize).fill(0);
        for (let j = 0; j < this.#feedForwardSize; j++) {
            activated[j] = this.#silu(gate[j]) * up[j];
        }

        const output = Array(this.#hiddenSize).fill(0);
        for (let j = 0; j < this.#hiddenSize; j++) {
            for (let i = 0; i < this.#feedForwardSize; i++) {
                const specWeight = isValidNumber(this.#specializationWeights[transformerIdx][j % this.#hiddenSize][i % this.#hiddenSize])
                    ? Math.min(Math.max(1 + this.#specializationScores[transformerIdx] * this.#specializationWeights[transformerIdx][j % this.#hiddenSize][i % this.#hiddenSize], 0.5), 1.5)
                    : 1;
                output[j] += isValidNumber(activated[i]) && isValidNumber(layer.down_proj[i][j])
                    ? activated[i] * layer.down_proj[i][j] * specWeight
                    : 0;
            }
        }

        return output;
    }

    #contextAwareAttention (inputs, transformerIdx) {
        if (
            !Array.isArray(inputs) ||
            inputs.length !== this.#inputSize ||
            !inputs.every(isValidNumber) ||
            !Number.isInteger(transformerIdx) ||
            transformerIdx < 0 ||
            transformerIdx >= this.#ensembleSize
        ) {
            return Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
        }

        let perf = Math.max(0, Math.min(1, this.#performanceScores[transformerIdx] || 0.5));

        const inputProjection = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
        for (let i = 0; i < this.#inputSize; i++) {
            for (let j = 0; j < this.#hiddenSize; j++) {
                const specWeight = isValidNumber(this.#specializationWeights[transformerIdx][i % this.#hiddenSize][j])
                    ? Math.min(Math.max(1 + this.#specializationScores[transformerIdx] * this.#specializationWeights[transformerIdx][i % this.#hiddenSize][j], 0.5), 1.5)
                    : 1;
                inputProjection[i][j] = isValidNumber(inputs[i])
                    ? inputs[i] * specWeight * (1 + this.#swarmIntelligenceFactor)
                    : 0;
            }
        }

        const rawComponent = inputProjection.map(row => row.slice());

        const currentProtos = this.#poolMultiPrototype(rawComponent, 8, 2);
        if (currentProtos.length === 0) {
            return rawComponent;
        }

        const longTermMemoryLength = this.#attentionMemory[transformerIdx].length;
        const shortTermMemoryLength = this.#adaptiveContext[transformerIdx].length;
        const fillFactor = (longTermMemoryLength / this.#contextWindow + shortTermMemoryLength / this.#adaptiveWindow) / 2;

        const dynamicEntrySelection = Math.max(12, Math.floor(this.#adaptiveWindow / 2));
        const selectedPast = this.#selectTopKRelevantEntries(transformerIdx, currentProtos, dynamicEntrySelection);

        if (selectedPast.length === 0) {
            return rawComponent;
        }

        const tokenPastLen = selectedPast.length * this.#inputSize;

        const pastStates = Array(tokenPastLen).fill().map(() => Array(this.#hiddenSize).fill(0));
        let pOffset = 0;
        for (const entry of selectedPast) {
            for (let pos = 0; pos < this.#inputSize; pos++) {
                const targetRow = pastStates[pOffset + pos];
                const sourceRow = entry[pos];
                for (let j = 0; j < this.#hiddenSize; j++) {
                    targetRow[j] = isValidNumber(sourceRow[j]) ? sourceRow[j] : 0;
                }
            }
            pOffset += this.#inputSize;
        }

        const transformer = this.#transformers[transformerIdx];
        const numLayers = this.#numLayers;
        if (numLayers === 0) return rawComponent;

        const avgWq = Array(this.#hiddenSize).fill().map(() => Array(this.#hiddenSize).fill(0));
        const avgWk = Array(this.#hiddenSize).fill().map(() => Array(this.#hiddenSize).fill(0));
        const avgWv = Array(this.#hiddenSize).fill().map(() => Array(this.#hiddenSize).fill(0));
        const avgWo = Array(this.#hiddenSize).fill().map(() => Array(this.#hiddenSize).fill(0));

        const scale = 1.0 / numLayers;
        for (let l = 0; l < numLayers; l++) {
            const att = transformer.attentionWeights[l];
            for (let ii = 0; ii < this.#hiddenSize; ii++) {
                for (let jj = 0; jj < this.#hiddenSize; jj++) {
                    avgWq[ii][jj] += att.Wq[ii][jj] * scale;
                    avgWk[ii][jj] += att.Wk[ii][jj] * scale;
                    avgWv[ii][jj] += att.Wv[ii][jj] * scale;
                    avgWo[ii][jj] += att.Wo[ii][jj] * scale;
                }
            }
        }

        const headSize = this.#hiddenSize / this.#numHeads;

        const K = Array(tokenPastLen).fill().map(() => Array(this.#hiddenSize).fill(0));
        const V = Array(tokenPastLen).fill().map(() => Array(this.#hiddenSize).fill(0));

        for (let seqi = 0; seqi < tokenPastLen; seqi++) {
            for (let j = 0; j < this.#hiddenSize; j++) {
                let sumK = 0;
                let sumV = 0;
                for (let kk = 0; kk < this.#hiddenSize; kk++) {
                    const specW = isValidNumber(this.#specializationWeights[transformerIdx][kk][j])
                        ? Math.min(Math.max(1 + this.#specializationScores[transformerIdx] * this.#specializationWeights[transformerIdx][kk][j], 0.5), 1.5)
                        : 1;
                    sumK += pastStates[seqi][kk] * avgWk[kk][j] * specW || 0;
                    sumV += pastStates[seqi][kk] * avgWv[kk][j] * specW || 0;
                }
                K[seqi][j] = sumK;
                V[seqi][j] = sumV;
            }
        }

        this.#applyRoPE(K, 0);

        let globalDiversity = 0;
        if (tokenPastLen > this.#inputSize) {
            const numSamples = Math.min(16, tokenPastLen);
            const sampleIndices = [];
            for (let s = 0; s < numSamples; s++) {
                sampleIndices.push(Math.floor(Math.random() * tokenPastLen));
            }
            const centroid = Array(this.#hiddenSize).fill(0);
            for (const si of sampleIndices) {
                for (let j = 0; j < this.#hiddenSize; j++) {
                    centroid[j] += pastStates[si][j];
                }
            }
            for (let j = 0; j < this.#hiddenSize; j++) centroid[j] /= numSamples;

            for (const si of sampleIndices) {
                globalDiversity += 1 - this.#cosineSimilarity(pastStates[si], centroid);
            }
            globalDiversity /= numSamples;
        }
        const diversityScore = Math.tanh(globalDiversity * 6);

        let augmented = rawComponent.map(row => row.slice());
        const maxHops = selectedPast.length;

        for (let hop = 0; hop < maxHops; hop++) {
            const Q = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));

            for (let seqi = 0; seqi < this.#inputSize; seqi++) {
                for (let j = 0; j < this.#hiddenSize; j++) {
                    let sum = 0;
                    for (let kk = 0; kk < this.#hiddenSize; kk++) {
                        const specW = isValidNumber(this.#specializationWeights[transformerIdx][kk][j])
                            ? Math.min(Math.max(1 + this.#specializationScores[transformerIdx] * this.#specializationWeights[transformerIdx][kk][j], 0.5), 1.5)
                            : 1;
                        sum += augmented[seqi][kk] * avgWq[kk][j] * specW || 0;
                    }
                    Q[seqi][j] = sum;
                }
            }

            this.#applyRoPE(Q, tokenPastLen);

            let totalEntropy = 0;
            let queryCount = 0;

            const preWoOutput = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));

            for (let h = 0; h < this.#numHeads; h++) {
                const offset = h * headSize;
                for (let i = 0; i < this.#inputSize; i++) {
                    let maxScore = -Infinity;
                    const scores = Array(tokenPastLen).fill(0);
                    for (let jj = 0; jj < tokenPastLen; jj++) {
                        let sum = 0;
                        for (let kk = 0; kk < headSize; kk++) {
                            sum += Q[i][offset + kk] * K[jj][offset + kk] || 0;
                        }
                        scores[jj] = sum / Math.sqrt(headSize);
                        if (scores[jj] > maxScore) maxScore = scores[jj];
                    }

                    const expScores = scores.map(s => Math.exp(s - maxScore));
                    const sumExp = expScores.reduce((a, b) => a + b, 0) || 1;

                    let headEntropy = 0;
                    let sumP = 0;
                    for (let jj = 0; jj < tokenPastLen; jj++) {
                        const p = expScores[jj] / sumExp;
                        if (p > 1e-6) {
                            headEntropy -= p * Math.log(p);
                            sumP += p;
                        }
                        if (p > 0) {
                            for (let kk = 0; kk < headSize; kk++) {
                                preWoOutput[i][offset + kk] += p * V[jj][offset + kk];
                            }
                        }
                    }

                    if (sumP > 0.1) {
                        totalEntropy += headEntropy;
                        queryCount++;
                    }
                }
            }

            const logPast = Math.log(tokenPastLen + 1);
            const avgEntropy = queryCount > 0 ? totalEntropy / queryCount : logPast;
            const attentionSharpness = Math.exp(-avgEntropy / logPast);

            const gate = 4.0 * attentionSharpness + 2.5 * fillFactor + 2.0 * perf + 2.0 * diversityScore;
            let retrievedRatio = 1 / (1 + Math.exp(-gate));

            const retrieved = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
            for (let i = 0; i < this.#inputSize; i++) {
                for (let j = 0; j < this.#hiddenSize; j++) {
                    let sum = 0;
                    for (let kk = 0; kk < this.#hiddenSize; kk++) {
                        const specW = isValidNumber(this.#specializationWeights[transformerIdx][kk][j])
                            ? Math.min(Math.max(1 + this.#specializationScores[transformerIdx] * this.#specializationWeights[transformerIdx][kk][j], 0.5), 1.5)
                            : 1;
                        sum += preWoOutput[i][kk] * avgWo[kk][j] * specW || 0;
                    }
                    retrieved[i][j] = sum;
                }
            }

            for (let i = 0; i < this.#inputSize; i++) {
                for (let j = 0; j < this.#hiddenSize; j++) {
                    augmented[i][j] = augmented[i][j] * (1 - retrievedRatio) + retrieved[i][j] * retrievedRatio;
                }
            }

            if (hop >= 1 && retrievedRatio < 0.01) {
                break;
            }
        }

        return augmented;
    }

    #processTransformer (inputs, idx, computeIntermediates = false, training = true, isLast = false) {
        const transformer = this.#transformers[idx];

        let x = this.#contextAwareAttention(inputs, idx);

        const layerOutputs = computeIntermediates ? [x] : [];
        const activations = computeIntermediates ? [] : [];
        const attentionIntermediates = computeIntermediates ? [] : [];

        for (let layer = 0; layer < this.#numLayers; layer++) {
            const normX = x.map(row => this.#rmsNorm(row, transformer.layerNormWeights[layer].gamma1));

            const attentionResult = this.#multiHeadAttention(normX, transformer.attentionWeights[layer], idx, training);
            const attentionOutput = attentionResult.output;

            const attentionResidual = x.map((row, i) => row.map((val, j) => val + attentionOutput[i][j]));

            const normAttention = attentionResidual.map(row => this.#rmsNorm(row, transformer.layerNormWeights[layer].gamma2));

            const ffnOutputs = normAttention.map(row => this.#feedForward(row, transformer.ffnWeights[layer], idx));

            x = attentionResidual.map((row, i) => row.map((val, j) => val + ffnOutputs[i][j]));

            if (computeIntermediates) {
                layerOutputs.push(x);
                activations.push({ normX, attentionOutput, normAttention });
                attentionIntermediates.push({
                    Q: attentionResult.Q,
                    K: attentionResult.K,
                    V: attentionResult.V,
                    attentionScores: attentionResult.scores,
                    attentionProbs: attentionResult.probs,
                    preWoOutput: attentionResult.preWoOutput
                });
            }
        }

        let finalHidden = Array(this.#hiddenSize).fill(0);
        let validCount = 0;
        for (let pos = 0; pos < this.#inputSize; pos++) {
            let posHidden = x[pos];
            if (this.#numLayers > 0) {
                const lastLayerIdx = this.#numLayers - 1;
                posHidden = this.#rmsNorm(posHidden, transformer.layerNormWeights[lastLayerIdx].gamma2);
            }
            for (let j = 0; j < this.#hiddenSize; j++) {
                if (isValidNumber(posHidden[j])) {
                    finalHidden[j] += posHidden[j];
                }
            }
            validCount++;
        }
        if (validCount > 0) {
            finalHidden = finalHidden.map(v => v / validCount);
        }

        let output = Array(1).fill(0);
        for (let i = 0; i < this.#hiddenSize; i++) {
            output[0] += isValidNumber(finalHidden[i]) && isValidNumber(transformer.outputWeights[i][0])
                ? finalHidden[i] * transformer.outputWeights[i][0]
                : 0;
        }
        output[0] = isValidNumber(output[0]) && isValidNumber(transformer.outputBias[0])
            ? output[0] + transformer.outputBias[0]
            : output[0];

        if (isLast) this.#computeAttentionWeights(inputs);

        if (computeIntermediates) {
            return { output: output[0], layerOutputs, activations, attentionIntermediates };
        }
        return output[0];
    }

    #updateHiveState (inputs, target, intermediates, training, shouldUpdateMetrics, shouldReturn) {
        let probability;
        let outputs = [];
        const layerOutputs = [];
        const activations = [];
        const attentionIntermediates = [];

        if (intermediates) {
            this.#transformers.forEach((_, idx, arr) => {
                const result = this.#processTransformer(inputs, idx, true, training, idx === arr.length - 1);
                outputs[idx] = result.output;
                layerOutputs[idx] = result.layerOutputs;
                activations[idx] = result.activations;
                attentionIntermediates[idx] = result.attentionIntermediates;
            });
        } else {
            outputs = this.#transformers.map((_, idx, arr) => 
                this.#processTransformer(inputs, idx, false, training, idx === arr.length - 1)
            );
        }

        const finalOutput = this.#computeWeightedSum(outputs);
        probability = this.#sigmoid(finalOutput);

        if (shouldUpdateMetrics) this.#updateMetrics(inputs, outputs, target, probability);

        if (intermediates && shouldReturn) { 
            return { outputs, layerOutputs, activations, attentionIntermediates, probability }; 
        }

        if (!intermediates && shouldReturn) {
            return { outputs, probability };
        }
    }

    #computeWeightedSum (outputs) {
        return outputs.reduce((sum, out, idx) => {
            if (isValidNumber(out) && isValidNumber(this.#ensembleWeights[idx])) {
                return sum + out * this.#ensembleWeights[idx];
            }
            return sum;
        }, 0);
    }

    #computeAttentionWeights (inputs) {
        if (
            !Array.isArray(inputs) ||
            inputs.length !== this.#inputSize ||
            !inputs.every(isValidNumber)
        ) {
            return Array(this.#ensembleSize).fill(1 / this.#ensembleSize);
        }

        const attentionScores = Array(this.#ensembleSize).fill(0);
        for (let t = 0; t < this.#ensembleSize; t++) {
            const memory = this.#attentionMemory[t];
            const memLen = memory.length;

            const queries = inputs.map(x =>
                this.#attentionWeightMatrix[t].map(w => isValidNumber(x) && isValidNumber(w) ? x * w : 0)
            );

            const pooledMemory = Array(memLen).fill().map(() => Array(this.#hiddenSize).fill(0));
            for (let m = 0; m < memLen; m++) {
                for (let k = 0; k < this.#hiddenSize; k++) {
                    let sum = 0;
                    for (let i = 0; i < this.#inputSize; i++) {
                        sum += isValidNumber(memory[m][i][k]) ? memory[m][i][k] : 0;
                    }
                    pooledMemory[m][k] = sum / this.#inputSize;
                }
            }

            const keys = pooledMemory.map(pool =>
                pool.map((v, k) => isValidNumber(v) && isValidNumber(this.#attentionWeightMatrix[t][k]) ? v * this.#attentionWeightMatrix[t][k] : 0)
            );

            const values = keys;

            const biasSum = this.#attentionBias[t].reduce((sum, val) => sum + (isValidNumber(val) ? val : 0), 0);
            const avgBias = isValidNumber(biasSum) ? biasSum / this.#hiddenSize : 0;

            let score = 0;
            for (let i = 0; i < this.#inputSize; i++) {
                const rowScores = Array(memLen).fill(0);
                for (let j = 0; j < memLen; j++) {
                    let dotProduct = 0;
                    for (let k = 0; k < this.#hiddenSize; k++) {
                        dotProduct += isValidNumber(queries[i][k]) && isValidNumber(keys[j][k])
                            ? queries[i][k] * keys[j][k]
                            : 0;
                    }
                    rowScores[j] = isValidNumber(dotProduct)
                        ? dotProduct / Math.sqrt(this.#hiddenSize) + avgBias
                        : avgBias;
                }

                const attentionWeights = this.#softmax(rowScores.map(s => isValidNumber(s) ? s : 0));

                for (let j = 0; j < memLen; j++) {
                    for (let k = 0; k < this.#hiddenSize; k++) {
                        score += isValidNumber(attentionWeights[j]) && isValidNumber(values[j][k])
                            ? attentionWeights[j] * values[j][k]
                            : 0;
                    }
                }
            }

            score /= this.#inputSize;

            const performanceScore = isValidNumber(this.#performanceScores[t]) ? this.#performanceScores[t] : 0.5;
            const specializationBoost = 1 + (isValidNumber(this.#specializationScores[t])
                ? this.#specializationScores[t] * this.#swarmIntelligenceFactor
                : 0);

            attentionScores[t] = score * (0.7 + 0.2 * performanceScore + 0.1 * specializationBoost);
        }

        const weights = this.#softmax(attentionScores.map(score => isValidNumber(score) ? score : 0));

        const finalWeights = weights.map((w, idx) => {
            const performanceScore = isValidNumber(this.#performanceScores[idx]) ? this.#performanceScores[idx] : 0.5;
            const specializationFactor = 1 + (isValidNumber(this.#specializationScores[idx])
                ? this.#specializationScores[idx] * this.#swarmIntelligenceFactor
                : 0);
            const weight = 0.6 * w + 0.3 * performanceScore + 0.1 * specializationFactor;
            return isValidNumber(weight) && weight >= 0 ? weight : 1 / this.#ensembleSize;
        });

        const sum = finalWeights.reduce((s, w) => s + (isValidNumber(w) && w >= 0 ? w : 0), 0);
        if (sum <= 1e-6) {
            return Array(this.#ensembleSize).fill(1 / this.#ensembleSize);
        }

        this.#ensembleWeights = finalWeights.map(w => (isValidNumber(w) && w >= 0 ? w / sum : 1 / this.#ensembleSize));
        this.#normalizeEnsembleWeights();
    }

    #computeSpecializationScores (inputs, outputs) {
        if (
            !Array.isArray(inputs) ||
            inputs.length !== this.#inputSize ||
            !inputs.every(isValidNumber) ||
            !Array.isArray(outputs) ||
            outputs.length !== this.#ensembleSize ||
            !outputs.every(isValidNumber)
        ) {
            return;
        }

        const outputMean = outputs.reduce((sum, v) => sum + v, 0) / outputs.length;
        const outputStd = Math.sqrt(
            outputs.reduce((sum, v) => sum + (v - outputMean) ** 2, 0) / outputs.length
        ) || 1e-6;

        const zScores = outputs.map(out => (out - outputMean) / outputStd);

        this.#specializationScores = zScores.map((z, i) => {
            const performance = isValidNumber(this.#performanceScores[i]) ? this.#performanceScores[i] : 0.5;
            const magnitude = Math.abs(z);
            const bounded = 1 / (1 + Math.exp(-magnitude));
            return bounded * (0.5 + 0.5 * performance);
        });
    }

    #updatePerformanceScores (linearOutputs, target) {
        this.#performanceScores = this.#performanceScores.map((score, idx) => {
            const individualProbability = this.#sigmoid(linearOutputs[idx]);
            const brierScore = Math.pow(individualProbability - target, 2);
            const performance = 1 - brierScore;
            const newScore = 0.9 * score + 0.1 * (isValidNumber(performance) ? performance : 0);
            
            this.#historicalPerformance[idx].push(isValidNumber(newScore) ? newScore : 0);
            if (this.#historicalPerformance[idx].length > this.#maxPerformanceHistory) {
                this.#historicalPerformance[idx].shift();
            }
            
            return isValidNumber(newScore) ? newScore : 0;
        });
    }

    #updateAgreementScores (linearOutputs, finalProbability) {
        this.#agreementScores = this.#agreementScores.map((score, idx) => {
            const individualProbability = this.#sigmoid(linearOutputs[idx]);
            const absDiff = Math.abs(individualProbability - finalProbability);
            let agreement = 1 - absDiff;
            
            const avgHistoricalPerformance = this.#historicalPerformance[idx].length > 0
                ? this.#historicalPerformance[idx].reduce((sum, val) => sum + val, 0) / this.#historicalPerformance[idx].length
                : 0;
            const weightedAgreement = agreement * (0.5 + 0.5 * avgHistoricalPerformance);
            
            const newScore = 0.9 * score + 0.1 * (isValidNumber(weightedAgreement) ? weightedAgreement : 0);
            
            return isValidNumber(newScore) ? newScore : 0;
        });
    }

    #updateTrustScores () {
        const performanceMean = this.#performanceScores.reduce((sum, score) => sum + (isValidNumber(score) ? score : 0), 0) / this.#ensembleSize || 1;
        const performanceStd = Math.sqrt(
            this.#performanceScores.reduce((sum, score) => sum + ((isValidNumber(score) ? score : 0) - performanceMean) ** 2, 0) / this.#ensembleSize
        ) || 1;

        this.#trustScoresHistory = this.#trustScoresHistory.map((history, idx) => {
            const normalizedScore = isValidNumber(this.#performanceScores[idx]) && performanceStd > 0
                ? (this.#performanceScores[idx] - performanceMean) / performanceStd
                : 0;
            const agreementFactor = isValidNumber(this.#agreementScores[idx]) ? this.#agreementScores[idx] : 0.5;
            const historicalTrend = this.#historicalPerformance[idx].length > 1
                ? (this.#historicalPerformance[idx][this.#historicalPerformance[idx].length - 1] - this.#historicalPerformance[idx][0]) /
                this.#historicalPerformance[idx].length
                : 0;
            const specializationBoost = 1 + this.#specializationScores[idx] * this.#swarmIntelligenceFactor;
            const trustScore = this.#sigmoid(normalizedScore * (0.6 + 0.2 * agreementFactor + 0.1 * historicalTrend + 0.1 * specializationBoost));
            history.push(isValidNumber(trustScore) ? trustScore : 0.5);
            if (history.length > this.#maxTrustHistory) history.shift();
            return history;
        });
    }

    #adjustPerformanceScores () {
        const trustMomentum = 0.6;
        this.#performanceScores = this.#performanceScores.map((score, idx) => {
            const recentTrust = this.#trustScoresHistory[idx].slice(-5);
            const avgTrust = recentTrust.length > 0
                ? recentTrust.reduce((sum, val) => sum + (isValidNumber(val) ? val : 0), 0) / recentTrust.length
                : 0.5;
            const historicalWeight = this.#trustScoresHistory[idx].length / this.#maxTrustHistory;
            const specializationFactor = 1 + this.#specializationScores[idx] * this.#swarmIntelligenceFactor;
            return trustMomentum * (isValidNumber(score) ? score : 0) +
                (1 - trustMomentum) * avgTrust * (0.7 + 0.2 * historicalWeight + 0.1 * specializationFactor);
        });
    }

    #updateEnsembleWeights () {
        this.#ensembleWeights = this.#trustScoresHistory.map((history, idx) => {
            const recentTrust = history.slice(-5);
            const avgTrust = recentTrust.length > 0
                ? recentTrust.reduce((sum, val) => sum + (isValidNumber(val) ? val : 0), 0) / recentTrust.length
                : 1 / this.#ensembleSize;
            const specializationFactor = 1 + this.#specializationScores[idx] * this.#swarmIntelligenceFactor;
            return avgTrust * (0.8 + 0.2 * specializationFactor);
        });

        this.#normalizeEnsembleWeights();
    }

    #normalizeEnsembleWeights () {
        const sum = this.#ensembleWeights.reduce((s, w) => s + (isValidNumber(w) && w >= 0 ? w : 0), 0);

        if (sum <= 1e-6) {
            this.#ensembleWeights = Array(this.#ensembleSize).fill(1 / this.#ensembleSize);
            return;
        }

        this.#ensembleWeights = this.#ensembleWeights.map(w => 
            isValidNumber(w) && w >= 0 ? w / sum : 0
        );

        const finalSum = this.#ensembleWeights.reduce((s, w) => s + w, 0);
        if (Math.abs(finalSum - 1) > 1e-6) {
            const maxIndex = this.#ensembleWeights.indexOf(Math.max(...this.#ensembleWeights));
            this.#ensembleWeights[maxIndex] += 1 - finalSum;
        }
    }

    #updateAdaptiveLearningRates () {
        const recent_performance = this.#historicalPerformance.map(history => {
            const len = history.length;
            if (len === 0) return 0.5;
            const sum = history.reduce((s, v) => s + (isValidNumber(v) ? v : 0), 0);
            return sum / len;
        });

        const recent_trust = this.#trustScoresHistory.map(history => {
            const len = history.length;
            if (len === 0) return 0.5;
            const sum = history.reduce((s, v) => s + (isValidNumber(v) ? v : 0), 0);
            return sum / len;
        });

        const sorted_specialization = this.#specializationScores.slice().sort((a, b) => b - a);
        const max_spec = sorted_specialization[Math.floor(this.#ensembleSize * 0.05)] || 1e-10;
        const min_spec = sorted_specialization[this.#ensembleSize - 1] || 0;
        const spec_range = Math.max(max_spec - min_spec, 1e-10);
        const normalized_specialization = this.#specializationScores.map(score => 
            Math.min(Math.max((score - min_spec) / spec_range, 0), 1)
        );

        const composite_scores = recent_performance.map((perf, idx) => {
            const trust = recent_trust[idx];
            const spec = normalized_specialization[idx];
            const weighted_sum = 0.4 * perf + 0.3 * trust + 0.3 * spec;
            return this.#sigmoid(5 * weighted_sum);
        });

        const sorted_composite = composite_scores.slice().sort((a, b) => b - a);
        const thresholdIdx = Math.max(1, Math.floor(composite_scores.length * 0.25));
        const acceptable_threshold = sorted_composite[thresholdIdx] || 0.5;

        const min_lr = this.#learningRate * 0.5;
        const max_lr = this.#learningRate * 1.5;

        this.#adaptiveLearningRate = this.#adaptiveLearningRate.map((lr, idx) => {
            let newLr = lr;
            const score = composite_scores[idx];
            const score_diff = score - acceptable_threshold;

            const adjustment_magnitude = Math.max(-1, Math.min(1, score_diff * 2));
            if (score >= acceptable_threshold) {
                newLr *= (1 - this.#learningRateDecay * this.#learningRate * Math.abs(adjustment_magnitude) * 0.8);
            } else {
                newLr *= (1 + this.#learningRateDecay * this.#learningRate * Math.abs(adjustment_magnitude) * 1.2);
            }

            newLr = 0.95 * newLr + 0.05 * lr;

            return Math.max(min_lr, Math.min(newLr, max_lr));
        });
    }

    #updateMetrics (inputs, outputs, target, prob) {
        this.#updatePerformanceScores(outputs, target);
        this.#updateAgreementScores(outputs, prob);
        this.#computeSpecializationScores(inputs, outputs);
        this.#updateTrustScores();
        this.#adjustPerformanceScores();
        this.#updateEnsembleWeights();
        this.#updateAdaptiveLearningRates();
    }

    #computeSpectralNorm (matrix) {
        if (!Array.isArray(matrix)) {
            const error = new Error('Input is not an array');
            console.error('Error:', error.message, '\nStack:', error.stack);
            throw error
            return 1;
        }
        if (matrix.length === 0) {
            const error = new Error('Matrix is empty');
            console.error('Error:', error.message, '\nStack:', error.stack);
            throw error
            return 1;
        }
        if (!matrix.every(row => Array.isArray(row))) {
            const error = new Error('Matrix rows are not arrays');
            console.error('Error:', error.message, '\nStack:', error.stack);
            throw error
            return 1;
        }
        if (!matrix.every(row => row.length === matrix[0].length)) {
            const error = new Error('Matrix rows have inconsistent lengths');
            console.error('Error:', error.message, '\nStack:', error.stack);
            throw error
            return 1;
        }
        if (!matrix.every(row => row.every(isValidNumber))) {
            const error = new Error('Matrix contains invalid numbers');
            console.error('Error:', error.message, '\nStack:', error.stack);
            throw error
            return 1;
        }
        if (matrix[0].length === 0) {
            const error = new Error('Matrix has no columns');
            console.error('Error:', error.message, '\nStack:', error.stack);
            throw error
            return 1;
        }

        const rows = matrix.length;
        const cols = matrix[0].length;
        let u = Array(cols).fill().map(() => Math.random());
        let v = Array(rows).fill(0);

        const uNormInit = Math.sqrt(u.reduce((sum, x) => sum + (isValidNumber(x) ? x * x : 0), 0)) || 1;
        if (!isValidNumber(uNormInit)) {
            const error = new Error('Invalid initial u norm');
            console.error('Error:', error.message, '\nStack:', error.stack);
            throw error;
        }
        u = u.map(x => isValidNumber(x) ? x / uNormInit : 0);

        const maxIter = 20;
        const tolerance = 1e-6;
        let prevNorm = 0;
        for (let iter = 0; iter < maxIter; iter++) {
            v = Array(rows).fill(0);
            for (let i = 0; i < rows; i++) {
                for (let j = 0; j < cols; j++) {
                    if (isValidNumber(matrix[i][j]) && isValidNumber(u[j])) {
                        v[i] += matrix[i][j] * u[j];
                    }
                }
            }
            const vNorm = Math.sqrt(v.reduce((sum, x) => sum + (isValidNumber(x) ? x * x : 0), 0)) || 1;
            if (!isValidNumber(vNorm)) {
                const error = new Error(`Invalid v norm in iteration ${iter + 1}`);
                console.error('Error:', error.message, '\nStack:', error.stack);
                throw error;
            }
            v = v.map(x => isValidNumber(x) ? x / vNorm : 0);

            u = Array(cols).fill(0);
            for (let j = 0; j < cols; j++) {
                for (let i = 0; i < rows; i++) {
                    if (isValidNumber(matrix[i][j]) && isValidNumber(v[i])) {
                        u[j] += matrix[i][j] * v[i];
                    }
                }
            }
            const uNorm = Math.sqrt(u.reduce((sum, x) => sum + (isValidNumber(x) ? x * x : 0), 0)) || 1;
            if (!isValidNumber(uNorm)) {
                const error = new Error(`Invalid u norm in iteration ${iter + 1}`);
                console.error('Error:', error.message, '\nStack:', error.stack);
                throw error;
            }
            u = u.map(x => isValidNumber(x) ? x / uNorm : 0);

            let currentNorm = 0;
            const temp = Array(rows).fill(0);
            for (let i = 0; i < rows; i++) {
                for (let j = 0; j < cols; j++) {
                    if (isValidNumber(matrix[i][j]) && isValidNumber(u[j])) {
                        temp[i] += matrix[i][j] * u[j];
                    }
                }
            }
            for (let i = 0; i < rows; i++) {
                if (isValidNumber(temp[i]) && isValidNumber(v[i])) {
                    currentNorm += temp[i] * v[i];
                }
            }
            currentNorm = Math.abs(currentNorm) || 1;
            if (!isValidNumber(currentNorm)) {
                const error = new Error(`Invalid norm in iteration ${iter + 1}`);
                console.error('Error:', error.message, '\nStack:', error.stack);
                throw error;
            }

            if (iter > 0 && Math.abs(currentNorm - prevNorm) < tolerance * Math.max(1, currentNorm)) {
                return currentNorm;
            }
            prevNorm = currentNorm;
        }

        let norm = 0;
        const temp = Array(rows).fill(0);
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                if (isValidNumber(matrix[i][j]) && isValidNumber(u[j])) {
                    temp[i] += matrix[i][j] * u[j];
                }
            }
        }
        for (let i = 0; i < rows; i++) {
            if (isValidNumber(temp[i]) && isValidNumber(v[i])) {
                norm += temp[i] * v[i];
            }
        }
        const finalNorm = Math.abs(norm) || 1;
        if (!isValidNumber(finalNorm)) {
            const error = new Error('Invalid final norm');
            console.error('Error:', error.message, '\nStack:', error.stack);
            throw error;
        }

        return finalNorm;
    }

    #computeGradientNorm (grad, isMatrix) {
        if (isMatrix) {
            return Math.sqrt(grad.reduce((sum, row) => 
                sum + row.reduce((s, val) => s + (isValidNumber(val) ? val * val : 0), 0), 0)) || 1;
        } else {
            return Math.sqrt(grad.reduce((sum, val) => 
                sum + (isValidNumber(val) ? val * val : 0), 0)) || 1;
        }
    }

    #computeVariance (arr) {
        if (arr.length < 2) return 0;

        const median = arr.slice().sort((a, b) => a - b)[Math.floor(arr.length / 2)];
        const mad = arr.reduce((sum, val) => sum + Math.abs(val - median), 0) / Math.max(arr.length, 1);

        return Math.min(mad, 10);
    }

    #computeEMA (arr, beta) {
        if (arr.length === 0) return 0;

        let ema = arr[0];
        for (let i = 1; i < arr.length; i++) {
            ema = beta * ema + (1 - beta) * arr[i];
        }

        return ema;
    }

    #computeDualEMA (arr, betaShort, betaLong) {
        const shortEMA = this.#computeEMA(arr, betaShort);
        const longEMA = this.#computeEMA(arr, betaLong);
        const variance = this.#computeVariance(arr.slice(-10));
        const svrWeight = Math.min(0.8, Math.max(0.2, 0.5 * (1 - variance / (variance + 1))));

        return svrWeight * shortEMA + (1 - svrWeight) * longEMA;
    }

    #computeGradientConformity (norms) {
        if (norms.length < 2) return 1;

        const directions = norms.slice(1).map((val, i) => Math.sign(val - norms[i]));
        const consistency = directions.reduce((sum, val, i, arr) => sum + (i > 0 && val === arr[i - 1] ? 1 : 0), 0) / (directions.length || 1);

        return Math.min(1, Math.max(0.5, consistency));
    }

    #computeKernelRate (history) {
        const median = history.slice(-10).sort((a, b) => a - b)[Math.floor(history.length / 2)] || 0;
        const scaledHistory = history.slice(-10).map(val => (val - median) / (Math.abs(median) + 1e-10));
        const kernel = scaledHistory.map((val, i) => Math.exp(-0.1 * (scaledHistory.length - i - 1) ** 2));
        const kernelSum = kernel.reduce((sum, val) => sum + val, 0);

        return kernelSum > 0 ? kernel.reduce((sum, val, i) => sum + val * scaledHistory[i], 0) / kernelSum + 0.9 : 0.9;
    }

    #computeFractalDimension (arr) {
        if (arr.length < 2) return 1;

        const diffs = arr.slice(1).map((val, i) => Math.abs(val - arr[i]));
        const logDiffs = diffs.map(d => Math.log(Math.max(d, 1e-10)));
        const logScale = Math.log(arr.length);
        const fractalDim = logDiffs.length > 0 ? Math.abs(logDiffs.reduce((sum, val) => sum + val, 0) / logScale) : 1;

        return Math.min(2, Math.max(1, fractalDim));
    }

    #computePercentile (norms, percentile) {
        const sortedNorms = [...norms].sort((a, b) => a - b);
        const index = Math.floor(percentile * (sortedNorms.length - 1));

        return sortedNorms[index] || 1.0;
    }

    #computeNTKStability (norms, lossVariance) {
        if (norms.length < 2) return 1;

        const average = (arr) => arr.reduce((sum, val) => sum + val, 0) / Math.max(arr.length, 1);

        const medianNorm = norms.slice(-10).sort((a, b) => a - b)[Math.floor(norms.length / 2)] || 1;
        const bandwidth = Math.min(-0.01, Math.max(-0.1, -0.05 / (1 + lossVariance * medianNorm)));
        const kernel = norms.slice(-10).map((val, i) => Math.exp(bandwidth * (val - average(norms)) ** 2));
        const stability = kernel.reduce((sum, val) => sum + val, 0) / kernel.length;

        return Math.min(1.5, Math.max(0.5, stability));
    }

    #computeDynamicPercentile (norms, basePercentile) {
        const mad = this.#computeVariance(norms);
        const smoothness = this.#computeGradientConformity(norms);
        const adjustment = Math.min(0.05, (mad + 0.1 * smoothness) / (mad + smoothness + 1));

        return Math.min(0.99, Math.max(0.75, basePercentile - adjustment));
    }

    #computeSparseThreshold (norms) {
        const mad = this.#computeVariance(norms);

        return Math.max(1e-6, Math.min(1e-4, mad / 10));
    }

    #scaleGradientMatrix (gradMatrix, threshold, minScaleFactor, alpha, decay, sparseThreshold) {
        const spectralNorm = this.#computeSpectralNorm(gradMatrix);
        if (spectralNorm <= threshold) return;

        const scale = Math.max(minScaleFactor, Math.pow(threshold / spectralNorm, alpha) * decay);
        const quantScale = Math.ceil(scale * 255) / 255;
        for (let i = 0; i < gradMatrix.length; i++) {
            for (let j = 0; j < gradMatrix[i].length; j++) {
                if (Math.abs(gradMatrix[i][j]) > sparseThreshold) {
                    gradMatrix[i][j] *= quantScale;
                }
            }
        }
    }

    #scaleGradientVector (gradVector, threshold, minScaleFactor, alpha, decay, sparseThreshold) {
        const l2Norm = this.#computeGradientNorm(gradVector, false);
        if (l2Norm <= threshold) return;

        const scale = Math.max(minScaleFactor, Math.pow(threshold / l2Norm, alpha) * decay);
        const quantScale = Math.ceil(scale * 255) / 255;
        for (let i = 0; i < gradVector.length; i++) {
            if (Math.abs(gradVector[i]) > sparseThreshold) {
                gradVector[i] *= quantScale;
            }
        }
    }

    #scaleGradients (idx) {
        const baseAttentionPercentile = 0.95;
        const baseFfnPercentile = 0.9;
        const baseVectorPercentile = 0.8;

        const trustVariance = this.#computeVariance(this.#trustScoresHistory[idx].slice(-10));
        const lossVariance = this.#computeVariance(this.#historicalPerformance[idx].slice(-10));
        const stabilityMetric = 0.5 * lossVariance + 0.5 * trustVariance;

        const minScaleFactor = Math.max(0.01, 0.1 / (Math.sqrt(this.#ensembleSize) * (1 + stabilityMetric)));

        const emaBetaShort = Math.min(0.95, Math.max(0.8, this.#computeKernelRate(this.#historicalPerformance[idx])));
        const emaBetaLong = Math.min(0.999, Math.max(0.95, this.#computeKernelRate(this.#trustScoresHistory[idx])));

        const fractalDim = this.#computeFractalDimension(this.#historicalPerformance[idx].slice(-10));
        const componentWeights = {
            ensemble: Math.min(0.4, Math.max(0.1, 0.2 * (1 + 0.1 * fractalDim))),
            specialization: Math.min(0.4, Math.max(0.1, 0.2 * (1 + 0.15 * fractalDim * Math.min(this.#specializationScores[idx], 10)))),
            trust: Math.min(0.4, Math.max(0.1, 0.2 * (1 - 0.1 * fractalDim))),
            historicalPerformance: Math.min(0.4, Math.max(0.1, 0.2 * (1 + 0.05 * fractalDim))),
            performance: Math.min(0.4, Math.max(0.1, 0.2 * (1 + 0.05 * fractalDim)))
        };
        const totalWeight = Object.values(componentWeights).reduce((sum, w) => sum + w, 0);
        Object.keys(componentWeights).forEach(key => {
            componentWeights[key] /= Math.max(totalWeight, 1e-8);
        });

        const trust_ema = this.#computeDualEMA(this.#trustScoresHistory[idx], emaBetaShort, emaBetaLong);
        const perf_ema = this.#computeDualEMA(this.#historicalPerformance[idx], emaBetaShort, emaBetaLong);

        const robustnessFactor = Math.min(0.9, Math.max(0.1, 0.5 + 0.5 * (trust_ema / (Math.abs(trust_ema) + Math.abs(perf_ema) + 1e-10))));
        const compositeScore = (
            this.#ensembleWeights[idx] * componentWeights.ensemble +
            this.#specializationScores[idx] * componentWeights.specialization +
            trust_ema * componentWeights.trust +
            perf_ema * componentWeights.historicalPerformance +
            this.#performanceScores[idx] * componentWeights.performance
        ) * robustnessFactor;

        const totalCompositeScore = this.#ensembleWeights.reduce((sum, _, i) => {
            const memberTrustEma = this.#computeDualEMA(this.#trustScoresHistory[i], emaBetaShort, emaBetaLong);
            const memberPerfEma = this.#computeDualEMA(this.#historicalPerformance[i], emaBetaShort, emaBetaLong);
            const memberRobustness = Math.min(0.9, Math.max(0.1, 0.5 + 0.5 * (memberTrustEma / (Math.abs(memberTrustEma) + Math.abs(memberPerfEma) + 1e-10))));
            return sum + (
                this.#ensembleWeights[i] * componentWeights.ensemble +
                this.#specializationScores[i] * componentWeights.specialization +
                memberTrustEma * componentWeights.trust +
                memberPerfEma * componentWeights.historicalPerformance +
                this.#performanceScores[i] * componentWeights.performance
            ) * memberRobustness;
        }, 0);

        const normalizedScore = Math.min(1, Math.max(0, 0.8 * (compositeScore / Math.max(totalCompositeScore, 1e-8)) + 0.2 / this.#ensembleSize));
        const ntkStability = this.#computeNTKStability(this.#historicalPerformance[idx].slice(-10), lossVariance);
        const weightFactorMin = 0.5 / (1 + lossVariance);
        const weightFactorMax = 1 + this.#ensembleSize / (10 * (1 + lossVariance));
        const weightFactor = Math.min(weightFactorMax, Math.max(weightFactorMin, normalizedScore * this.#ensembleSize * ntkStability));

        const sigmoidSlope = 5 / (1 + trustVariance);
        const alpha = Math.min(1, Math.max(0.1, 0.5 + 0.5 * (1 / (1 + Math.exp(sigmoidSlope * trust_ema)) - 0.5)));

        const dynamicK = 4 / (1 + trustVariance);
        const decay = Math.max(0.1, Math.min(1, 1 / (1 + dynamicK * trustVariance)));

        const attentionMatrixNorms = [];
        const ffnMatrixNorms = [];
        const vectorNorms = [];
        const specMatrixNorms = [];

        ffnMatrixNorms.push(this.#computeSpectralNorm(this.#gradientAccumulation[idx].outputWeights));
        vectorNorms.push(this.#computeGradientNorm(this.#gradientAccumulation[idx].outputBias, false));

        for (let layer = 0; layer < this.#numLayers; layer++) {
            ['Wq', 'Wk', 'Wv', 'Wo'].forEach(key => {
                const gradMatrix = this.#gradientAccumulation[idx].attentionWeights[layer][key];
                attentionMatrixNorms.push(this.#computeSpectralNorm(gradMatrix));
            });

            ['gate_proj', 'up_proj'].forEach(key => {
                const gradMatrix = this.#gradientAccumulation[idx].ffnWeights[layer][key];
                ffnMatrixNorms.push(this.#computeSpectralNorm(gradMatrix));
            });
            const downGrad = this.#gradientAccumulation[idx].ffnWeights[layer].down_proj;
            ffnMatrixNorms.push(this.#computeSpectralNorm(downGrad));

            vectorNorms.push(this.#computeGradientNorm(this.#gradientAccumulation[idx].layerNormWeights[layer].gamma1, false));
            vectorNorms.push(this.#computeGradientNorm(this.#gradientAccumulation[idx].layerNormWeights[layer].gamma2, false));
        }

        vectorNorms.push(this.#computeGradientNorm(this.#gradientAccumulation[idx].attentionBias, false));
        vectorNorms.push(this.#computeGradientNorm(this.#gradientAccumulation[idx].attentionWeightMatrix, false));

        specMatrixNorms.push(this.#computeSpectralNorm(this.#gradientAccumulation[idx].specializationWeights));

        const sparseThreshold = this.#computeSparseThreshold([...attentionMatrixNorms, ...ffnMatrixNorms, ...vectorNorms, ...specMatrixNorms]);

        const attentionPercentile = this.#computeDynamicPercentile(attentionMatrixNorms, baseAttentionPercentile);
        const ffnPercentile = this.#computeDynamicPercentile(ffnMatrixNorms, baseFfnPercentile);
        const vectorPercentile = this.#computeDynamicPercentile(vectorNorms, baseVectorPercentile);
        const specPercentile = this.#computeDynamicPercentile(specMatrixNorms, baseAttentionPercentile);

        const attentionMatrixThreshold = Math.min(this.#computePercentile(attentionMatrixNorms, attentionPercentile), 1.0) * weightFactor;
        const ffnMatrixThreshold = Math.min(this.#computePercentile(ffnMatrixNorms, ffnPercentile), 1.0) * weightFactor;
        const vectorThreshold = Math.min(this.#computePercentile(vectorNorms, vectorPercentile), 1.0) * weightFactor;
        const specMatrixThreshold = Math.min(this.#computePercentile(specMatrixNorms, specPercentile), 1.0) * weightFactor;

        this.#scaleGradientMatrix(this.#gradientAccumulation[idx].outputWeights, ffnMatrixThreshold, minScaleFactor, alpha, decay, sparseThreshold);
        this.#scaleGradientVector(this.#gradientAccumulation[idx].outputBias, vectorThreshold, minScaleFactor, alpha, decay, sparseThreshold);

        for (let layer = 0; layer < this.#numLayers; layer++) {
            ['Wq', 'Wk', 'Wv', 'Wo'].forEach(key => {
                this.#scaleGradientMatrix(this.#gradientAccumulation[idx].attentionWeights[layer][key], attentionMatrixThreshold, minScaleFactor, alpha, decay, sparseThreshold);
            });

            ['gate_proj', 'up_proj'].forEach(key => {
                this.#scaleGradientMatrix(this.#gradientAccumulation[idx].ffnWeights[layer][key], ffnMatrixThreshold, minScaleFactor, alpha, decay, sparseThreshold);
            });
            this.#scaleGradientMatrix(this.#gradientAccumulation[idx].ffnWeights[layer].down_proj, ffnMatrixThreshold, minScaleFactor, alpha, decay, sparseThreshold);

            this.#scaleGradientVector(this.#gradientAccumulation[idx].layerNormWeights[layer].gamma1, vectorThreshold, minScaleFactor, alpha, decay, sparseThreshold);
            this.#scaleGradientVector(this.#gradientAccumulation[idx].layerNormWeights[layer].gamma2, vectorThreshold, minScaleFactor, alpha, decay, sparseThreshold);
        }

        this.#scaleGradientVector(this.#gradientAccumulation[idx].attentionBias, vectorThreshold, minScaleFactor, alpha, decay, sparseThreshold);
        this.#scaleGradientVector(this.#gradientAccumulation[idx].attentionWeightMatrix, vectorThreshold, minScaleFactor, alpha, decay, sparseThreshold);
        this.#scaleGradientMatrix(this.#gradientAccumulation[idx].specializationWeights, specMatrixThreshold, minScaleFactor, alpha, decay, sparseThreshold);
    }

    #accumulateGradients (inputs, outputs, target, probability, layerOutputs, activations, attentionIntermediates) {
        const dL_dLogit = probability - target;

        let finalLogit = 0;
        for (let i = 0; i < this.#ensembleSize; i++) {
            if (isValidNumber(outputs[i]) && isValidNumber(this.#ensembleWeights[i])) {
                finalLogit += this.#ensembleWeights[i] * outputs[i];
            }
        }

        this.#transformers.forEach((transformer, idx) => {
            const lr = this.#adaptiveLearningRate[idx];

            const memberWeight = isValidNumber(this.#ensembleWeights[idx]) ? this.#ensembleWeights[idx] : 1 / this.#ensembleSize;
            const effective_dL = dL_dLogit * memberWeight;

            let agreement_dL = 0;
            if (isValidNumber(outputs[idx])) {
                agreement_dL = dL_dLogit * memberWeight * (outputs[idx] - finalLogit);
            }

            let attWeightGrad = 0;
            if (isValidNumber(agreement_dL)) {
                attWeightGrad = agreement_dL / Math.sqrt(this.#hiddenSize);
            }
            for (let k = 0; k < this.#hiddenSize; k++) {
                if (isValidNumber(attWeightGrad)) {
                    this.#gradientAccumulation[idx].attentionBias[k] += attWeightGrad * lr;
                }
                for (let i = 0; i < this.#inputSize; i++) {
                    if (!isValidNumber(inputs[i])) continue;
                    const specFactor = isValidNumber(this.#specializationScores[idx])
                        ? 1 + this.#specializationScores[idx] * this.#swarmIntelligenceFactor
                        : 1;
                    const update = attWeightGrad * inputs[i] * specFactor * lr;
                    if (isValidNumber(update)) {
                        this.#gradientAccumulation[idx].attentionWeightMatrix[k] += update;
                    }
                }
            }

            for (let j = 0; j < this.#hiddenSize; j++) {
                for (let k = 0; k < this.#hiddenSize; k++) {
                    const inputIdx = (j + k) % this.#inputSize;
                    if (!isValidNumber(inputs[inputIdx])) continue;
                    const specFactor = isValidNumber(this.#specializationScores[idx])
                        ? 1 + this.#specializationScores[idx] * this.#swarmIntelligenceFactor
                        : 1;
                    const update = dL_dLogit * inputs[inputIdx] * specFactor * lr * 0.01;
                    if (isValidNumber(update)) {
                        this.#gradientAccumulation[idx].specializationWeights[j][k] += update;
                    }
                }
            }

            const finalX = layerOutputs[idx][this.#numLayers];
            let pooled = Array(this.#hiddenSize).fill(0);
            let validCount = 0;
            for (let pos = 0; pos < this.#inputSize; pos++) {
                let posHidden = finalX[pos];
                if (this.#numLayers > 0) {
                    const lastLayerIdx = this.#numLayers - 1;
                    posHidden = this.#rmsNorm(posHidden, transformer.layerNormWeights[lastLayerIdx].gamma2);
                }
                for (let j = 0; j < this.#hiddenSize; j++) {
                    if (isValidNumber(posHidden[j])) {
                        pooled[j] += posHidden[j];
                    }
                }
                validCount++;
            }
            if (validCount > 0) {
                pooled = pooled.map(v => v / validCount);
            }

            let gradPooled = Array(this.#hiddenSize).fill(0);
            for (let i = 0; i < this.#hiddenSize; i++) {
                const w = transformer.outputWeights[i][0];
                if (isValidNumber(pooled[i]) && isValidNumber(w)) {
                    this.#gradientAccumulation[idx].outputWeights[i][0] += effective_dL * pooled[i] * lr;
                    gradPooled[i] += effective_dL * w;
                }
            }
            this.#gradientAccumulation[idx].outputBias[0] += effective_dL * lr;

            let grad = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
            if (this.#numLayers > 0) {
                const lastLayer = this.#numLayers - 1;
                const gamma = transformer.layerNormWeights[lastLayer].gamma2;
                for (let pos = 0; pos < this.#inputSize; pos++) {
                    const x = finalX[pos];
                    const sq_sum = x.reduce((s, v) => s + v * v, 0);
                    const rms = Math.sqrt(sq_sum / this.#hiddenSize + 1e-6);
                    const norm_x = x.map(v => v / rms);

                    const incoming = gradPooled.map(g => g / validCount);

                    const d_gamma = incoming.map((g, i) => g * norm_x[i]);
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        this.#gradientAccumulation[idx].layerNormWeights[lastLayer].gamma2[j] += d_gamma[j] * lr;
                    }

                    const d_norm = incoming.map((g, i) => g * gamma[i]);
                    const dot = d_norm.reduce((s, dn, i) => s + dn * norm_x[i], 0);
                    const d_x = d_norm.map((dn, i) => dn / rms - norm_x[i] * dot / (this.#hiddenSize + 1e-8));
                    grad[pos] = d_x;
                }
            } else {
                for (let pos = 0; pos < this.#inputSize; pos++) {
                    grad[pos] = gradPooled.map(g => g / validCount);
                }
            }

            for (let layer = this.#numLayers - 1; layer >= 0; layer--) {
                const act = activations[idx][layer];
                const inter = attentionIntermediates[idx][layer];
                const headSize = this.#hiddenSize / this.#numHeads;

                let gradToAttentionResidual = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
                for (let pos = 0; pos < this.#inputSize; pos++) {
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        gradToAttentionResidual[pos][j] = grad[pos][j];
                    }
                }

                for (let pos = 0; pos < this.#inputSize; pos++) {
                    const x = act.normAttention[pos];
                    const gate = Array(this.#feedForwardSize).fill(0);
                    const up = Array(this.#feedForwardSize).fill(0);

                    for (let i = 0; i < this.#hiddenSize; i++) {
                        for (let j = 0; j < this.#feedForwardSize; j++) {
                            const specWeight = isValidNumber(this.#specializationWeights[idx][i % this.#hiddenSize][j % this.#hiddenSize])
                                ? Math.min(Math.max(1 + this.#specializationScores[idx] * this.#specializationWeights[idx][i % this.#hiddenSize][j % this.#hiddenSize], 0.5), 1.5)
                                : 1;
                            gate[j] += isValidNumber(x[i]) && isValidNumber(transformer.ffnWeights[layer].gate_proj[i][j])
                                ? x[i] * transformer.ffnWeights[layer].gate_proj[i][j] * specWeight
                                : 0;
                            up[j] += isValidNumber(x[i]) && isValidNumber(transformer.ffnWeights[layer].up_proj[i][j])
                                ? x[i] * transformer.ffnWeights[layer].up_proj[i][j] * specWeight
                                : 0;
                        }
                    }

                    const silu_gate = gate.map(this.#silu);
                    const activated = silu_gate.map((s, j) => s * up[j]);

                    const incoming = grad[pos];

                    for (let j = 0; j < this.#feedForwardSize; j++) {
                        for (let k = 0; k < this.#hiddenSize; k++) {
                            const specWeight = isValidNumber(this.#specializationWeights[idx][k % this.#hiddenSize][j % this.#hiddenSize])
                                ? Math.min(Math.max(1 + this.#specializationScores[idx] * this.#specializationWeights[idx][k % this.#hiddenSize][j % this.#hiddenSize], 0.5), 1.5)
                                : 1;
                            this.#gradientAccumulation[idx].ffnWeights[layer].down_proj[j][k] += incoming[k] * activated[j] * specWeight * lr;
                        }
                    }

                    const d_activated = Array(this.#feedForwardSize).fill(0);
                    for (let j = 0; j < this.#feedForwardSize; j++) {
                        for (let k = 0; k < this.#hiddenSize; k++) {
                            const specWeight = isValidNumber(this.#specializationWeights[idx][k % this.#hiddenSize][j % this.#hiddenSize])
                                ? Math.min(Math.max(1 + this.#specializationScores[idx] * this.#specializationWeights[idx][k % this.#hiddenSize][j % this.#hiddenSize], 0.5), 1.5)
                                : 1;
                            d_activated[j] += incoming[k] * transformer.ffnWeights[layer].down_proj[j][k] * specWeight;
                        }
                    }

                    const d_silu = d_activated.map((d, j) => d * up[j]);
                    const d_up = d_activated.map((d, j) => d * silu_gate[j]);
                    const d_gate = d_silu.map((d, j) => d * this.#siluDerivative(gate[j]));

                    let d_normAttention = Array(this.#hiddenSize).fill(0);
                    for (let i = 0; i < this.#hiddenSize; i++) {
                        for (let j = 0; j < this.#feedForwardSize; j++) {
                            const specWeight = isValidNumber(this.#specializationWeights[idx][i % this.#hiddenSize][j % this.#hiddenSize])
                                ? Math.min(Math.max(1 + this.#specializationScores[idx] * this.#specializationWeights[idx][i % this.#hiddenSize][j % this.#hiddenSize], 0.5), 1.5)
                                : 1;

                            this.#gradientAccumulation[idx].ffnWeights[layer].gate_proj[i][j] += d_gate[j] * x[i] * specWeight * lr;
                            this.#gradientAccumulation[idx].ffnWeights[layer].up_proj[i][j] += d_up[j] * x[i] * specWeight * lr;

                            d_normAttention[i] += d_gate[j] * transformer.ffnWeights[layer].gate_proj[i][j] * specWeight +
                                                  d_up[j] * transformer.ffnWeights[layer].up_proj[i][j] * specWeight;
                        }
                    }

                    const gamma2 = transformer.layerNormWeights[layer].gamma2;
                    const sq_sum = x.reduce((s, v) => s + v * v, 0);
                    const rms = Math.sqrt(sq_sum / this.#hiddenSize + 1e-6);
                    const norm_x = x.map(v => v / rms);

                    const d_gamma = d_normAttention.map((g, i) => g * norm_x[i]);
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        this.#gradientAccumulation[idx].layerNormWeights[layer].gamma2[j] += d_gamma[j] * lr;
                    }

                    const d_norm = d_normAttention.map((g, i) => g * gamma2[i]);
                    const dot = d_norm.reduce((s, dn, i) => s + dn * norm_x[i], 0);
                    const d_x_branch = d_norm.map((dn, i) => dn / rms - norm_x[i] * dot / (this.#hiddenSize + 1e-8));

                    for (let j = 0; j < this.#hiddenSize; j++) {
                        gradToAttentionResidual[pos][j] += d_x_branch[j];
                    }
                }

                const gradToAttentionOutput = gradToAttentionResidual;

                const d_v = Array(this.#numHeads).fill().map(() => Array(this.#inputSize).fill().map(() => Array(headSize).fill(0)));
                const d_scores = Array(this.#numHeads).fill().map(() => Array(this.#inputSize).fill().map(() => Array(this.#inputSize).fill(0)));
                const d_q = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
                const d_k = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));

                for (let queryPos = 0; queryPos < this.#inputSize; queryPos++) {
                    const incoming = gradToAttentionOutput[queryPos];

                    for (let outDim = 0; outDim < this.#hiddenSize; outDim++) {
                        for (let preDim = 0; preDim < this.#hiddenSize; preDim++) {
                            const specWeight = isValidNumber(this.#specializationWeights[idx][preDim % this.#hiddenSize][outDim])
                                ? Math.min(Math.max(1 + this.#specializationScores[idx] * this.#specializationWeights[idx][preDim % this.#hiddenSize][outDim], 0.5), 1.5)
                                : 1;
                            this.#gradientAccumulation[idx].attentionWeights[layer].Wo[preDim][outDim] += incoming[outDim] * inter.preWoOutput[queryPos][preDim] * specWeight * lr;
                        }
                    }

                    const d_preWo = Array(this.#hiddenSize).fill(0);
                    for (let outDim = 0; outDim < this.#hiddenSize; outDim++) {
                        for (let preDim = 0; preDim < this.#hiddenSize; preDim++) {
                            const specWeight = isValidNumber(this.#specializationWeights[idx][preDim % this.#hiddenSize][outDim])
                                ? Math.min(Math.max(1 + this.#specializationScores[idx] * this.#specializationWeights[idx][preDim % this.#hiddenSize][outDim], 0.5), 1.5)
                                : 1;
                            d_preWo[preDim] += incoming[outDim] * transformer.attentionWeights[layer].Wo[preDim][outDim] * specWeight;
                        }
                    }

                    for (let h = 0; h < this.#numHeads; h++) {
                        const offset = h * headSize;
                        for (let dim = 0; dim < headSize; dim++) {
                            const idx = offset + dim;
                            let d_att = isValidNumber(d_preWo[idx]) ? d_preWo[idx] : 0;

                            for (let valuePos = 0; valuePos < this.#inputSize; valuePos++) {
                                let prob = isValidNumber(inter.attentionProbs[h][queryPos][valuePos]) ? inter.attentionProbs[h][queryPos][valuePos] : 0;
                                let vVal = isValidNumber(inter.V[valuePos][idx]) ? inter.V[valuePos][idx] : 0;

                                d_scores[h][queryPos][valuePos] += d_att * vVal;
                                d_v[h][valuePos][dim] += d_att * prob;
                            }
                        }
                    }
                }

                for (let h = 0; h < this.#numHeads; h++) {
                    for (let qp = 0; qp < this.#inputSize; qp++) {
                        let weighted = 0;
                        for (let vp = 0; vp < this.#inputSize; vp++) {
                            const p = isValidNumber(inter.attentionProbs[h][qp][vp]) ? inter.attentionProbs[h][qp][vp] : 0;
                            const dp = isValidNumber(d_scores[h][qp][vp]) ? d_scores[h][qp][vp] : 0;
                            weighted += p * dp;
                        }
                        for (let vp = 0; vp < this.#inputSize; vp++) {
                            const p = isValidNumber(inter.attentionProbs[h][qp][vp]) ? inter.attentionProbs[h][qp][vp] : 0;
                            const dp = isValidNumber(d_scores[h][qp][vp]) ? d_scores[h][qp][vp] : 0;
                            d_scores[h][qp][vp] = p * (dp - weighted);
                        }
                    }
                }

                for (let h = 0; h < this.#numHeads; h++) {
                    for (let queryPos = 0; queryPos < this.#inputSize; queryPos++) {
                        for (let keyPos = 0; keyPos < this.#inputSize; keyPos++) {
                            let scaled = isValidNumber(d_scores[h][queryPos][keyPos]) ? d_scores[h][queryPos][keyPos] / Math.sqrt(headSize) : 0;
                            for (let dim = 0; dim < headSize; dim++) {
                                const idx = h * headSize + dim;
                                d_q[queryPos][idx] += scaled * (isValidNumber(inter.K[keyPos][idx]) ? inter.K[keyPos][idx] : 0);
                                d_k[keyPos][idx] += scaled * (isValidNumber(inter.Q[queryPos][idx]) ? inter.Q[queryPos][idx] : 0);
                            }
                        }
                    }
                }

                let gradToNormX = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));

                for (let pos = 0; pos < this.#inputSize; pos++) {
                    for (let inputDim = 0; inputDim < this.#hiddenSize; inputDim++) {
                        let contrib = 0;
                        for (let outputDim = 0; outputDim < this.#hiddenSize; outputDim++) {
                            const specWeight = isValidNumber(this.#specializationWeights[idx][inputDim % this.#hiddenSize][outputDim])
                                ? Math.min(Math.max(1 + this.#specializationScores[idx] * this.#specializationWeights[idx][inputDim % this.#hiddenSize][outputDim], 0.5), 1.5)
                                : 1;
                            contrib += d_q[pos][outputDim] * transformer.attentionWeights[layer].Wq[inputDim][outputDim] * specWeight;
                        }
                        gradToNormX[pos][inputDim] += contrib;
                    }
                }

                for (let pos = 0; pos < this.#inputSize; pos++) {
                    for (let inputDim = 0; inputDim < this.#hiddenSize; inputDim++) {
                        let contrib = 0;
                        for (let outputDim = 0; outputDim < this.#hiddenSize; outputDim++) {
                            const specWeight = isValidNumber(this.#specializationWeights[idx][inputDim % this.#hiddenSize][outputDim])
                                ? Math.min(Math.max(1 + this.#specializationScores[idx] * this.#specializationWeights[idx][inputDim % this.#hiddenSize][outputDim], 0.5), 1.5)
                                : 1;
                            contrib += d_k[pos][outputDim] * transformer.attentionWeights[layer].Wk[inputDim][outputDim] * specWeight;
                        }
                        gradToNormX[pos][inputDim] += contrib;
                    }
                }

                for (let pos = 0; pos < this.#inputSize; pos++) {
                    for (let inputDim = 0; inputDim < this.#hiddenSize; inputDim++) {
                        let contrib = 0;
                        for (let h = 0; h < this.#numHeads; h++) {
                            for (let dim = 0; dim < headSize; dim++) {
                                const outputDim = h * headSize + dim;
                                const specWeight = isValidNumber(this.#specializationWeights[idx][inputDim % this.#hiddenSize][outputDim])
                                    ? Math.min(Math.max(1 + this.#specializationScores[idx] * this.#specializationWeights[idx][inputDim % this.#hiddenSize][outputDim], 0.5), 1.5)
                                    : 1;
                                contrib += d_v[h][pos][dim] * transformer.attentionWeights[layer].Wv[inputDim][outputDim] * specWeight;
                            }
                        }
                        gradToNormX[pos][inputDim] += contrib;
                    }
                }

                for (let inputDim = 0; inputDim < this.#hiddenSize; inputDim++) {
                    for (let outputDim = 0; outputDim < this.#hiddenSize; outputDim++) {
                        let wqUpdate = 0;
                        let wkUpdate = 0;
                        let wvUpdate = 0;

                        const specWeight = isValidNumber(this.#specializationWeights[idx][inputDim % this.#hiddenSize][outputDim])
                            ? Math.min(Math.max(1 + this.#specializationScores[idx] * this.#specializationWeights[idx][inputDim % this.#hiddenSize][outputDim], 0.5), 1.5)
                            : 1;

                        for (let pos = 0; pos < this.#inputSize; pos++) {
                            wqUpdate += d_q[pos][outputDim] * act.normX[pos][inputDim];
                            wkUpdate += d_k[pos][outputDim] * act.normX[pos][inputDim];

                            const head = Math.floor(outputDim / headSize);
                            const dim = outputDim % headSize;
                            wvUpdate += d_v[head][pos][dim] * act.normX[pos][inputDim];
                        }

                        this.#gradientAccumulation[idx].attentionWeights[layer].Wq[inputDim][outputDim] += wqUpdate * specWeight * lr;
                        this.#gradientAccumulation[idx].attentionWeights[layer].Wk[inputDim][outputDim] += wkUpdate * specWeight * lr;
                        this.#gradientAccumulation[idx].attentionWeights[layer].Wv[inputDim][outputDim] += wvUpdate * specWeight * lr;
                    }
                }

                const gamma1 = transformer.layerNormWeights[layer].gamma1;
                for (let pos = 0; pos < this.#inputSize; pos++) {
                    const x = act.normX[pos];
                    const sq_sum = x.reduce((s, v) => s + v * v, 0);
                    const rms = Math.sqrt(sq_sum / this.#hiddenSize + 1e-6);
                    const norm_x = x.map(v => v / rms);

                    const incoming = gradToNormX[pos];

                    const d_gamma = incoming.map((g, i) => g * norm_x[i]);
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        this.#gradientAccumulation[idx].layerNormWeights[layer].gamma1[j] += d_gamma[j] * lr;
                    }

                    const d_norm = incoming.map((g, i) => g * gamma1[i]);
                    const dot = d_norm.reduce((s, dn, i) => s + dn * norm_x[i], 0);
                    const d_x = d_norm.map((dn, i) => dn / rms - norm_x[i] * dot / (this.#hiddenSize + 1e-8));

                    for (let j = 0; j < this.#hiddenSize; j++) {
                        grad[pos][j] = gradToAttentionOutput[pos][j] + d_x[j];
                    }
                }
            }
        });
    }

    #applyGradients (shouldScale = true, shouldClone = false, steps = 1) {
        let gradClone;
        if (shouldClone) gradClone = this.#setGradientStructure();

        this.#transformers.forEach((transformer, idx) => {
            if (shouldScale) this.#scaleGradients(idx);

            for (let k = 0; k < this.#hiddenSize; k++) {
                const biasAcc = this.#gradientAccumulation[idx].attentionBias[k];
                if (isValidNumber(biasAcc)) {
                    const finalBiasAcc = biasAcc / steps;
                    this.#attentionBias[idx][k] -= finalBiasAcc;
                    if (shouldClone) gradClone[idx].attentionBias[k] = finalBiasAcc;
                }

                const matrixAcc = this.#gradientAccumulation[idx].attentionWeightMatrix[k];
                if (isValidNumber(matrixAcc)) {
                    const finalMatrixAcc = matrixAcc / steps;
                    this.#attentionWeightMatrix[idx][k] -= finalMatrixAcc;
                    if (shouldClone) gradClone[idx].attentionWeightMatrix[k] = finalMatrixAcc;
                }
            }

            for (let j = 0; j < this.#hiddenSize; j++) {
                for (let k = 0; k < this.#hiddenSize; k++) {
                    const specAcc = this.#gradientAccumulation[idx].specializationWeights[j][k];
                    if (isValidNumber(specAcc)) {
                        const finalSpecAcc = specAcc / steps;
                        this.#specializationWeights[idx][j][k] -= finalSpecAcc;
                        if (shouldClone) gradClone[idx].specializationWeights[j][k] = finalSpecAcc;
                    }
                }
            }

            for (let i = 0; i < this.#hiddenSize; i++) {
                const outWeightAcc = this.#gradientAccumulation[idx].outputWeights[i][0];
                if (isValidNumber(outWeightAcc)) {
                    const finalOutWeightAcc = outWeightAcc / steps;
                    transformer.outputWeights[i][0] -= finalOutWeightAcc;
                    if (shouldClone) gradClone[idx].outputWeights[i][0] = finalOutWeightAcc;
                }
            }
            const outBiasAcc = this.#gradientAccumulation[idx].outputBias[0];
            if (isValidNumber(outBiasAcc)) {
                const finalOutBiasAcc = outBiasAcc / steps;
                transformer.outputBias[0] -= finalOutBiasAcc;
                if (shouldClone) gradClone[idx].outputBias[0] = finalOutBiasAcc;
            }

            for (let layer = 0; layer < this.#numLayers; layer++) {
                for (let i = 0; i < this.#hiddenSize; i++) {
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        ['Wq', 'Wk', 'Wv', 'Wo'].forEach(key => {
                            const acc = this.#gradientAccumulation[idx].attentionWeights[layer][key][i][j];
                            if (isValidNumber(acc)) {
                                const finalAcc = acc / steps;
                                transformer.attentionWeights[layer][key][i][j] -= finalAcc;
                                if (shouldClone) gradClone[idx].attentionWeights[layer][key][i][j] = finalAcc;
                            }
                        });
                    }
                }

                for (let i = 0; i < this.#hiddenSize; i++) {
                    for (let j = 0; j < this.#feedForwardSize; j++) {
                        const gateAcc = this.#gradientAccumulation[idx].ffnWeights[layer].gate_proj[i][j];
                        if (isValidNumber(gateAcc)) {
                            const finalAcc = gateAcc / steps;
                            transformer.ffnWeights[layer].gate_proj[i][j] -= finalAcc;
                            if (shouldClone) gradClone[idx].ffnWeights[layer].gate_proj[i][j] = finalAcc;
                        }
                        const upAcc = this.#gradientAccumulation[idx].ffnWeights[layer].up_proj[i][j];
                        if (isValidNumber(upAcc)) {
                            const finalAcc = upAcc / steps;
                            transformer.ffnWeights[layer].up_proj[i][j] -= finalAcc;
                            if (shouldClone) gradClone[idx].ffnWeights[layer].up_proj[i][j] = finalAcc;
                        }
                    }
                }
                for (let i = 0; i < this.#feedForwardSize; i++) {
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        const downAcc = this.#gradientAccumulation[idx].ffnWeights[layer].down_proj[i][j];
                        if (isValidNumber(downAcc)) {
                            const finalAcc = downAcc / steps;
                            transformer.ffnWeights[layer].down_proj[i][j] -= finalAcc;
                            if (shouldClone) gradClone[idx].ffnWeights[layer].down_proj[i][j] = finalAcc;
                        }
                    }
                }

                for (let i = 0; i < this.#hiddenSize; i++) {
                    const gamma1Acc = this.#gradientAccumulation[idx].layerNormWeights[layer].gamma1[i];
                    if (isValidNumber(gamma1Acc)) {
                        const finalAcc = gamma1Acc / steps;
                        transformer.layerNormWeights[layer].gamma1[i] -= finalAcc;
                        if (shouldClone) gradClone[idx].layerNormWeights[layer].gamma1[i] = finalAcc;
                    }
                    const gamma2Acc = this.#gradientAccumulation[idx].layerNormWeights[layer].gamma2[i];
                    if (isValidNumber(gamma2Acc)) {
                        const finalAcc = gamma2Acc / steps;
                        transformer.layerNormWeights[layer].gamma2[i] -= finalAcc;
                        if (shouldClone) gradClone[idx].layerNormWeights[layer].gamma2[i] = finalAcc;
                    }
                }
            }
        });

        if (shouldClone) return gradClone;
    }

    #rollbackGradients (gradClone) {
        this.#transformers.forEach((transformer, idx) => {
            for (let k = 0; k < this.#hiddenSize; k++) {
                const attBias = gradClone[idx].attentionBias[k];
                if (isValidNumber(attBias)) this.#attentionBias[idx][k] += attBias;

                const attMatrix = gradClone[idx].attentionWeightMatrix[k];
                if (isValidNumber(attMatrix)) this.#attentionWeightMatrix[idx][k] += attMatrix;
            }

            for (let j = 0; j < this.#hiddenSize; j++) {
                for (let k = 0; k < this.#hiddenSize; k++) {
                    const specWeight = gradClone[idx].specializationWeights[j][k];
                    if (isValidNumber(specWeight)) this.#specializationWeights[idx][j][k] += specWeight;
                }
            }

            for (let i = 0; i < this.#hiddenSize; i++) {
                const outputWeight = gradClone[idx].outputWeights[i][0];
                if (isValidNumber(outputWeight)) transformer.outputWeights[i][0] += outputWeight;
            }
            const outputBias = gradClone[idx].outputBias[0];
            if (isValidNumber(outputBias)) transformer.outputBias[0] += outputBias;

            for (let layer = 0; layer < this.#numLayers; layer++) {
                for (let i = 0; i < this.#hiddenSize; i++) {
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        ['Wq', 'Wk', 'Wv', 'Wo'].forEach(key => {
                            const weight = gradClone[idx].attentionWeights[layer][key][i][j];
                            if (isValidNumber(weight)) transformer.attentionWeights[layer][key][i][j] += weight;
                        });
                    }
                }

                for (let i = 0; i < this.#hiddenSize; i++) {
                    for (let j = 0; j < this.#feedForwardSize; j++) {
                        const gateWeight = gradClone[idx].ffnWeights[layer].gate_proj[i][j];
                        if (isValidNumber(gateWeight)) transformer.ffnWeights[layer].gate_proj[i][j] += gateWeight;
                        const upWeight = gradClone[idx].ffnWeights[layer].up_proj[i][j];
                        if (isValidNumber(upWeight)) transformer.ffnWeights[layer].up_proj[i][j] += upWeight;
                    }
                }
                for (let i = 0; i < this.#feedForwardSize; i++) {
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        const downWeight = gradClone[idx].ffnWeights[layer].down_proj[i][j];
                        if (isValidNumber(downWeight)) transformer.ffnWeights[layer].down_proj[i][j] += downWeight;
                    }
                }

                for (let i = 0; i < this.#hiddenSize; i++) {
                    const gamma1Weight = gradClone[idx].layerNormWeights[layer].gamma1[i];
                    if (isValidNumber(gamma1Weight)) transformer.layerNormWeights[layer].gamma1[i] += gamma1Weight;
                    const gamma2Weight = gradClone[idx].layerNormWeights[layer].gamma2[i];
                    if (isValidNumber(gamma2Weight)) transformer.layerNormWeights[layer].gamma2[i] += gamma2Weight;
                }
            }
        });
    }

    #distillKnowledge (outputs, attentionIntermediatesAll, activationsAll, layerOutputsAll) {
        if (
            !Array.isArray(outputs) ||
            outputs.length !== this.#ensembleSize ||
            !outputs.every(isValidNumber) ||
            attentionIntermediatesAll.length !== this.#ensembleSize ||
            activationsAll.length !== this.#ensembleSize ||
            layerOutputsAll.length !== this.#ensembleSize
        ) {
            return;
        }

        const sortedIndices = this.#performanceScores
            .map((score, idx) => ({ score: isValidNumber(score) ? score : 0, idx }))
            .sort((a, b) => b.score - a.score);

        const topTierCount = Math.max(1, Math.floor(this.#ensembleSize * 0.30));
        const middleTierEnd = topTierCount + Math.floor(this.#ensembleSize * 0.40);
        const topPerformers = sortedIndices.slice(0, topTierCount).map(({ idx }) => idx);

        const topOutputs = topPerformers.map(idx => outputs[idx]);
        const topWeights = topPerformers.map(idx => this.#ensembleWeights[idx]);
        const weightSum = topWeights.reduce((sum, w) => sum + (isValidNumber(w) ? w : 0), 0) || 1;
        const teacherLogit = topOutputs.reduce((sum, output, i) =>
            sum + output * (topWeights[i] / weightSum), 0
        );

        const varianceTop = topOutputs.reduce((sum, out) => sum + Math.pow(out - teacherLogit, 2), 0) / topOutputs.length;
        const stdTopOutputs = Math.sqrt(varianceTop + 1e-8);

        const temperature = 2.0;
        const momentum = 0.9;
        const outputKdScale = 0.2;

        this.#transformers.forEach((transformer, idx) => {
            if (topPerformers.includes(idx)) return;

            const rank = sortedIndices.findIndex(entry => entry.idx === idx);
            const isMiddleTier = rank >= topTierCount && rank < middleTierEnd;
            const isBottomTier = rank >= middleTierEnd;

            const kdStrengthMultiplier = isMiddleTier ? 1.0 : 0.7;

            const diversityScore = Math.min(1, Math.abs(outputs[idx] - teacherLogit) / (stdTopOutputs || 1));
            const performanceGap = this.#performanceScores[topPerformers[0]] - this.#performanceScores[idx];

            let klWeight = Math.min(1.0, Math.max(0.1, performanceGap / (this.#performanceScores[topPerformers[0]] || 1)));
            klWeight *= (1 - diversityScore);
            klWeight *= kdStrengthMultiplier;
            klWeight = Math.min(1.0, klWeight);

            const studentLogit = outputs[idx];
            const studentProb = this.#sigmoid(studentLogit / temperature);
            const teacherProb = this.#sigmoid(teacherLogit / temperature);

            let baseGrad = (studentProb - teacherProb) / temperature;
            baseGrad *= klWeight;
            if (!isValidNumber(baseGrad)) baseGrad = 0;

            const adjustedLearningRate = this.#adaptiveLearningRate[idx];

            const finalX = layerOutputsAll[idx][this.#numLayers];
            let pooled = Array(this.#hiddenSize).fill(0);
            let validCount = 0;
            for (let pos = 0; pos < this.#inputSize; pos++) {
                let posHidden = finalX[pos];
                if (this.#numLayers > 0) {
                    const lastLayerIdx = this.#numLayers - 1;
                    posHidden = this.#rmsNorm(posHidden, transformer.layerNormWeights[lastLayerIdx].gamma2);
                }
                for (let j = 0; j < this.#hiddenSize; j++) {
                    if (isValidNumber(posHidden[j])) {
                        pooled[j] += posHidden[j];
                    }
                }
                validCount++;
            }
            if (validCount > 0) {
                pooled = pooled.map(v => v / validCount);
            }

            const kd_dL = baseGrad * outputKdScale;
            for (let i = 0; i < this.#hiddenSize; i++) {
                if (isValidNumber(pooled[i])) {
                    const update = kd_dL * pooled[i] * adjustedLearningRate;
                    this.#gradientAccumulation[idx].outputWeights[i][0] += (1 - momentum) * update;
                }
            }
            this.#gradientAccumulation[idx].outputBias[0] += (1 - momentum) * kd_dL * adjustedLearningRate;

            if (!isBottomTier) return;

            let gradPooled = Array(this.#hiddenSize).fill(0);
            for (let i = 0; i < this.#hiddenSize; i++) {
                if (isValidNumber(transformer.outputWeights[i][0])) {
                    gradPooled[i] = kd_dL * transformer.outputWeights[i][0];
                }
            }

            let grad = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
            if (this.#numLayers > 0) {
                const lastLayerIdx = this.#numLayers - 1;
                const gamma = transformer.layerNormWeights[lastLayerIdx].gamma2;
                for (let pos = 0; pos < this.#inputSize; pos++) {
                    const x = finalX[pos];
                    const sq_sum = x.reduce((s, v) => s + v * v, 0);
                    const rms = Math.sqrt(sq_sum / this.#hiddenSize + 1e-6);
                    const norm_x = x.map(v => v / rms);

                    const incoming = gradPooled.map(g => g / validCount);

                    const d_gamma = incoming.map((g, i) => g * norm_x[i]);
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        if (isValidNumber(d_gamma[j])) {
                            const update = d_gamma[j] * adjustedLearningRate * 0.5;
                            this.#gradientAccumulation[idx].layerNormWeights[lastLayerIdx].gamma2[j] += (1 - momentum) * update;
                        }
                    }

                    const d_norm = incoming.map((g, i) => g * gamma[i]);
                    const dot = d_norm.reduce((s, dn, i) => s + dn * norm_x[i], 0);
                    const d_x = d_norm.map((dn, i) => dn / rms - norm_x[i] * dot / (this.#hiddenSize + 1e-8));
                    grad[pos] = d_x;
                }
            } else {
                for (let pos = 0; pos < this.#inputSize; pos++) {
                    grad[pos] = gradPooled.map(g => g / validCount);
                }
            }

            for (let layer = this.#numLayers - 1; layer >= 0; layer--) {
                const act = activationsAll[idx][layer];
                const inter = attentionIntermediatesAll[idx][layer];
                const headSize = this.#hiddenSize / this.#numHeads;

                let gradToAttentionResidual = grad.map(row => row.slice());

                for (let pos = 0; pos < this.#inputSize; pos++) {
                    const x = act.normAttention[pos];

                    const gate = Array(this.#feedForwardSize).fill(0);
                    const up = Array(this.#feedForwardSize).fill(0);
                    for (let i = 0; i < this.#hiddenSize; i++) {
                        for (let j = 0; j < this.#feedForwardSize; j++) {
                            const specWeight = isValidNumber(this.#specializationWeights[idx][i % this.#hiddenSize][j % this.#hiddenSize])
                                ? Math.min(Math.max(1 + this.#specializationScores[idx] * this.#specializationWeights[idx][i % this.#hiddenSize][j % this.#hiddenSize], 0.5), 1.5)
                                : 1;
                            gate[j] += isValidNumber(x[i]) && isValidNumber(transformer.ffnWeights[layer].gate_proj[i][j])
                                ? x[i] * transformer.ffnWeights[layer].gate_proj[i][j] * specWeight
                                : 0;
                            up[j] += isValidNumber(x[i]) && isValidNumber(transformer.ffnWeights[layer].up_proj[i][j])
                                ? x[i] * transformer.ffnWeights[layer].up_proj[i][j] * specWeight
                                : 0;
                        }
                    }
                    const silu_gate = gate.map(this.#silu);
                    const activated = silu_gate.map((s, j) => s * up[j]);

                    const incoming = grad[pos];

                    for (let j = 0; j < this.#feedForwardSize; j++) {
                        for (let k = 0; k < this.#hiddenSize; k++) {
                            const specWeight = isValidNumber(this.#specializationWeights[idx][k % this.#hiddenSize][j % this.#hiddenSize])
                                ? Math.min(Math.max(1 + this.#specializationScores[idx] * this.#specializationWeights[idx][k % this.#hiddenSize][j % this.#hiddenSize], 0.5), 1.5)
                                : 1;
                            const update = incoming[k] * activated[j] * specWeight * adjustedLearningRate;
                            if (isValidNumber(update)) {
                                this.#gradientAccumulation[idx].ffnWeights[layer].down_proj[j][k] += (1 - momentum) * update;
                            }
                        }
                    }

                    const d_activated = Array(this.#feedForwardSize).fill(0);
                    for (let j = 0; j < this.#feedForwardSize; j++) {
                        for (let k = 0; k < this.#hiddenSize; k++) {
                            const specWeight = isValidNumber(this.#specializationWeights[idx][k % this.#hiddenSize][j % this.#hiddenSize])
                                ? Math.min(Math.max(1 + this.#specializationScores[idx] * this.#specializationWeights[idx][k % this.#hiddenSize][j % this.#hiddenSize], 0.5), 1.5)
                                : 1;
                            d_activated[j] += incoming[k] * transformer.ffnWeights[layer].down_proj[j][k] * specWeight;
                        }
                    }
                    const d_silu = d_activated.map((d, j) => d * up[j]);
                    const d_up = d_activated.map((d, j) => d * silu_gate[j]);
                    const d_gate = d_silu.map((d, j) => d * this.#siluDerivative(gate[j]));

                    for (let i = 0; i < this.#hiddenSize; i++) {
                        for (let j = 0; j < this.#feedForwardSize; j++) {
                            const specWeight = isValidNumber(this.#specializationWeights[idx][i % this.#hiddenSize][j % this.#hiddenSize])
                                ? Math.min(Math.max(1 + this.#specializationScores[idx] * this.#specializationWeights[idx][i % this.#hiddenSize][j % this.#hiddenSize], 0.5), 1.5)
                                : 1;
                            const gateUpdate = d_gate[j] * x[i] * specWeight * adjustedLearningRate;
                            if (isValidNumber(gateUpdate)) {
                                this.#gradientAccumulation[idx].ffnWeights[layer].gate_proj[i][j] += (1 - momentum) * gateUpdate;
                            }
                            const upUpdate = d_up[j] * x[i] * specWeight * adjustedLearningRate;
                            if (isValidNumber(upUpdate)) {
                                this.#gradientAccumulation[idx].ffnWeights[layer].up_proj[i][j] += (1 - momentum) * upUpdate;
                            }
                        }
                    }

                    let d_normAttention = Array(this.#hiddenSize).fill(0);
                    for (let i = 0; i < this.#hiddenSize; i++) {
                        for (let j = 0; j < this.#feedForwardSize; j++) {
                            const specWeight = isValidNumber(this.#specializationWeights[idx][i % this.#hiddenSize][j % this.#hiddenSize])
                                ? Math.min(Math.max(1 + this.#specializationScores[idx] * this.#specializationWeights[idx][i % this.#hiddenSize][j % this.#hiddenSize], 0.5), 1.5)
                                : 1;
                            d_normAttention[i] += d_gate[j] * transformer.ffnWeights[layer].gate_proj[i][j] * specWeight +
                                                  d_up[j] * transformer.ffnWeights[layer].up_proj[i][j] * specWeight;
                        }
                    }

                    const gamma2 = transformer.layerNormWeights[layer].gamma2;
                    const sq_sum2 = x.reduce((s, v) => s + v * v, 0);
                    const rms2 = Math.sqrt(sq_sum2 / this.#hiddenSize + 1e-6);
                    const norm_x2 = x.map(v => v / rms2);
                    const d_gamma2 = d_normAttention.map((g, i) => g * norm_x2[i]);
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        if (isValidNumber(d_gamma2[j])) {
                            const update = d_gamma2[j] * adjustedLearningRate * 0.5;
                            this.#gradientAccumulation[idx].layerNormWeights[layer].gamma2[j] += (1 - momentum) * update;
                        }
                    }
                    const d_norm2 = d_normAttention.map((g, i) => g * gamma2[i]);
                    const dot2 = d_norm2.reduce((s, dn, i) => s + dn * norm_x2[i], 0);
                    const d_branch = d_norm2.map((dn, i) => dn / rms2 - norm_x2[i] * dot2 / (this.#hiddenSize + 1e-8));

                    for (let j = 0; j < this.#hiddenSize; j++) {
                        gradToAttentionResidual[pos][j] += d_branch[j];
                    }
                }

                const gradAttentionOutput = gradToAttentionResidual;

                for (let pos = 0; pos < this.#inputSize; pos++) {
                    for (let outDim = 0; outDim < this.#hiddenSize; outDim++) {
                        for (let preDim = 0; preDim < this.#hiddenSize; preDim++) {
                            const specWeight = isValidNumber(this.#specializationWeights[idx][preDim % this.#hiddenSize][outDim])
                                ? Math.min(Math.max(1 + this.#specializationScores[idx] * this.#specializationWeights[idx][preDim % this.#hiddenSize][outDim], 0.5), 1.5)
                                : 1;
                            const update = gradAttentionOutput[pos][outDim] * inter.preWoOutput[pos][preDim] * specWeight * adjustedLearningRate;
                            if (isValidNumber(update)) {
                                this.#gradientAccumulation[idx].attentionWeights[layer].Wo[preDim][outDim] += (1 - momentum) * update;
                            }
                        }
                    }
                }

                let d_preWo = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
                for (let pos = 0; pos < this.#inputSize; pos++) {
                    for (let outDim = 0; outDim < this.#hiddenSize; outDim++) {
                        for (let preDim = 0; preDim < this.#hiddenSize; preDim++) {
                            const specWeight = isValidNumber(this.#specializationWeights[idx][preDim % this.#hiddenSize][outDim])
                                ? Math.min(Math.max(1 + this.#specializationScores[idx] * this.#specializationWeights[idx][preDim % this.#hiddenSize][outDim], 0.5), 1.5)
                                : 1;
                            d_preWo[pos][preDim] += gradAttentionOutput[pos][outDim] * transformer.attentionWeights[layer].Wo[preDim][outDim] * specWeight;
                        }
                    }
                }

                const d_v = Array(this.#numHeads).fill().map(() => Array(this.#inputSize).fill().map(() => Array(headSize).fill(0)));
                const d_scores = Array(this.#numHeads).fill().map(() => Array(this.#inputSize).fill().map(() => Array(this.#inputSize).fill(0)));
                for (let pos = 0; pos < this.#inputSize; pos++) {
                    for (let h = 0; h < this.#numHeads; h++) {
                        const offset = h * headSize;
                        for (let dim = 0; dim < headSize; dim++) {
                            const fullDim = offset + dim;
                            const d_att = d_preWo[pos][fullDim];
                            if (!isValidNumber(d_att)) continue;
                            for (let vPos = 0; vPos < this.#inputSize; vPos++) {
                                d_scores[h][pos][vPos] += d_att * inter.V[vPos][fullDim];
                                d_v[h][vPos][dim] += d_att * inter.attentionProbs[h][pos][vPos];
                            }
                        }
                    }
                }

                for (let h = 0; h < this.#numHeads; h++) {
                    for (let pos = 0; pos < this.#inputSize; pos++) {
                        let weighted = 0;
                        for (let vPos = 0; vPos < this.#inputSize; vPos++) {
                            weighted += inter.attentionProbs[h][pos][vPos] * d_scores[h][pos][vPos];
                        }
                        for (let vPos = 0; vPos < this.#inputSize; vPos++) {
                            d_scores[h][pos][vPos] = inter.attentionProbs[h][pos][vPos] * (d_scores[h][pos][vPos] - weighted);
                        }
                    }
                }

                const d_q = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
                const d_k = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
                for (let h = 0; h < this.#numHeads; h++) {
                    for (let qPos = 0; qPos < this.#inputSize; qPos++) {
                        for (let kPos = 0; kPos < this.#inputSize; kPos++) {
                            const scaled = d_scores[h][qPos][kPos] / Math.sqrt(headSize);
                            if (!isValidNumber(scaled)) continue;
                            for (let dim = 0; dim < headSize; dim++) {
                                const fullDim = h * headSize + dim;
                                d_q[qPos][fullDim] += scaled * inter.K[kPos][fullDim];
                                d_k[kPos][fullDim] += scaled * inter.Q[qPos][fullDim];
                            }
                        }
                    }
                }

                let gradNormX = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
                for (let pos = 0; pos < this.#inputSize; pos++) {
                    for (let inDim = 0; inDim < this.#hiddenSize; inDim++) {
                        let contribQ = 0, contribK = 0, contribV = 0;
                        for (let outDim = 0; outDim < this.#hiddenSize; outDim++) {
                            const specWeight = isValidNumber(this.#specializationWeights[idx][inDim % this.#hiddenSize][outDim])
                                ? Math.min(Math.max(1 + this.#specializationScores[idx] * this.#specializationWeights[idx][inDim % this.#hiddenSize][outDim], 0.5), 1.5)
                                : 1;
                            contribQ += d_q[pos][outDim] * transformer.attentionWeights[layer].Wq[inDim][outDim] * specWeight;
                            contribK += d_k[pos][outDim] * transformer.attentionWeights[layer].Wk[inDim][outDim] * specWeight;
                            contribV += d_v[Math.floor(outDim / headSize)][pos][outDim % headSize] * transformer.attentionWeights[layer].Wv[inDim][outDim] * specWeight;
                        }
                        gradNormX[pos][inDim] = contribQ + contribK + contribV;
                    }
                }

                for (let inDim = 0; inDim < this.#hiddenSize; inDim++) {
                    for (let outDim = 0; outDim < this.#hiddenSize; outDim++) {
                        let wqUpdate = 0, wkUpdate = 0, wvUpdate = 0;
                        const specWeight = isValidNumber(this.#specializationWeights[idx][inDim % this.#hiddenSize][outDim])
                            ? Math.min(Math.max(1 + this.#specializationScores[idx] * this.#specializationWeights[idx][inDim % this.#hiddenSize][outDim], 0.5), 1.5)
                            : 1;
                        for (let pos = 0; pos < this.#inputSize; pos++) {
                            wqUpdate += d_q[pos][outDim] * act.normX[pos][inDim];
                            wkUpdate += d_k[pos][outDim] * act.normX[pos][inDim];
                            const h = Math.floor(outDim / headSize);
                            const dim = outDim % headSize;
                            wvUpdate += d_v[h][pos][dim] * act.normX[pos][inDim];
                        }
                        if (isValidNumber(wqUpdate)) this.#gradientAccumulation[idx].attentionWeights[layer].Wq[inDim][outDim] += (1 - momentum) * wqUpdate * specWeight * adjustedLearningRate;
                        if (isValidNumber(wkUpdate)) this.#gradientAccumulation[idx].attentionWeights[layer].Wk[inDim][outDim] += (1 - momentum) * wkUpdate * specWeight * adjustedLearningRate;
                        if (isValidNumber(wvUpdate)) this.#gradientAccumulation[idx].attentionWeights[layer].Wv[inDim][outDim] += (1 - momentum) * wvUpdate * specWeight * adjustedLearningRate;
                    }
                }

                const gamma1 = transformer.layerNormWeights[layer].gamma1;
                for (let pos = 0; pos < this.#inputSize; pos++) {
                    const x = act.normX[pos];
                    const sq_sum = x.reduce((s, v) => s + v * v, 0);
                    const rms = Math.sqrt(sq_sum / this.#hiddenSize + 1e-6);
                    const norm_x = x.map(v => v / rms);

                    const incoming = gradNormX[pos];
                    const d_gamma = incoming.map((g, i) => g * norm_x[i]);
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        if (isValidNumber(d_gamma[j])) {
                            const update = d_gamma[j] * adjustedLearningRate * 0.2;
                            this.#gradientAccumulation[idx].layerNormWeights[layer].gamma1[j] += (1 - momentum) * update;
                        }
                    }

                    const d_norm = incoming.map((g, i) => g * gamma1[i]);
                    const dot = d_norm.reduce((s, dn, i) => s + dn * norm_x[i], 0);
                    const d_pre = d_norm.map((dn, i) => dn / rms - norm_x[i] * dot / (this.#hiddenSize + 1e-8));

                    for (let j = 0; j < this.#hiddenSize; j++) {
                        grad[pos][j] = gradToAttentionResidual[pos][j] + d_pre[j];
                    }
                }
            }
        });
    }

    predict (inputs) {
        if ( !Array.isArray(inputs) || inputs.length !== this.#inputSize || !inputs.every(isValidNumber) ) { return 0 }

        const predictionResult = this.#updateHiveState(inputs, null, false, false, false, true);
        return predictionResult.probability;
    }

    train (inputs, target) {
        if ( !Array.isArray(inputs) || inputs.length !== this.#inputSize || !inputs.every(isValidNumber) || !isValidNumber(target) ) { return }

        this.#trainingStepCount++;

        const trainingResults = this.#updateHiveState(inputs, target, true, true, true, true);

        this.#accumulateGradients(
            inputs, trainingResults.outputs, target, trainingResults.probability, 
            trainingResults.layerOutputs, trainingResults.activations, trainingResults.attentionIntermediates
        );

        if (this.#trainingStepCount % this.#gradientResetFrequency === 0) {
            const clonedGrads = this.#applyGradients(true, true, this.#gradientResetFrequency);

            this.#gradientAccumulation = structuredClone(clonedGrads);

            const freshResults = this.#updateHiveState(inputs, target , true, false, true, true);

            this.#distillKnowledge(
                freshResults.outputs, freshResults.attentionIntermediates, 
                freshResults.activations, freshResults.layerOutputs
            );

            this.#rollbackGradients(clonedGrads);

            this.#applyGradients(false, false, 1);

            this.#updateHiveState(inputs, target, false, false, true, false);

            this.#gradientAccumulation = this.#setGradientStructure();
        }

        return this.#trainingStepCount;
    }

    dumpState () {
        return this.#saveState()
    }
}

export default HiveMind;