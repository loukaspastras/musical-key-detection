document.addEventListener('DOMContentLoaded', async function() {
    console.log("DOM fully loaded and parsed");

    // Load the TensorFlow.js model
    const model = await tf.loadLayersModel('my_tfjs_model/model.json');
    console.log('Model loaded');

    document.getElementById('file-input').addEventListener('change', async function(event) {
        await handleFileSelect(event, model);
    });

    document.getElementById('select-file-button').addEventListener('click', function() {
        document.getElementById('file-input').click();
    });
});

async function handleFileSelect(event, model) {
    console.log("File selected");
    const file = event.target.files[0];

    if (file) {
        document.getElementById('file-upload-container').insertAdjacentHTML('afterend', `<p id="file-name-cell">${file.name}</p>`);
        const resultTable = document.getElementById('result-table');

        if (resultTable) {
            resultTable.classList.remove('hidden');

            const loadingCircle = document.getElementById('loading-circle');
            if (loadingCircle) {
                loadingCircle.classList.remove('hidden');
                loadingCircle.classList.add('loader');

                try {
                    const pcpVector = await extractPCP(file);
                    const transformedPcpVector = transformInput(tf.tensor2d([pcpVector], [1, 12])); // Apply your transformation here
                    const predictions = await detectKey(transformedPcpVector, model);
                    displayKeyResults(predictions);
                    displayPCP(pcpVector);
                } catch (error) {
                    console.error('Error detecting key:', error);
                    document.getElementById('key-cell-1').textContent = 'Error';
                    document.getElementById('key-cell-2').textContent = 'Error';
                    document.getElementById('key-cell-3').textContent = 'Error';
                } finally {
                    loadingCircle.classList.remove('loader');
                    loadingCircle.classList.add('hidden');
                }

                document.getElementById('select-file-button').textContent = 'Select Another File';
                document.getElementById('select-file-button').addEventListener('click', function() {
                    location.reload();
                });
                document.getElementById('file-input').disabled = true;
            } else {
                console.error('Element with id="loading-circle" not found.');
            }
        } else {
            console.error('Element with id="result-table" not found.');
        }
    } else {
        console.error('No file selected.');
    }
}

class Yin {
    constructor(bufferSize, threshold) {
        this.bufferSize = bufferSize;
        this.halfBufferSize = Math.floor(bufferSize / 2);
        this.probability = 0.0;
        this.threshold = threshold;
        this.yinBuffer = new Float32Array(this.halfBufferSize);
    }

    difference(buffer) {
        const yinBuffer = this.yinBuffer;
        yinBuffer.fill(0);

        for (let tau = 0; tau < this.halfBufferSize; tau++) {
            for (let i = 0; i < this.halfBufferSize; i++) {
                const delta = buffer[i] - buffer[i + tau];
                yinBuffer[tau] += delta * delta;
            }
        }
    }

    cumulativeMeanNormalizedDifference() {
        const yinBuffer = this.yinBuffer;
        let runningSum = 0;
        yinBuffer[0] = 1;

        for (let tau = 1; tau < this.halfBufferSize; tau++) {
            runningSum += yinBuffer[tau];
            yinBuffer[tau] = (tau * yinBuffer[tau]) / runningSum;
        }
    }

    absoluteThreshold() {
        const yinBuffer = this.yinBuffer;
        for (let tau = 2; tau < this.halfBufferSize; tau++) {
            if (yinBuffer[tau] < this.threshold) {
                while (tau + 1 < this.halfBufferSize && yinBuffer[tau + 1] < yinBuffer[tau]) {
                    tau++;
                }
                this.probability = 1 - yinBuffer[tau];
                return tau;
            }
        }

        this.probability = 0;
        return -1;
    }

    parabolicInterpolation(tauEstimate) {
        const yinBuffer = this.yinBuffer;
        const x0 = tauEstimate < 1 ? tauEstimate : tauEstimate - 1;
        const x2 = tauEstimate + 1 < this.halfBufferSize ? tauEstimate + 1 : tauEstimate;
        const s0 = yinBuffer[x0];
        const s1 = yinBuffer[tauEstimate];
        const s2 = yinBuffer[x2];

        if (x0 === tauEstimate) {
            return s1 <= s2 ? tauEstimate : x2;
        } else if (x2 === tauEstimate) {
            return s1 <= s0 ? tauEstimate : x0;
        } else {
            return tauEstimate + (s2 - s0) / (2 * (2 * s1 - s2 - s0));
        }
    }

    getPitch(buffer, sampleRate) {
        this.difference(buffer);
        this.cumulativeMeanNormalizedDifference();
        const tauEstimate = this.absoluteThreshold();
        if (tauEstimate !== -1) {
            const betterTau = this.parabolicInterpolation(tauEstimate);
            return sampleRate / betterTau;
        }
        return -1;
    }

    getProbability() {
        return this.probability;
    }
}

function closestPowerOfTwo(num) {
    return Math.pow(2, Math.ceil(Math.log2(num)));
}

function averageArrays(arrays) {
    const length = arrays[0].length;
    const sum = new Array(length).fill(0);

    arrays.forEach(array => {
        array.forEach((value, index) => {
            sum[index] += value;
        });
    });

    return sum.map(value => value / arrays.length);
}

function softmax(arr) {
    const max = Math.max(...arr);
    const exps = arr.map(value => Math.exp(value - max));
    const sum = exps.reduce((acc, val) => acc + val, 0);
    return exps.map(value => value / sum);
}

async function extractPCP(file) {
    console.log('Extracting PCP from file:', file.name);

    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const arrayBuffer = await file.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

    const channelData = audioBuffer.getChannelData(0);
    const sampleRate = audioBuffer.sampleRate;

    // Split buffer into small frames
    const frameSize = 1024;
    const hopSize = frameSize/4;
    const frames = [];

    for (let i = 0; i < channelData.length - frameSize; i += hopSize) {
        frames.push(channelData.slice(i, i + frameSize));
    }

    const yin = new Yin(frameSize, 0.5);
    const chromaSum = Array(12).fill(0);

    frames.forEach(frame => {
        const pitch = yin.getPitch(frame, sampleRate);
        if (pitch !== -1) {
            const midi = 69 + 12 * Math.log2(pitch / 440);
            const cents = midi % 1;
            if (cents < 0.1 || cents > 0.9) {
                const note = Math.round(midi) % 12;
                chromaSum[note] += 1;
            }
        }
    });

    // Normalize chromaSum by dividing each value by the total sum
    const sum = chromaSum.reduce((acc, val) => acc + val, 0);
    const normalizedChromaSum = chromaSum.map(value => value / sum);

    console.log('Normalized Chroma Sum:', normalizedChromaSum);
    return normalizedChromaSum;
}

function normalizeProfile(profile) {
    const mean = tf.mean(profile, -1, true);
    const X = tf.sub(profile, mean);
    const sumX2 = tf.sum(tf.square(X), -1, true);
    const normalized = tf.div(X, tf.sqrt(sumX2));
    return normalized;
}

function transformInput(X) {
    const batchSize = X.shape[0];
    const normalized = normalizeProfile(X);

    // Extend each profile and concatenate
    const extended = tf.concat([normalized, normalized.slice([0, 0], [-1, 11])], 1);

    // Reshape to match the desired output shape
    const transformed = tf.reshape(extended, [batchSize, 23, 1]);

    return transformed;
}

async function detectKey(transformedPcpVector, model) {
    console.log('Detecting key from transformed PCP vector:', transformedPcpVector);

    // Make prediction using the loaded model
    const prediction = model.predict(transformedPcpVector);

    // Get the probabilities by applying softmax
    const probabilities = await tf.softmax(prediction).data();

    // Get the indices of the top 3 probabilities
    const topIndices = Array.from(probabilities)
        .map((p, i) => ({ probability: p, index: i }))
        .sort((a, b) => b.probability - a.probability)
        .slice(0, 3);

    const keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
                  'Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am', 'A#m', 'Bm'];

    // Return the top 3 keys with their probabilities
    return topIndices.map(item => ({
        key: keys[item.index],
        probability: item.probability/(probabilities[0]+probabilities[1]+probabilities[2])
    }));
}

function displayKeyResults(predictions) {
    document.getElementById('result-table').classList.remove('hidden');
    predictions.forEach((prediction, index) => {
        document.getElementById(`key-cell-${index + 1}`).textContent = prediction.key;
        document.getElementById(`confidence-cell-${index + 1}`).textContent = (prediction.probability * 100).toFixed(2) + '%';
    });
}

function displayPCP(pcpVector) {
    const ctx = document.getElementById('pcp-chart').getContext('2d');
    const pcpChartContainer = document.getElementById('pcp-chart-container');
    if (pcpChartContainer) {
        pcpChartContainer.classList.remove('hidden');
    }

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'],
            datasets: [{
                label: 'PCP',
                data: pcpVector,
                backgroundColor: 'rgba(255, 140, 0, 0.2)',
                borderColor: 'rgba(255, 140, 0, 1)',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}
