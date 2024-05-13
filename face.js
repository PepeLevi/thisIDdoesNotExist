class ModelGenerator {
    constructor(modelPath) {
        this.modelPath = modelPath;
        this.modelInfo = {};
        this.latentSize = null;
        this.model = {};
    }

    async loadModel() {
        this.modelInfo = await fetch(this.modelPath).then(r => r.json());
        this.latentSize = this.modelInfo.modelLatentDim;
        if (this.modelInfo.modelType === "graph") {
            this.model = await tf.loadGraphModel(this.modelInfo.model);
        } else if (this.modelInfo.modelType === "layers") {
            this.model = await tf.loadLayersModel(this.modelInfo.model);
        } else {
            throw new Error("You must specify 'graph' or 'layers' for the modelType parameter of `manifest.json`.");
        }
    }

    async generate(latentVector) {
        if (!latentVector) latentVector = tf.randomNormal([1, this.latentSize]);
        else latentVector = tf.tensor(latentVector, [1, this.latentSize]);

        let transpose = this.modelInfo.transpose || [0, 1, 2]; // such that shape=[128, 128, 3]
        let imageTensor = this.model.predict(latentVector).squeeze().transpose(transpose);
        if (this.modelInfo.outputRange && this.modelInfo.outputRange[0] === -1) imageTensor = imageTensor.div(tf.scalar(2)).add(tf.scalar(0.5));

        const raw = await tf.browser.toPixels(imageTensor);
        const blob = await this.rawToBlob(raw, imageTensor.shape[0], imageTensor.shape[1]);

        imageTensor.dispose();

        return { raw, blob };
    }

    async rawToBlob(raws, x, y) {
        const arr = Array.from(raws)
        const canvas = new OffscreenCanvas(x, y);
        const ctx = canvas.getContext("2d");

        const imgData = ctx.createImageData(x, y);
        const { data } = imgData;

        for (let i = 0; i < x * y * 4; i += 1) data[i] = arr[i];
        ctx.putImageData(imgData, 0, 0);

        return canvas.convertToBlob({ type: "image/jpeg", quality: 0.95 });
    }
}

// Function to generate image on button click
async function generateImage() {
    const modelPath = 'model/resnet128/manifest.json';
    const modelGenerator = new ModelGenerator(modelPath);
    await modelGenerator.loadModel();
    const { raw, blob } = await modelGenerator.generate();
    // Display the image on a canvas
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const image = new Image();
    image.onload = function () {
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    };
    image.src = URL.createObjectURL(blob);
}

// Run the model when the page is loaded
window.onload = generateImage;
