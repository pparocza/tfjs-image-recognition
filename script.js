// 8. JAVASCRIPT: KEY CONSTANTS AND LISTENERS

const STATUS = document.getElementById('status');
const VIDEO = document.getElementById('webcam');
const ENABLE_CAM_BUTTON = document.getElementById('enableCam');
const RESET_BUTTON = document.getElementById('reset');
const TRAIN_BUTTON = document.getElementById('train');
const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;
const STOP_DATA_GATHER = -1;
const CLASS_NAMES = [];

ENABLE_CAM_BUTTON.addEventListener('click', enableCam);
TRAIN_BUTTON.addEventListener('click', trainAndPredict);
RESET_BUTTON.addEventListener('click', reset);

// find all buttons that have a class of dataCollector using document.querySelectorAll()
let dataCollectorButtons = document.querySelectorAll('button.dataCollector');
for (let i = 0; i < dataCollectorButtons.length; i++){
	dataCollectorButtons[i].addEventListener('mousedown', gatherDataForClass);
	dataCollectorButtons[i].addEventListener('mouseup', gatherDataForClass);
	// Populate the human readable names for classes.
	CLASS_NAMES.push(dataCollectorButtons[i].getAttribute('data-name'));
}

let mobileNet = undefined;
let gatherDataState = STOP_DATA_GATHER;
let videoPlaying = false;
let trainingDataInputs = [];
let trainingDataOutputs = [];
let examplesCount = [];
let predict = false;

// 9. LOAD THE MOBILENET BASE MODEL

/**
 * Loads the MobileNet model and warms it up so ready for use
 */
async function loadMobileNetFeatureModel(){
	const URL =
    'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1';
	mobileNet = await tf.loadGraphModel(URL, {fromTFHub: true});
	STATUS.innerText = 'MobileNet v3 loaded successfully!';

	// War up the model by passing zeros through it once
	tf.tidy(function(){
		let answer = mobileNet.predict(tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3]));
		console.log(answer.shape);
	});
}

// Call the function immediately to start loading.
loadMobileNetFeatureModel();

// 10. DEFINE THE NEW MODEL HEAD

let model = tf.sequential();
model.add(tf.layers.dense({inputShape: [1024], units: 128, activation: 'relu'}));
model.add(tf.layers.dense({units: CLASS_NAMES.length, activation: 'softmax'}));

model.summary();

// Compile the model with the defined optimizer and specify a loss function use.
model.compile({
	// Adam changes the learning rate over time which is useful.
	optimizer: 'adam',
	// Use the correct loss function. If 2 classes of data, must use binaryCrossentropy.
	// Else categoricalCrossentropy is used if more than 2 classes.
	loss: (CLASS_NAMES.length === 2) ? 'binaryCrossentropy' : 'categoricalCrossEntropy',
	// As this is a classification problem you can record accuracy in the logs too!
	metrics: ['accuracy']
});

// 11. ENABLE THE WEBCAM

function hasGetUserMedia(){
	return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

function enableCam(){
	if(hasGetUserMedia()){
		// get Usermedia parameters
		const constraints = {
			video: true,
			width: 640,
			height: 480
		};

		// Activate the webcam stream.
		navigator.mediaDevices.getUserMedia(constraints).then(function(stream){
			VIDEO.srcObject = stream;
			VIDEO.addEventListener('loadeddata', function(){
				videoPlaying = true;
				ENABLE_CAM_BUTTON.classList.add('removed');
			});
		});
	}
	else {
		console.warn('getUserMedia() is not supported by your browser');
	}
}

// 12. DATA COLLECTION BUTTON EVENT HANDLER
/**
 * Handle Data Gather for button mouseup/mousedown
 */

function gatherDataForClass(){
	let classNumber = parseInt(this.getAttribute('data-1hot'));
	gatherDataState = (gatherDataState === STOP_DATA_GATHER) ? classNumber : STOP_DATA_GATHER;
	dataGatherLoop();
}

// 13. DATA COLLECTION
function dataGatherLoop(){
	if(videoPlaying && gatherDataState !== STOP_DATA_GATHER){
		// tf.tidy diposes any created tensors in the code that follows
		let imageFeatures = tf.tidy(function(){
			let videoFrameAsTensor = tf.browser.fromPixels(VIDEO);
			let resizedTensorFrame = tf.image.resizeBilinear(videoFrameAsTensor, [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH], true);
			let normalizedTensorFrame = resizedTensorFrame.div(255);
			return mobileNet.predict(normalizedTensorFrame.expandDims()).squeeze();
		});

		trainingDataInputs.push(imageFeatures);
		trainingDataOutputs.push(gatherDataState);

		// Initialize array index element if currently undefined.
		if(examplesCount[gatherDataState] === undefined){
			examplesCount[gatherDataState] = 0;
		}
		examplesCount[gatherDataState]++;

		STATUS.innerText = '';
		for(let n = 0; n < CLASS_NAMES.length; n++){
			STATUS.innerText += CLASS_NAMES[n] + 'data count: ' + examplesCount[n] + '.';
		}
		window.requestAnimationFrame(dataGatherLoop);
	}
}

// 14. TRAIN AND PREDICT

async function trainAndPredict(){
	// stop any current predictions from taking place
	predict = false;
	// shuffle input and output arrays to ensure the order does not cause issues in training
	tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs);
	// convert trainingDataOutputs to be a tensor 1d of type int32 so it is ready to be used in a one hot encoding
	let outputsAsTensor = tf.tensor1d(trainingDataOutputs, 'int32');
	let oneHotOutputs = tf.oneHot(outputsAsTensor, CLASS_NAMES.length);
	// convert an array of tensors to a 2D tensor
	let inputsAsTensor = tf.stack(trainingDataInputs);

	// train the custom model head
	let results = await model.fit(inputsAsTensor, oneHotOutputs, {shuffle: true, batchSize: 5, epochs: 10, callbacks: {onEpochEnd: logProgress} });

	// dispose of created tensors
	outputsAsTensor.dispose();
	oneHotOutputs.dispose();
	inputsAsTensor.dispose();
	predict = true;
	predictLoop();
}

function logProgress(epoch, logs){
	console.log('Data for epoch ' + epoch, logs);
}

function predictLoop(){
	// predictions are only made after a model is trained and available
	if(predict){
		tf.tidy(function(){
			// grab a frame from the webcam and normalize it (div(255))
			let videoFrameAsTensor = tf.browser.fromPixels(VIDEO).div(255);
			// resize the tensor to 244x244
			let resizedTensorFrame = tf.image.resizeBilinear(videoFrameAsTensor, [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH], true);

			// pass the data through the MobileNet model
			let imageFeatures = mobileNet.predict(resizedTensorFrame.expandDims());
			// pass imageFeatures through predict, and make it 1-dimensional (squeeze())
			let prediction = model.predict(imageFeatures).squeeze();
			// find the index that has the highest value (argMax()) and convert the tensor to an array (arraySync())
			let highestIndex = prediction.argMax().arraySync();
			// get the prediction confidence scores
			let predictionArray = prediction.arraySync();

			STATUS.innerText = 'Prediction: ' + CLASS_NAMES[highestIndex] + ' with ' + Math.floor(predictionArray[highestIndex] * 100) + '% confidence';
		});

		window.requestAnimationFrame(predictLoop);
	}
}

// 15. IMPLEMENT THE RESET BUTTON
/**
 * Purge data and start over. Note this does not dispose of the loaded
 * MobileNet model and MLP head tensors as you will need to reuse them
 * to train a new model
 */
function reset(){
	predict = false;
	examplesCount.length = 0;
	// dispose of each tensor in trainingDataInputs
	for(let i = 0; i < trainingDataInputs.length; i++){
		trainingDataInputs[i].dispose();
	}
	trainingDataInputs.length = 0;
	trainingDataOutputs.length = 0;
	STATUS.innerText = 'No data collected';

	console.log('Tensors in memory: ' + tf.memory().numTensors);
}
