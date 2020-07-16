%% Load Training Data 
ttds = tabularTextDatastore("train_metadata.csv","ReadSize",'file',"TextType","string");
train =read(ttds);
%% Preview the datastore. 
preview(ttds)
%% Load Training Labels
trainlabels = readtable("train_labels.csv");
trainlabels.stalled = categorical(trainlabels.stalled);
%% Prepare nano dataset
nanotrain = train(train.nano == 'True',:);
nanotrainlabels = trainlabels(train.nano == 'True',:);

%% Acces & Process Video Files
tempfds = fullfile(tempdir,"fds_nano.mat");

if exist(tempfds,'file')
    load(tempfds,'fds')
else
    fds = fileDatastore(nanotrain.url,'ReadFcn', @readVideo);
    save(tempfds,"fds");
end

files = fds.Files;
dataOut = preview(fds);
tile = imtile(dataOut);
imshow(tile);
%% Classification

% Load Pretrained Convolutional Network
netCNN = googlenet;

%% Transform datastore  
inputSize = netCNN.Layers(1).InputSize(1:2);
fdsReSz = transform(fds,@(x) imresize(x,inputSize));     

%% Calculate sequences
layerName = "pool5-7x7_s1";

tempFile = fullfile(tempdir,"sequences_nano.mat");

if exist(tempFile,'file')
    load(tempFile,"sequences")
else
    numFiles = numel(files);
    sequences = cell(numFiles,1);
    
    for i = 1:numFiles
        fprintf("Reading file %d of %d...\n", i, numFiles);
        sequences{i,1} = activations(netCNN,read(fdsReSz),layerName,...
            'OutputAs','columns','ExecutionEnvironment',"auto");
    end
    
    save(tempFile,"sequences");
end

% Prepare Training Data
labels = nanotrainlabels.stalled;

numObservations = numel(sequences);
idx = randperm(numObservations);
N = floor(0.9 * numObservations);

idxTrain = idx(1:N);
sequencesTrain = sequences(idxTrain);
labelsTrain = labels(idxTrain);

idxValidation = idx(N+1:end);
sequencesValidation = sequences(idxValidation);
labelsValidation = labels(idxValidation);

%% Plot sequence length
numObservationsTrain = numel(sequencesTrain);
sequenceLengths = zeros(1,numObservationsTrain);

for i = 1:numObservationsTrain
    sequence = sequencesTrain{i};
    sequenceLengths(i) = size(sequence,2);
end

figure
histogram(sequenceLengths)
title("Sequence Lengths")
xlabel("Sequence Length")
ylabel("Frequency")

% Create LSTM Network
numFeatures = size(sequencesTrain{1},1);
numClasses = 2;

layers = [
    sequenceInputLayer(numFeatures,'Name','sequence')
    bilstmLayer(2000,'OutputMode','last','Name','bilstm')
    dropoutLayer(0.5,'Name','drop')
    fullyConnectedLayer(numClasses,'Name','fc')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classification')];

% Specify Training Options
miniBatchSize = 16;
numObservations = numel(sequencesTrain);
numIterationsPerEpoch = floor(numObservations / miniBatchSize);

options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',1e-4, ...
    'GradientThreshold',2, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{sequencesValidation,labelsValidation}, ...
    'ValidationFrequency',numIterationsPerEpoch, ...
    'Plots','training-progress', ...
    'Verbose',false, ...
    'ExecutionEnvironment','auto');

% Train LSTM Network
[netLSTM,info] = trainNetwork(sequencesTrain,labelsTrain,layers,options);

%% calculate the classification accuracy of the network 
YPred = classify(netLSTM,sequencesValidation,'MiniBatchSize',miniBatchSize);
YValidation = labelsValidation;
accuracy = mean(YPred == YValidation);

% Assemble Video Classification Network
cnnLayers = layerGraph(netCNN);
layerNames = ["data" "pool5-drop_7x7_s1" "loss3-classifier" "prob" "output"];
cnnLayers = removeLayers(cnnLayers,layerNames);

% *Add Sequence Input Layer*

inputSize = netCNN.Layers(1).InputSize(1:2);
averageImage = netCNN.Layers(1).Mean;

inputLayer = sequenceInputLayer([inputSize 3], ...
    'Normalization','zerocenter', ...
    'Mean',averageImage, ...
    'Name','input');

layers = [
    inputLayer
    sequenceFoldingLayer('Name','fold')];

lgraph = addLayers(cnnLayers,layers);
lgraph = connectLayers(lgraph,"fold/out","conv1-7x7_s2");

% *Add LSTM Layers*
lstmLayers = netLSTM.Layers;
lstmLayers(1) = [];

layers = [
    sequenceUnfoldingLayer('Name','unfold')
    flattenLayer('Name','flatten')
    lstmLayers];

lgraph = addLayers(lgraph,layers);
lgraph = connectLayers(lgraph,"pool5-7x7_s1","unfold/in");

lgraph = connectLayers(lgraph,"fold/miniBatchSize","unfold/miniBatchSize");

% *Assemble Network*
analyzeNetwork(lgraph)
net = assembleNetwork(lgraph);

% Prepare Test Data  
testfds = fileDatastore('s3://drivendata-competition-clog-loss/test/','ReadFcn', @readVideo);
testfdsReSz = transform(testfds,@(x) {imresize(x,inputSize)});

% Classify Using Test Data
testFiles = testfds.Files;
numTestFiles = numel(testFiles);

YPred = cell(numTestFiles,1);

for i = 1:numTestFiles
    fprintf("Reading file %d of %d...\n", i, numTestFiles);
    YPred{i,1} = classify(net,read(testfdsReSz),'ExecutionEnvironment','auto');
end

%% Save Submission to File
test = readtable("test_metadata.csv");
testResults = table(test.filename,YPred(:,1),'VariableNames',{'filename','stalled'});
writetable(testResults,'testResults_nano.csv');

%% Helper Functions
function video = readVideo(filename)

vr = VideoReader(filename);
i = 0;
% video = zeros;

while hasFrame(vr)
    i = i+1;
    frame = readFrame(vr);
    if i < 2
        Bbox = detectROI(frame);
    end
    frame = imcrop(frame, Bbox);
    video(:,:,:,i)=frame; 
end

end

function [Bbox] = detectROI(frameIn)

%% Setup the detector and initialize variables
persistent detector 
if isempty(detector)
    detector = vision.BlobAnalysis('BoundingBoxOutputPort',true,'MajorAxisLengthOutputPort',true);
end

threshold = [104 255; 13 143; 9 98];

mask = (frameIn (:,:,1) >= threshold(1,1))& (frameIn (:,:,1) <= threshold(1,2))&...
    (frameIn (:,:,2) >= threshold(2,1))& (frameIn (:,:,2) <= threshold(2,2))&...
    (frameIn (:,:,3) >= threshold(3,1))& (frameIn (:,:,3) <= threshold(3,2));

[~, ~, Bbox1, majorAxis] = detector(mask);

if ~isempty(majorAxis)
% Identify Largest Blob
    [~,mIdx] = max(majorAxis);
    Bbox = Bbox1(mIdx,:);
 
end 

end 
