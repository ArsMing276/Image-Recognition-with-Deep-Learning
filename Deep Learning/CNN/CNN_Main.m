%% Reset environment
clear; close all; clc;
addpath('forward', 'backward');
addpath(genpath('DeepLearningToolbox'));

%addpath('AnimalFaceClassification/dataset/AnimalFace');
%% Load Data
load('mnist_uint8');

% training data
labels = unique(train_y, 'rows');
index = randperm(size(train_x,1));
train_x = train_x(index,:);
train_y = train_y(index,:);


training = cell(1);
trainingLabels = cell(1);

for i = 1:size(labels,1)
    counter = 0;
    for n = 1:size(train_y) 
        if counter < 10
            if(sum(labels(i,:) == train_y(n,:)) == 10)
                training{i,1}(n,:) = train_x(n,:);
                trainingLabels{i,1}(n,:) = train_y(n,:);
                counter = counter +1;
            end
        end
    end
    training{i,1} = training{i,1}(sum(training{i,1},2) ~= 0,:);
    trainingLabels{i,1} = trainingLabels{i,1}(sum(trainingLabels{i,1},2) ~= 0,:);
end

training = double(cell2mat(training));
trainingLabels = double(cell2mat(trainingLabels));



%testing data
labels = unique(test_y, 'rows');
index = randperm(size(test_x,1));
test_x = test_x(index,:);
test_y = test_y(index,:);


testing = cell(1);
testingLabels = cell(1);

for i = 1:size(labels,1)
    counter = 0;
    for n = 1:size(test_y) 
        if counter < 10
            if(sum(labels(i,:) == test_y(n,:)) == 10)
                testing{i,1}(n,:) = test_x(n,:);
                testingLabels{i,1}(n,:) = test_y(n,:);
                counter = counter +1;
            end
        end
    end
    testing{i,1} = testing{i,1}(sum(testing{i,1},2) ~= 0,:);
    testingLabels{i,1} = testingLabels{i,1}(sum(testingLabels{i,1},2) ~= 0,:);
end

testing = double(cell2mat(testing));
testingLabels = double(cell2mat(testingLabels));

%% distortion 1 - noise
noiseTest = zeros(100,784);
for i = 1:size(testing,1)
    X = imnoise(reshape(testing(i,:),28,28), 'salt & pepper',0.2);
    X = (X - min(min(X))) / max(max(X)) - min(min(X)) * 2 - 1;
    X = reshape(X, 1, 28*28);
    noiseTest(i,:) = X;
end
%% distortion 1 - scale
scaleTest = zeros(100,784);
for i = 1:size(testing,1)
    X = imresize(reshape(testing(i,:),28,28), 1+rand*0.1);
    X = imcrop(X,[round(size(X)/2)-28/2 27 27]);
    X = (X - min(min(X))) / max(max(X)) - min(min(X)) * 2 - 1;
    X = reshape(X, 1, 28*28);
    scaleTest(i,:) = X;
end
%% distortion 1 - rotation
rotationTest = zeros(100,784);
for i = 1:size(testing,1)
    X = imrotate(reshape(testing(i,:),28,28),(rand*2-1)*90);
    X = imcrop(X,[round(size(X)/2)-28/2 27 27]);
    X = (X - min(min(X))) / max(max(X)) - min(min(X)) * 2 - 1;
    X = reshape(X, 1, 28*28);
    rotationTest(i,:) = X;
end
%% distortion 1 - location
locationTest = zeros(100,784);
for i = 1:size(testing,1)
    X = imtranslate(reshape(testing(i,:),28,28),[4*(rand*2-1), 4*(rand*2-1)],'FillValues',0);
    X = (X - min(min(X))) / max(max(X)) - min(min(X)) * 2 - 1;
    X = reshape(X, 1, 28*28);
    locationTest(i,:) = X;
end
%% distortion 1 - light exposure
exposureTest = zeros(100,784);
for i = 1:size(testing,1)
    X = imadjust(reshape(testing(i,:),28,28));
    X = (X - min(min(X))) / max(max(X)) - min(min(X)) * 2 - 1;
    X = reshape(X, 1, 28*28);
    exposureTest(i,:) = X;
end

%% Load Data
% meaning of y := [straight, left, right, up, neutral, happy, sad, angry, open, sunglasses]
%[X, Y] = PrepData('cmufaces/FULL');
%im2double(rgb2gray(imread()));
%cifar 10

animaltype = ls('AnimalFaceClassification/dataset/AnimalFace/');
animaltype = textscan( animaltype, '%s', 'delimiter', '\t\n' );
animaltype = animaltype{1}(~cellfun('isempty',(animaltype{1})));
% read image
image = cell(1);
label = cell(1);
for t = 1:length(animaltype)
    imagefilenames = ls(sprintf('%s%s','AnimalFaceClassification/dataset/AnimalFace/', animaltype{t}));
    imagefilenames = textscan( imagefilenames, '%s', 'delimiter', '\t\n' );
    imagefilenames = imagefilenames{1}(~cellfun('isempty',(imagefilenames{1})));
    image{t,1} = zeros(length(imagefilenames), 22500);
    label{t,1} = zeros(length(imagefilenames), 1);
    for f = 1:length(imagefilenames)
       X = imread(sprintf('%s%s/%s','AnimalFaceClassification/dataset/AnimalFace/', animaltype{t}, imagefilenames{f}));
       if size(X,1) == 150 && size(X,2) == 150 && size(X,3) == 3
        image{t}(f,:) = reshape(rgb2gray(X),1,22500);
        label{t,1}(f,1) = t;
       end
       image{t} = image{t}(sum(image{t},2) ~= 0,:);
       label{t,1} = label{t,1}(label{t,1} ~= 0);
    end
end

% remove category with sample count < 98;
for i = 1:length(image)
    tf(i) = size(image{i},1) >= 98;
end
image = image(tf,:);
label = label(tf);

% combine data
X = cell2mat(image);
Y = cell2mat(label);
% define Y
label_animal = animaltype(unique(Y));
labels = unique(Y);
for i = 1:length(labels)
   Y(Y == labels(i)) = i;
end
% convert Y into matrix
Y_matrix = zeros(size(Y,1), length(labels));
for i = 1:size(Y_matrix,1)
    Y_matrix(i,Y(i)) = 1;
end
Y = Y_matrix;
% Normalize  X   [-1, 1]
X = (X - min(min(X))) / max(max(X)) - min(min(X)) * 2 - 1;
% rescale Y
Y = Y * 2 - 1;

%% setup training and testing data
% normalize training
for i = 1:size(training,1)
    X = training(i,:);
    training(i,:) = (X - min(min(X))) / max(max(X)) - min(min(X)) * 2 - 1;
end
% rescale label
trainingLabels = trainingLabels * 2 - 1;
%% Define Convolutional Neural Network Layer Setups
% define subsampling to convolutional layer connection relation
sc_connection = cell(1);

sc_connection{4} = zeros(25,16);
while(sum(sum(sc_connection{4}) == 0) > 0)
    for i = 1:25
        sc_connection{4}(i,randperm(16,2)) = 1;
    end
end


sc_connection{6} = zeros(16,25);
while(sum(sum(sc_connection{6}) == 0) > 0)
    for i = 1:16
        sc_connection{6}(i,randperm(25,4)) = 1;
    end
end

sc_connection{8} = zeros(16,25);
while(sum(sum(sc_connection{8}) == 0) > 0)
    for i = 1:16
        sc_connection{8}(i,randperm(25,2)) = 1;
    end
end


cnn.layer_def = {
    struct('type', 'input layer'      , 'layersize',   1, 'nodesize'    , [28 28]) %input layer
    struct('type', 'convolution layer', 'layersize',   25, 'filtersize'  , [5 5])   %convolution layer
    struct('type', 'subsample layer'  , 'layersize',   25, 'scale'       , 2)       %sub sampling layer
    struct('type', 'convolution layer', 'layersize',   16, 'filtersize'  , [3 3])   %convolution layer
    struct('type', 'subsample layer'  , 'layersize',   16, 'scale'       , 2)       %subsampling layer
    %struct('type', 'convolution layer', 'layersize',  25, 'filtersize'  , [4 4])   %convolution layer
    %struct('type', 'subsample layer'  , 'layersize',  25, 'scale'       , 2)       %subsampling layer
    %struct('type', 'convolution layer', 'layersize',  25, 'filtersize'  , [3 3])   %convolution layer
    %struct('type', 'subsample layer'  , 'layersize',  25, 'scale'       , 2)       %subsampling layer
    struct('type', 'output layer'     , 'layersize',  10, 'nodesize'    , 1)       %subsampling layer
};

%% initialize Layers

[cnn.layer, cnn.weight, cnn.bias]= initializer(cnn.layer_def);


%%
iter = 100;
X = training;
Y = trainingLabels;
index = randperm(size(X,1),2);

%%
alpha = 0.009;
for i = 1:iter
    disp(sprintf('Iteration %3d', i));
    cnn = cnnTrain(X(index,:), Y(index,:), cnn, sc_connection, alpha);
end

%% predict

pred = forwardProp(reshape(scaleTest(61,:), cnn.layer{1,3}(1), cnn.layer{1,3}(2))', cnn, sc_connection);
figure(1)
colormap gray;
imagesc(pred.layer{1,2}{1})

figure(2)
colormap gray;
for p = 1:25
    subplot(5,5,p)
    imagesc(pred.layer{3,2}{p})
end
%% ANN
Y(Y == -1) = 0;
%%
% hidden layer(s)
hiddenlayer = [200];
nn = nnsetup([size(X,2) hiddenlayer size(Y,2)]);
nn.activation_function = 'sigm';
%%
nn.learningRate = 0.1;
opts.numepochs = 300;
opts.batchsize = 1;
%% Train NN
nn = nntrain(nn, X, Y , opts);
%% Predict training

Predictions = predict(nn.W, X);

for i = 1:size(Predictions,1)
Predictions(i,:)  = max(Predictions(i,:))  == Predictions(i,:);
end

Accuracy = sum(sum(Predictions == testingLabels,2) == 10)/100

%% Noise
Predictions = predict(nn.W, noiseTest);

for i = 1:size(Predictions,1)
Predictions(i,:)  = max(Predictions(i,:))  == Predictions(i,:);
end

Accuracy = sum(sum(Predictions == testingLabels,2) == 10)/100
%% Scale
Predictions = predict(nn.W, scaleTest);

for i = 1:size(Predictions,1)
Predictions(i,:)  = max(Predictions(i,:))  == Predictions(i,:);
end

Accuracy = sum(sum(Predictions == testingLabels,2) == 10)/100
%% Rotation
Predictions = predict(nn.W, rotationTest);

for i = 1:size(Predictions,1)
Predictions(i,:)  = max(Predictions(i,:))  == Predictions(i,:);
end

Accuracy = sum(sum(Predictions == testingLabels,2) == 10)/100
%% Shift
Predictions = predict(nn.W, locationTest);

for i = 1:size(Predictions,1)
Predictions(i,:)  = max(Predictions(i,:))  == Predictions(i,:);
end

Accuracy = sum(sum(Predictions == testingLabels,2) == 10)/100
%% Exposure
Predictions = predict(nn.W, exposureTest);

for i = 1:size(Predictions,1)
Predictions(i,:)  = max(Predictions(i,:))  == Predictions(i,:);
end

Accuracy = sum(sum(Predictions == testingLabels,2) == 10)/100
