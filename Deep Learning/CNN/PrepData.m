function [ data, labels ] = PrepData( folderpath ) 
%% import image

%get image names
addpath(folderpath);
imagefilenames = ls(folderpath);
imagefilenames = textscan( imagefilenames, '%s', 'delimiter', '\t\n' );
imagefilenames = imagefilenames{1}(~cellfun('isempty',(imagefilenames{1})));

%%

%get image

data = zeros(length(imagefilenames),size(imread(imagefilenames{1}),1)*size(imread(imagefilenames{1}),2));
for i = 1:length(imagefilenames)
    %read in
    img = imread(imagefilenames{i});
    %vectorize
    img = reshape(img, [1,size(img,1)*size(img,2)]);
    data(i,:) = img;
end

clear i img;

%% 
% prep label
% label = [straight, left, right, up, neutral, happy, sad, angry, open, sunglasses]
labels = zeros(length(imagefilenames), 10);
for i = 1:length(imagefilenames)
    text = textscan( imagefilenames{i}, '%s', 'delimiter', '_.' );
    text = text{1}(2:4);
    if strcmp(text{1}, 'straight')
            labels(i,1) = 1;
    elseif strcmp(text{1}, 'left')
            labels(i,2) = 1;
    elseif strcmp(text{1}, 'right')
            labels(i,3) = 1;
    elseif strcmp(text{1}, 'up')
            labels(i,4) = 1;
    else
            disp(i) 
            disp(text{1})
    end

    if strcmp(text{2}, 'neutral')
            labels(i,5) = 1;
    elseif strcmp(text{2}, 'happy')
            labels(i,6) = 1;
    elseif strcmp(text{2}, 'sad')
            labels(i,7) = 1;
    elseif strcmp(text{2}, 'angry')
            labels(i,8) = 1;
    else
            disp(i) 
            disp(text{2})
    end 

    if strcmp(text{3}, 'open')
            labels(i,9) = 1;
    elseif strcmp(text{3}, 'sunglasses')
            labels(i,10) = 1;
    else
            disp(i) 
            disp(text{3})
    end 

end

end

