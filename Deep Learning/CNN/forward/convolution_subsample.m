function [ tempcnn ] = convolution_subsample(cnn, l)
%CONVOLUTION_SUBSAMPLE Summary of this function goes here
%   Detailed explanation goes here

    tempcnn = cnn;

    NumS1 = tempcnn.layer{l+1,1};

    V = cell(NumS1,1);

    for k = 1:NumS1
        reducedmap = tempcnn.layer{l+1,2}{k};
        featuremap = tanh((tempcnn.layer{l,2}{k})/2);
        for i = 1:size(featuremap,1)/2
            for j = 1:size(featuremap,2)/2
               reducedmap(i,j) = sum(sum(featuremap(((tempcnn.layer_def{l+1}.scale*i-1):tempcnn.layer_def{l+1}.scale*i), (tempcnn.layer_def{l+1}.scale*j-1:tempcnn.layer_def{l+1}.scale*j))));
            end
        end
        V{k} = tempcnn.weight{l+1,k} * (reducedmap/(tempcnn.layer_def{l+1}.scale^2)) + tempcnn.bias{l+1,k};
    end
    
    tempcnn.layer{l+1,2} = V;
end

