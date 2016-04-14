function [ tempcnn ] = input_convolution(cnn, l)
tempcnn = cnn;
    filterNum = tempcnn.layer{l+1,1};
    
    V = cell(filterNum,1);

    for k = 1:filterNum
        V{k} = -1*conv2(tempcnn.layer{l,2}{1},tempcnn.weight{l+1,k}{1},'valid');
    end

    tempcnn.layer{l+1,2} = V;

end

