function [ tempcnn ] = forwardProp(X, cnn , sc_connection)
    tempcnn = cnn;
    tempcnn.layer{1,2}{1} = X;
%     disp('Forward Propogating');
    for l = 1:(length(tempcnn.layer_def)-1)
        
        %disp(cnn.layer_def{l}.type);
        if strcmp(tempcnn.layer_def{l}.type, 'input layer') && strcmp(tempcnn.layer_def{l+1}.type, 'convolution layer')
            tempcnn = input_convolution(tempcnn, l);
            
        elseif strcmp(tempcnn.layer_def{l}.type, 'convolution layer') && strcmp(tempcnn.layer_def{l+1}.type, 'subsample layer')
            tempcnn = convolution_subsample(tempcnn, l);
            
        elseif strcmp(tempcnn.layer_def{l}.type, 'subsample layer') && strcmp(tempcnn.layer_def{l+1}.type, 'convolution layer')
            tempcnn = subsample_convolution(tempcnn, l, sc_connection{l+1});
            
        elseif strcmp(tempcnn.layer_def{l}.type, 'subsample layer') && strcmp(tempcnn.layer_def{l+1}.type, 'output layer')
            tempcnn = subsample_output(tempcnn, l);
            
        end
    end
end

