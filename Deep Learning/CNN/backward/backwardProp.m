function [ tempcnn ] = backwardProp( Y, cnn , sc_connection, alpha)
%     disp('Backward Propogating'); 
    tempcnn = cnn;
    for l = sort(2:length(tempcnn.layer_def),2, 'descend')
        
        %disp(cnn.layer_def{l}.type);
        if  strcmp(tempcnn.layer_def{l}.type, 'output layer') && strcmp(tempcnn.layer_def{l-1}.type, 'subsample layer')
            tempcnn = Back_Propogation(tempcnn, Y, l, alpha);
            
        elseif strcmp(tempcnn.layer_def{l+1}.type, 'output layer') && strcmp(tempcnn.layer_def{l}.type, 'subsample layer')
            tempcnn = Back_subsample_output(tempcnn, l, alpha);
            
        elseif strcmp(tempcnn.layer_def{l+1}.type, 'subsample layer') && strcmp(tempcnn.layer_def{l}.type, 'convolution layer')
            tempcnn = Back_convolution_subsample(tempcnn, l, alpha );
            
        elseif strcmp(tempcnn.layer_def{l+1}.type, 'convolution layer') && strcmp(tempcnn.layer_def{l}.type, 'subsample layer')
            tempcnn = Back_subsample_convolution(tempcnn, l, alpha, sc_connection{l+1});
            
        end
    end
end