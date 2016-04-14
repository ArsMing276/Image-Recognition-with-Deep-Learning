function [tempcnn] = Back_Propogation(cnn, y_true, layer, alpha)
% y_true is a matrix
% layer is current layer in backward propogation

    tempcnn  = cnn;
    for k=1:tempcnn.layer{layer,1}
        deltaK{k} = (tanh(tempcnn.layer{layer,2}{k}/2) - y_true(k))*dtan(tempcnn.layer{layer,2}{k});
        
    end
    
    for k=1:tempcnn.layer{layer,1}
        for j= 1:tempcnn.layer{layer-1,1}
            %dell = alpha*deltaK{k}*tanh(tempcnn.layer{layer-1,2}{j}/2);
            %disp(dell(1:4,1:4));
            %dt = tanh(tempcnn.layer{layer-1,2}{j}/2);
            %disp(dt(1:4,1:4));
            
            tempcnn.weight{layer,k}{j} = tempcnn.weight{layer,k}{j} - alpha*deltaK{k}*tanh(tempcnn.layer{layer-1,2}{j}/2);
            %disp(sum(sum(tempcnn.weight{layer,k}{j} == cnn.weight{layer,k}{j})));
        end
    end
    tempcnn.layer{layer,4} = deltaK;
    
    
    
end
 