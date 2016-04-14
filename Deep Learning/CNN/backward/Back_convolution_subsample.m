function [tempcnn] = Back_convolution_subsample(cnn, layer, alpha)
    %% from l-2 to l-3
    tempcnn  = cnn;
    deltaJ = tempcnn.layer{layer+1, 4};
    sub = tempcnn.layer_def{layer+1}.scale;
    
    for i = 1:tempcnn.layer{layer,1}
        for x = 1:tempcnn.layer{layer,3}(1)
            for y = 1:tempcnn.layer{layer,3}(2)
                deltaI{i}(x,y) = deltaJ{i}(round(x/sub), round(y/sub))*tempcnn.weight{layer+1,i}*dtan(tempcnn.layer{layer,2}{i}(x,y));
            end
        end
    end
    
    for i=1:tempcnn.layer{layer,1}
        for m=1:tempcnn.layer{layer-1,1}
            for u = 0:tempcnn.layer_def{layer}.filtersize(1)-1
                for v = 0:tempcnn.layer_def{layer}.filtersize(2)-1
                    temp = 0;
                    for x = 1:tempcnn.layer{layer,3}(1)
                        for y = 1:tempcnn.layer{layer,3}(2)
                            temp = temp +  deltaI{i}(x,y)*tanh(tempcnn.layer{layer-1,2}{m}(x+u, y+v)/2);
                        end
                    end
                    deltaWim{i,m}(u+1, v+1) = temp;
                end
            end
        end
        deltabi{i} = sum(sum(deltaI{i}));
    end


    for i=1:tempcnn.layer{layer,1}
        for m=1:tempcnn.layer{layer-1,1}
            if(length(tempcnn.weight{layer,i}{m})>0)
                'before'
                tempcnn.weight{layer,i}{m}
                tempcnn.weight{layer,i}{m} = tempcnn.weight{layer,i}{m} - alpha*deltaWim{i,m};
                'after'
                tempcnn.weight{layer,i}{m}
                'delta'
                deltaWim{i,m}
            end
        end
        tempcnn.bias{layer,i} = tempcnn.bias{layer,i} - alpha*deltabi{i};
    end
    tempcnn.layer{layer, 4} = deltaI;
end
