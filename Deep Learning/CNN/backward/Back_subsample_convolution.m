function [tempcnn] = Back_subsample_convolution(cnn, layer, alpha, sc_connection)
    %% from layer l-3 to l-4
    tempcnn = cnn;
    deltaI = tempcnn.layer{layer+1, 4};
    sc_connection = sc_connection';
    
    for x= 1:tempcnn.layer{layer,3}(1)
        for y =1:tempcnn.layer{layer,3}(2)
            for m=1:tempcnn.layer{layer,1}
                temp = 0;
                for i=find(sc_connection(:,m)==1)'
                    for u=0:tempcnn.layer_def{layer+1}.filtersize(1)-1
                        for v=0:tempcnn.layer_def{layer+1}.filtersize(2)-1   
                            temp = temp + deltaI{i}(min(x,tempcnn.layer{layer+1,3}(1)),min(y,tempcnn.layer{layer+1,3}(2)))*tempcnn.weight{layer+1,i}{m}(u+1, v+1);
                        end
                    end
                end   
                deltaM{m}(x,y) = temp*dtan(tempcnn.layer{layer,2}{m}(x,y));
            end
        end
    end
    
    sub = tempcnn.layer_def{layer}.scale;
    for s=1:tempcnn.layer{layer,1}
        temp = 0;
        for x = 1:tempcnn.layer{layer,3}(1)
            for y = 1:tempcnn.layer{layer,3}(2)
                temp = temp + deltaM{s}(x,y) * sum(sum(tanh(tempcnn.layer{layer-1,2}{s}((sub*(x-1)+1):(sub*x),(sub*(y-1)+1):(sub*y))/2)));
            end
        end
        tempcnn.weight{layer,s} = tempcnn.weight{layer,s} - alpha*temp;
        tempcnn.bias{layer,s} = tempcnn.bias{layer,s} - alpha*sum(sum(deltaM{s}));
    end
    
    tempcnn.layer{layer, 4} = deltaM;
end
