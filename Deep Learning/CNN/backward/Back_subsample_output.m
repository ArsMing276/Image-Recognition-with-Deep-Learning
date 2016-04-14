function [tempcnn] = Back_subsample_output(cnn, layer, alpha)
    %% from l-1 to l-2
%     tempcnn = cnn;
%     deltaK = tempcnn.layer{layer+1, 4};
%     
%     for j=1:tempcnn.layer{layer,1}
%         temp = 0;
%         for k=1:tempcnn.layer{layer+1,1}
%             temp = temp + deltaK{k}*sum(sum(tempcnn.weight{layer+1,k}{j}));
%         end
%         deltaJ{j} = temp*dtan(tempcnn.layer{layer,2}{j});
%     end
% 
%     sub = tempcnn.layer_def{layer}.scale;
%     for j=1:tempcnn.layer{layer,1}
%         temp = 0;
%         for x = 1:tempcnn.layer{layer,3}(1)   % Node Size
%             for y = 1:tempcnn.layer{layer,3}(2)
%                 temp = temp + deltaJ{j}(x,y)* sum(sum(tanh(tempcnn.layer{layer-1,2}{j}(sub*(x-1)+1:sub*x,sub*(y-1)+1:sub*y)/2)));
%             end
%         end
%         tempcnn.weight{layer,j} = tempcnn.weight{layer,j} - alpha*temp;
%         tempcnn.bias{layer,j} = tempcnn.bias{layer,j} - alpha*sum(sum(deltaJ{j}));
%     end
% 
%     tempcnn.layer{layer, 4} = deltaJ;
% end


    %% from l-1 to l-2
    tempcnn = cnn;
    deltaK = tempcnn.layer{layer+1, 4};
    
    for j=1:tempcnn.layer{layer,1}
        for y = 1:tempcnn.layer{layer,3}(2)
            for x = 1:tempcnn.layer{layer,3}(1)
                temp = 0;
                for k=1:tempcnn.layer{layer+1,1}
                    temp = temp + deltaK{k}*tempcnn.weight{layer+1,k}{j}(x,y);
                end
                deltaJ{j} = temp*dtan(tempcnn.layer{layer,2}{j});
            end
        end
    end

    sub = tempcnn.layer_def{layer}.scale;
    for j=1:tempcnn.layer{layer,1}
        temp = 0;
        for x = 1:tempcnn.layer{layer,3}(1)   % Node Size
            for y = 1:tempcnn.layer{layer,3}(2)
                temp = temp + deltaJ{j}(x,y)* sum(sum(tanh(tempcnn.layer{layer-1,2}{j}(sub*(x-1)+1:sub*x,sub*(y-1)+1:sub*y)/2)));
            end
        end
        tempcnn.weight{layer,j} = tempcnn.weight{layer,j} - alpha*temp;
        tempcnn.bias{layer,j} = tempcnn.bias{layer,j} - alpha*sum(sum(deltaJ{j}));
    end

    tempcnn.layer{layer, 4} = deltaJ;
end