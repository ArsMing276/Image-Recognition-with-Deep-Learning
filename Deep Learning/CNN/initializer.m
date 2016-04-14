function [ layer, weights, bias] = initializer( layer_def )

    layer = cell(1);
    weights = cell(1);
    bias = cell(1);
    
    for i = 1:length(layer_def)
        if strcmp(layer_def{i}.type, 'input layer')
            % layer Size
            layer{i,1} = layer_def{i}.layersize; 
            % allocate memory for input layer
            layer{i,2} = zeros(layer_def{i}.nodesize(1:2)); 
            % node size
            layer{i,3} = layer_def{i}.nodesize(1:2); 
            
        elseif strcmp(layer_def{i}.type, 'convolution layer') && strcmp(layer_def{i-1}.type, 'input layer')
            % layer Size
            layer{i,1} = layer_def{i}.layersize; 
            % node size
            layer{i,3} = (layer{i-1,3} - layer_def{i}.filtersize) + 1;
            % allocate memory for convolution layer's V
            layer{i,2} = cell(layer_def{i}.layersize,1); 
            for n = 1:length(layer{i,2})
                layer{i,2}{n} = zeros(layer{i,3});
                weights{i,n} = cell(layer{i-1,1},1);
                for k = 1:layer{i-1,1}
                    weights{i,n}{k} = rand(layer_def{i}.filtersize)*2-1;
                end
                
                bias{i,n} = rand(1)*2-1;
            end
        elseif strcmp(layer_def{i}.type, 'convolution layer') && strcmp(layer_def{i-1}.type, 'subsample layer')
            % layer Size
            layer{i,1} = layer_def{i}.layersize; 
            % node size
            layer{i,3} = (layer{i-1,3} - layer_def{i}.filtersize) + 1;
            % allocate memory for convolution layer's V
            layer{i,2} = cell(layer_def{i}.layersize,1); 
            for n = 1:length(layer{i,2})
                layer{i,2}{n} = zeros(layer{i,3});
                weights{i,n} = cell(layer{i-1,1},1);
                for k = 1:layer{i-1,1}
                    weights{i,n}{k} = rand(layer_def{i}.filtersize)*2-1;
                end
                bias{i,n} = rand(1)*2-1;
            end
        elseif strcmp(layer_def{i}.type, 'subsample layer')
            % layer Size
            layer{i,1} = layer_def{i}.layersize; 
            % allocate memory for subsample layer's V
            layer{i,2} = cell(layer_def{i-1}.layersize,1);
            for n = 1:length(layer{i,2})
                layer{i,2}{n} = zeros(layer{i-1,3}/layer_def{i}.scale);
                weights{i,n} = rand(1)*2-1;
                bias{i,n} = rand(1)*2-1;
            end
            % node size
            layer{i,3} = layer{i-1,3}/layer_def{i}.scale;
    
        elseif strcmp(layer_def{i}.type, 'output layer')
            % layer Size
            layer{i,1} = layer_def{i}.layersize; 
            % allocate memory for output layer's V
            layer{i,2} = cell(layer_def{i}.layersize,1);
            % node Size
            layer{i,3} = [layer_def{i}.nodesize, 1];
            
            for k = 1:layer{i,1}
                weights{i,k} = cell(layer{i-1,1},1);
                for kn = 1:layer{i-1,1}
                    weights{i,k}{kn} = rand(layer{i-1,3})*2-1;
                end
            end
        end
    end
    
    layer{1,2} = cell(layer(1,2));
end

