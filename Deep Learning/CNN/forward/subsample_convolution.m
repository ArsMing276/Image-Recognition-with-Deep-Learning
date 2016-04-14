function [ tempcnn ] = subsample_convolution( cnn, l, sc_connection)
    tempcnn = cnn;
    for i = 1:length(tempcnn.layer{l,2})
        Ym{i,1} = tanh(tempcnn.layer{l,2}{i}/2);
    end


    NumM = tempcnn.layer{l,1};
    NumI = tempcnn.layer{l+1,1};

    % all Yi
    MapC2 = cell(NumI, NumM);
    for i = 1:NumI
        for m = 1:NumM
            % sparse weights
            if(sc_connection(m,i) == 1)

                filter = tempcnn.weight{l+1,i}{m};
                y = zeros(size(Ym{m}) - (size(filter)-1));
                for i2 = 1:size(y,1)
                    for j2 = 1:size(y,2)
                        y(i2, j2) = sum(sum(filter.*Ym{m}(i2:i2+size(filter,1)-1,j2:j2+size(filter,2)-1)));
                    end
                end
                MapC2{i,m} = y;
            end
        end
    end

    % some feature map combined
    Yi = cell(NumI,1);
    for i = 1:NumI
        Yi{i} = 0;
        for m =1:NumM
            if(length(MapC2{i,m})>0)
                Yi{i} = MapC2{i,m} + Yi{i};
            end
        end
    end


    for i = 1:NumI
        Yi{i} = Yi{i} + tempcnn.bias{l+1,i};
    end 

    tempcnn.layer{l+1,2} = Yi;

end

