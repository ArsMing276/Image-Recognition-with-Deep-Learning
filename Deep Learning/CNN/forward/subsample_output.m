function [ tempcnn ] = subsample_output( cnn , l)
tempcnn = cnn;
% Output is a matrix
Yj = tempcnn.layer{l,2};


NumJ = length(Yj);
NumOutput = tempcnn.layer{l+1,1};

% initial output
V = zeros(NumOutput,1);

Vk = cell(NumOutput,1);
for k=1:NumOutput
    for j=1:NumJ
        V(k) = V(k) + (sum(sum(tempcnn.weight{l+1,k}{j}.*tanh(Yj{j}/2))));
    end
    Vk{k} = V(k);
end

tempcnn.layer{l+1,2} = Vk;

end

