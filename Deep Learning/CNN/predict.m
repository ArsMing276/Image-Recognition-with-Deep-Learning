function [ A ] = predict(Ws, A )
    nsample = length(A);
    for i = 1:length(Ws)
        A = sigm([ones(nsample,1) A] * Ws{i}');
    end
end

