function [y] = dtan(x)

%     if(exp(x)==Inf)
%         y = 0;
%     else
%         y = 2*exp(x)/(1+exp(x))^2;
%     end
    y = 1./(1+cosh(x));

end