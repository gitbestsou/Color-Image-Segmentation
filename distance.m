function [d] = distance(x,y)
d = sqrt(sum((x-y).^2));
end