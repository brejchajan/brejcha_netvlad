% Author Jan Brejcha 2017
% All rights reserved

function [ paramIdx ] = brejcha_getParamIndicesDag( net, paramNames )
%brejcha_getParamIndicesDag 
%   For given cell array of parameter names <paramNames> returns 
%   indexes of the parameters in the net.params structure.
    
    paramsCount = max(size(paramNames));
    paramIdx = size(1, paramsCount);
    for i = 1 : paramsCount
        paramIdx(i) = find(strcmp({net.params.name}, 'score2_filter'));
    end
end

