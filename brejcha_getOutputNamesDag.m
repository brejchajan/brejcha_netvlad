function [ outNames ] = brejcha_getOutputNamesDag( net, layerName )
%getOutputIndicesDag returns indexes of outputs from given layer name in a
%DagNN net.

    layer = find(strcmp({net.layers.name}, layerName));
    if ~isempty(layer)
        outNames = net.layers(layer).outputs;
    else
        error(['Unable to find layer with name: ' layerName ... 
            ', unable to proceed']);
    end

end