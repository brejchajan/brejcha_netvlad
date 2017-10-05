% Author Jan Brejcha 2017

function [ output_args ] = brejcha_whichLayerDag( net, layerName )
% brejcha_whichLayer
%   Returns the index of the layer with layerName in the net in DagNN
%   format.
    iLayer = find(strcmp({net.layers.name}, layerName))
    assert(numel(iLayer) > 0);
end

