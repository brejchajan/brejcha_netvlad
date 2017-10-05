function [ layerNames ] = brejcha_layerNamesDag( net )
%brejcha_layerNamesDag 
%   Returns names of all layers in the DagNN network.
    layerNames = {net.layers.name};


end

