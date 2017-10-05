% Author Jan Brejcha 2017
% All rights reserved.

function [ outDims, outszs ] = brejcha_netOutputDimDag( net, layerNames )
% There should be a better way to obtain the output dimension size, 
% but according to Relja's reference, who cares, since this code is 
% not going to be run often.
    
    im = single(imread('peppers.png'));
    if isfield(net, 'onGPU') && net.onGPU
        im= gpuArray(im);
    end
    
    outIdx = []
    for i = 1 : size(layerNames)
        outIdx(end+1) = brejcha_getOutputIndicesDag(net, layerNames{i});
    end
    
    %forward pass
    net.eval({'data', im});
    outsize = max(size(outIdx));
    outDims = zeros(outsize, 1);
    outszs = zeros(outsize, 3);
    for i = 1 : outsize
        %get output
        out = net.vars(outIdx).value;
        outDims(end) = numel(out);
        outszs= size(out);
    end
end

