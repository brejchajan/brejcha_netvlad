% Modified from the original NetVLAD code to add prepare for test of DagNN 
% networks.
% Author Jan Brejcha.

% Prepare a network for 'test' mode
% - disable `precious`
% - optionally remove `dropout`

function net= netPrepareForTestDag(net, removeDropout, removeExtras)
    if nargin<2, removeDropout= false; end
    if nargin<3, removeExtras= false; end
    
    for iLayer= 1:length(net.layers)
        outIdx = net.layers(iLayer).outputIndexes;
        for idx = 1 : max(size(outIdx))
            net.vars(outIdx(idx)).precious = false;
        end
        
        if removeExtras
            paramIdx = net.layers(iLayer).paramIndexes;
            for idx = 1 : max(size(paramIdx))
                iParam = paramIdx(idx);
                if isFieldOrProp(net.params(iParam), 'momentum')
                    net.params(iParam).momentum= [];    
                end
            end
            
        end
    end
    
    if removeDropout
        for iLayer = 1:length(net.layers)
            layer = net.layers(iLayer);
            if isa(layer.block, 'dagnn.DropOut')
                net.removeLayer(layer.name);
            end
        end
    end
    
end



function is= isFieldOrProp(l, propName)
    is= isprop(l, propName) || isfield(l, propName);
end
