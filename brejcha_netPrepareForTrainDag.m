% Modified from original NetVLAD version netPrepareForTrain.m
% Jan Brejcha 2017

function net= brejcha_netPrepareForTrainDag(net, backPropToLayer)
    if nargin<2, backPropToLayer= 1; end
    
    nLayers= numel(net.layers);
    for i= 1:nLayers
        layerParams = net.layers(i).params;
        if i>=backPropToLayer && ~isempty(layerParams)
            paramIdx = net.layers(i).paramIndexes;
            paramCnt = numel(layerParams);
            for J = 1 : paramCnt
                iParam = paramIdx(J);
                net.params(iParam).momentum= zeros(size(net.params(iParam).value), 'single');
                if shouldAdd(net.params(iParam), 'learningRate')
                    net.params(iParam).learningRate= single(1);
                end
                if shouldAdd(net.layers(i), 'weightDecay')
                    net.params(iParam).weightDecay= single(1);
                end
            end
        end
        
        % --- This is a bit of a hack:
        % When doing the forward pass during training, we have to keep all
        % intermediate values in order to do gradient computations via
        % backprop. However, ReLU inputs can be forgotten as is already done
        % in MatConvNet when the backprop is done at the same time as the
        % forward pass, and this provides quite a lot of memory saving for
        % some networks (e.g. VGG-16). In order to achieve the desired behaviour
        % (remember everything apart from ReLU input), we will actually use
        % conserveMemory=true (i.e. forget everything), but then we explicitly
        % set precious= true for every layer apart from the one before ReLU.
        % We can further save memory by forgetting all values below backPropToLayer,
        % as they are not needed if you are doing only partial backprop
        % (this can also be done automatically with vl_simplenn but only if
        % the backward pass is done simultaneously with the forward).
        % So in the end, we mark precious only layers >=backPropToLayer-1 & !before-ReLU
        if i>=backPropToLayer-1 && (i<nLayers && ~isa(net.layers(i).block, 'dagnn.ReLU'))
            outIdx = net.layers(i).outputIndexes;
            for iVar = 1 : max(size(outIdx))
                net.vars(outIdx(iVar)).precious = true;
            end
        else
            outIdx = net.layers(i).outputIndexes;
            for iVar = 1 : max(size(outIdx))
                net.vars(outIdx(iVar)).precious = false;
            end
        end
    end
end


function should= shouldAdd(l, propName)
    if ~isa(l, 'struct')
        assert(isprop(l, propName));
        should= isempty(l.(propName));
    else
        should= ~isfield(l, propName) || isempty(l.(propName));
    end
end
