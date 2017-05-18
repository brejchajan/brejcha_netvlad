function [ output_args ] = trainOrientationSupervised( dbTrain, dbVal, varargin )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    
    opts= struct(...
        'netID', 'caffe', ...
        'layerName', 'conv5', ...
        'method', 'vlad_preL2_intra', ...
        'batchSize', 4, ...
        'learningRate', 0.0001, ...
        'lrDownFreq', 5, ...
        'lrDownFactor', 2, ...
        'weightDecay', 0.001, ...
        'momentum', 0.9, ...
        'backPropToLayer', 1, ...
        'fixLayers', [], ...
        'nNegChoice', 10, ...
        'nNegCap', 10, ...
        'nNegCache', 10, ...
        'nEpoch', 30, ...
        'margin', 0.1, ...
        'excludeVeryHard', false, ...
        'jitterFlip', false, ...
        'jitterScale', [], ...
        'sessionID', [], ...
        'outPrefix', [], ...
        'dbCheckpoint0', [], ...
        'qCheckpoint0', [], ...
        'dbCheckpoint0val', [], ...
        'qCheckpoint0val', [], ...
        'checkpoint0suffix', '', ...
        'info', '', ...
        'test0', true, ...
        'saveFrequency', 2000, ...
        'compFeatsFrequency', 1000, ...
        'computeBatchSize', 10, ...
        'epochTestFrequency', 1, ... % recommended not to be changed (pickBestNet won't work otherwise)
        'doDraw', false, ...
        'printLoss', false, ...
        'printBatchLoss', false, ...
        'nTestSample', 1000, ...
        'nTestRankSample', 5000, ...
        'recallNs', [1:5, 10:5:100], ...
        'useGPU', true, ...
        'numThreads', 12, ...
        'startEpoch', 1 ...
        );
    paths = localPaths();
    opts = vl_argparse(opts, varargin);
    
    if isempty(opts.sessionID),
        if opts.startEpoch>1, error('Have to specify sessionID to restart'); end
        rng('shuffle'); opts.sessionID= relja_randomHex(4);
    end
    if isempty(opts.fixLayers), opts.fixLayers= {}; end;
    
    %% load net and add netvlad
    net = loadNet(opts.netID, opts.layerName);
    %net = initializeCharacterCNN;
    net = addLayers(net, opts, dbTrain);
    
    %% back prop settings
    if isempty(opts.backPropToLayer)
        opts.backPropToLayer= 1;
    else
        if ~isnumeric( opts.backPropToLayer )
            assert( isstr(opts.backPropToLayer) );
            opts.backPropToLayer= relja_whichLayer(net, opts.backPropToLayer);
        end
    end
    opts.backPropToLayerName= net.layers{opts.backPropToLayer}.name;
    opts.backPropDepth= length(net.layers)-opts.backPropToLayer+1;
    assert( all(ismember(opts.fixLayers, relja_layerNames(net))) );

    display(opts);
    
    %% prepare for train and optionally move to gpu
    net = netPrepareForTrain(net, opts.backPropToLayer);
    
    if opts.useGPU
        net = relja_simplenn_move(net, 'gpu');
    end
    
    %% train
    nBatches = floor( dbTrain.numQueries / opts.batchSize );
    trainOrder= randperm(dbTrain.numQueries);
    
    losses = [];
    lr = opts.learningRate;
    
    for iBatch = 1 : nBatches
        qIDs = trainOrder((iBatch-1)*opts.batchSize + (1:opts.batchSize));
        allRes = [];
        thisBatchSize = 0;
        for iQuery = 1 : opts.batchSize
            wait(gpuDevice());
            %% get positive, and negative samples for current query
            qID = qIDs(iQuery);
            posIDs = dbTrain.nontrivialPosQ(qID);
            if isempty(posIDs), continue; end
            negIDs = dbTrain.sampleNegsQ(qID, opts.nNegChoice)
            labels = [0 ones(1, size(posIDs, 1)) -ones(1, size(negIDs, 1))];
            thisBatchSize = thisBatchSize + size(labels, 2) - 1; %TODO check if correct
            %% load images
            [ims, thisNumIms] = loadImages(qID, posIDs, negIDs, dbTrain, net, opts);

            %% forward pass
            % the memory saving related to backPropDepth is obayed 
            %implicitly due to running netPrepareForTrain before, see the 
            %comments in the function for an explanation
            gpuDevice()
            res = vl_simplenn(net, ims, [], [], 'mode', ...
                'normal', 'conserveMemory', true); 
            
            % because of the 'conserveMemory' the input is deleted, 
            % restore it if needed
            if opts.backPropToLayer==1, res(1).x= ims; end 
            ims = [];
            feats = reshape( gather(res(end).x), [], thisNumIms );

            %% loss
            x_q = feats(:, find(labels == 0));   %query feats
            x_p = feats(:, find(labels == 1));   %positive feats
            x_n = feats(:, find(labels == -1));  %negative feats
            
            neg_denom = sum(sum((repmat(x_q, [1 size(x_n, 2)]) - x_n).^2));
            E_p = single(sum(sum((repmat(x_q, [1 size(x_p, 2)]) - x_p).^2)));
            E_n = single(1.0./neg_denom);
            E = E_p + E_n;
            losses(end + 1) = E;
            %% backpropagation
            neg_denom_sq = neg_denom.^2;
            dEdx_q = 2.*sum(repmat(x_q, [1 size(x_p, 2)]) - x_p, 2) + ...
                2.*sum(repmat(x_q, [1 size(x_n, 2)]) - x_n, 2) ./ neg_denom_sq;
            
            dEdx_p = -2 .* repmat(x_q, [1 size(x_p, 2)]) - x_p;
            dEdx_n = -2 .* repmat(x_q, [1 size(x_n, 2)]) - x_n ./ neg_denom_sq;
            
            dEdx = [dEdx_q dEdx_p dEdx_n];

            if opts.useGPU
                dEdx = gpuArray(dEdx);
            end
            allRes = [allRes ; vl_simplenn(net, ims, dEdx, res, ...
                        'mode', 'normal', ...
                        'skipForward', true, ...
                        'backPropDepth', opts.backPropDepth, ...
                        'conserveMemory', true)];
            clear dEdx_q;
            clear dEdx_p;
            clear dEdx_n;
            clear E_p;
            clear E_n; 
            clear E;
            clear x_p;
            clear x_q;
            clear x_n;
            clear negDenom;
            clear labels;
            clear res;
            clear feats;
            clear dEdx;
           
        end %iQuery
        
        %% update net
        %updateNet(net, opts, lr, thisBatchSize, allRes);
        if size(allRes, 1) > 0
            for l= 1:numel(net.layers)
                for j= 1:numel(allRes(1, l).dzdw)
                    if ismember(net.layers{l}.name, opts.fixLayers) continue; end

                    dzdw= allRes(1, l).dzdw{j};
                    for iQuery= 2:size(allRes,1)
                        dzdw= dzdw + allRes(iQuery, l).dzdw{j};
                    end

                    thisDecay= opts.weightDecay * net.layers{l}.weightDecay(j);
                    thisLR= lr * net.layers{l}.learningRate(j);

                    net.layers{l}.momentum{j}= ...
                        opts.momentum * net.layers{l}.momentum{j} ...
                        - thisDecay * net.layers{l}.weights{j} ...
                        - (1 / thisBatchSize) * dzdw;
                    net.layers{l}.weights{j}= net.layers{l}.weights{j} + thisLR * net.layers{l}.momentum{j};
                    clear dzdw;
                end
            end
        end
        clear allRes;
        
        %{
        if opts.useGPU    
            net = relja_simplenn_move(net, 'cpu');
            save net.mat net losses trainOrder opts dbTrain dbVal nBatches
            clear all;
            gpu = gpuDevice(1);
            load('net.mat', 'net', 'losses', 'trainOrder', 'opts', 'dbTrain', 'dbVal', 'nBatches');
            net = relja_simplenn_move(net, 'gpu');
            wait(gpu);
        end
        %}
    
        plotAll(losses);
    end
end

function plotAll(losses)
    plot(losses);
    drawnow update
end

function [net] = updateNet(net, opts, lr, thisBatchSize, allRes)
    
end

function [ims, num] = loadImages(qID, posIDs, negIDs, dbTrain, net, opts)
    imageFns= [ [dbTrain.qPath, dbTrain.qImageFns{qID}]; ...
        strcat( dbTrain.dbPath, dbTrain.dbImageFns([posIDs; negIDs]) ) ];
    num = length(imageFns);
    
    ims_= vl_imreadjpeg(imageFns, 'numThreads', opts.numThreads);
    ims = cat(4, ims_{:});

    ims(:,:,1,:)= ims(:,:,1,:) - net.meta.normalization.averageImage(1,1,1);
    ims(:,:,2,:)= ims(:,:,2,:) - net.meta.normalization.averageImage(1,1,2);
    ims(:,:,3,:)= ims(:,:,3,:) - net.meta.normalization.averageImage(1,1,3);

    if opts.useGPU
        ims= gpuArray(ims);
    end
end

