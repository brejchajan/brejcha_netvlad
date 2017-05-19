function [ output_args ] = trainOrientationSupervised( dbTrain, dbVal, varargin )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    
    opts= struct(...
        'netID', 'caffe', ...
        'layerName', 'conv5', ...
        'method', 'vlad_preL2_intra', ...
        'batchSize', 4, ...
        'learningRate', 0.0001, ...
        'lrDownFreq', 30, ...
        'lrDownFactor', 2, ...
        'weightDecay', 0.001, ...
        'momentum', 0.9, ...
        'backPropToLayer', 1, ...
        'fixLayers', [], ...
        'nNegChoice', 10, ...
        'nPosChoice', 2, ...
        'nNegCap', 10, ...
        'nNegCache', 10, ...
        'nEpoch', 10, ...
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
        'startEpoch', 1, ...
        'showTrainingImgs', true ...
        );
    paths = localPaths();
    opts = vl_argparse(opts, varargin);
        
    if isempty(opts.sessionID),
        if opts.startEpoch>1, error('Have to specify sessionID to restart'); end
        rng('shuffle'); opts.sessionID= relja_randomHex(4);
    end
    if isempty(opts.fixLayers), opts.fixLayers= {}; end;
    
    %% initial checkpoints
    if opts.startEpoch<2
        
        % ----- Checkpoint names
        
        if ~isempty(opts.checkpoint0suffix)
            opts.checkpoint0suffix= [opts.checkpoint0suffix, '_'];
        end
        if isempty(opts.dbCheckpoint0)
            opts.dbCheckpoint0= sprintf('%s%s_%s_%s_%s_%sdb.bin', opts.outPrefix, dbTrain.name, opts.netID, opts.layerName, opts.method, opts.checkpoint0suffix);
        end
        if isempty(opts.qCheckpoint0)
            opts.qCheckpoint0= sprintf('%s%s_%s_%s_%s_%sq.bin', opts.outPrefix, dbTrain.name, opts.netID, opts.layerName, opts.method, opts.checkpoint0suffix);
        end
        if isempty(opts.dbCheckpoint0val)
            opts.dbCheckpoint0val= sprintf('%s%s_%s_%s_%s_%sdb.bin', opts.outPrefix, dbVal.name, opts.netID, opts.layerName, opts.method, opts.checkpoint0suffix);
        end
        if isempty(opts.qCheckpoint0val)
            opts.qCheckpoint0val= sprintf('%s%s_%s_%s_%s_%sq.bin', opts.outPrefix, dbVal.name, opts.netID, opts.layerName, opts.method, opts.checkpoint0suffix);
        end
        
        % ----- Network setup
        
        net= loadNet(opts.netID, opts.layerName);
        
        % --- Add my layers
        net= addLayers(net, opts, dbTrain);
        
        % --- BackProp depth
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
       
        % ----- Init
        [obj, auxData] = initObj(dbTrain);
        
    else
        % ----- Continue from an epoch
        ID= sprintf('ep%06d_latest', opts.startEpoch-1);
        outFnCurrent= sprintf('%s%s_%s.mat', opts.outPrefix, opts.sessionID, ID);
        tmpopts= opts;
        load(outFnCurrent, 'net', 'obj', 'opts', 'auxData'); % rewrites opts
        clear ID outFnCurrent;
        
        opts.startEpoch= tmpopts.startEpoch;
        opts.nEpoch= tmpopts.nEpoch;
        opts.test0= false;
        opts.useGPU= tmpopts.useGPU;
        opts.numThreads= tmpopts.numThreads;
        opts.batchSize= tmpopts.batchSize;
        opts.computeBatchSize= tmpopts.computeBatchSize;
        
        if ~isfield(opts, 'dbCheckpoint0_orig')
            opts.dbCheckpoint0_orig= opts.dbCheckpoint0;
            opts.qCheckpoint0_orig= opts.qCheckpoint0;
        end
        opts.dbCheckpoint0= tmpopts.dbCheckpoint0;
        opts.qCheckpoint0= tmpopts.qCheckpoint0;
        
        if isempty(opts.qCheckpoint0)
            opts.dbCheckpoint0= sprintf('%s%s_%s_%s_%s_%s%s_ep%06d_db.bin', opts.outPrefix, dbTrain.name, opts.netID, opts.layerName, opts.method, opts.checkpoint0suffix, opts.sessionID, opts.startEpoch-1);
        end
        if isempty(opts.qCheckpoint0)
            opts.qCheckpoint0= sprintf('%s%s_%s_%s_%s_%s%s_ep%06d_q.bin', opts.outPrefix, dbTrain.name, opts.netID, opts.layerName, opts.method, opts.checkpoint0suffix, opts.sessionID, opts.startEpoch-1);
        end
        
        display(opts);
   end
    
    %% prepare for train and optionally move to gpu
    net = netPrepareForTrain(net, opts.backPropToLayer);
    
    if opts.useGPU
        net = relja_simplenn_move(net, 'gpu');
    end
    
    if ~isfield(net.meta.normalization, 'currentdataset')
        net.meta.normalization.currentdataset = struct('averageImage', []);
        net.meta.normalization.currentdataset.averageImage = averageImage(dbTrain);
    end
    
    
    %{
    if ~exist(opts.qCheckpoint0, 'file')
        serialAllFeats(net, dbTrain.qPath, dbTrain.qImageFns, ...
            opts.qCheckpoint0, 'useGPU', opts.useGPU, 'batchSize', opts.computeBatchSize);
    end
    if ~exist(opts.dbCheckpoint0, 'file')
        serialAllFeats(net, dbTrain.dbPath, dbTrain.dbImageFns, ...
            opts.dbCheckpoint0, 'useGPU', opts.useGPU, 'batchSize', opts.computeBatchSize);
    end
    if opts.test0
        if ~exist(opts.qCheckpoint0val, 'file')
            serialAllFeats(net, dbVal.qPath, dbVal.qImageFns, ...
                opts.qCheckpoint0val, 'useGPU', opts.useGPU, 'batchSize', opts.computeBatchSize);
        end
        if ~exist(opts.dbCheckpoint0val, 'file')
            serialAllFeats(net, dbVal.dbPath, dbVal.dbImageFns, ...
                opts.dbCheckpoint0val, 'useGPU', opts.useGPU, 'batchSize', opts.computeBatchSize);
        end
        
        [obj.pretrain.val.recall, obj.pretrain.val.rankloss]= ...
            testFromFn(dbVal, opts.dbCheckpoint0val, opts.qCheckpoint0val, opts);
        %{
        [obj.pretrain.train.recall, obj.pretrain.train.rankloss]= ...
            testFromFn(dbTrain, opts.dbCheckpoint0, opts.qCheckpoint0, opts);
        %}
        
    end
    %}
    %% train
    nBatches = floor( dbTrain.numQueries / opts.batchSize );
    trainOrder= randperm(dbTrain.numQueries);
    
    losses = [];
    lr = opts.learningRate;
    figprev = figure('Name', 'Training images');
    hfigbatchloss = figure('Name', 'Loss per batch');
    figbatchloss = semilogy(1);
    
    for iEpoch = 1 : opts.nEpoch
        ID = sprintf('ep%06d_latest', iEpoch);
        trainID = sprintf('%s_train', ID);
        valID = sprintf('%s_val', ID);
        for iBatch = 1 : nBatches
            qIDs = trainOrder((iBatch-1)*opts.batchSize + (1:opts.batchSize));
            allRes = [];
            thisBatchSize = 0;
            for iQuery = 1 : opts.batchSize
                %% get positive, and negative samples for current query
                qID = qIDs(iQuery);
                posIDs = dbTrain.nontrivialPosQ(qID);
                if isempty(posIDs), continue; end
                thisNumPos = size(posIDs, 1);
                
                %select at max nPosChoice of positive samples
                posIDs = posIDs(randsample(thisNumPos, ...
                                min(thisNumPos, opts.nPosChoice))); 
                
                negIDs = dbTrain.sampleNegsQ(qID, opts.nNegChoice);
                labels = [0 ones(1, size(posIDs, 1)) -ones(1, size(negIDs, 1))];
                thisBatchSize = thisBatchSize + size(labels, 2) - 1; %TODO check if correct
                %% load images
                [ims, thisNumIms] = loadImages(qID, posIDs, negIDs, dbTrain, net, opts);
                
                %% preview images in this batch
                if opts.showTrainingImgs
                    posCount = size(posIDs, 1);
                    negCount = size(negIDs, 1);
                    figure(figprev)
                    prevsize = max(posCount, negCount);
                    subplot(3, prevsize, 1)
                    imshow(ims(:,:,:,1)/255); %query
                    for i = 1 : posCount
                        subplot(3, prevsize, i+prevsize)
                        imshow(ims(:,:,:,1+i)/255);
                    end
                    for i = 1 : negCount
                        subplot(3, prevsize, i+2*prevsize)
                        imshow(ims(:,:,:,1+posCount+i)/255);
                    end
                end
                
                
                %% forward pass
                % the memory saving related to backPropDepth is obayed 
                %implicitly due to running netPrepareForTrain before, see the 
                %comments in the function for an explanation
                res = vl_simplenn(net, ims, [], [], 'mode', ...
                    'normal', 'conserveMemory', true, 'CuDNN', true); 

                % because of the 'conserveMemory' the input is deleted, 
                % restore it if needed
                if opts.backPropToLayer==1, res(1).x= ims; end 
                ims = [];
                feats = reshape( gather(res(end).x), [], thisNumIms );

                %% my loss
                %{
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
                %}
                
                
                %% triplet loss
                
                x_q = feats(:, find(labels == 0));   %query feats
                x_p = feats(:, find(labels == 1));   %positive feats
                x_n = feats(:, find(labels == -1));  %negative feats

                E_p = single(sum((repmat(x_q, [1 size(x_p, 2)]) - x_p).^2, 2));
                E_n = single(sum((repmat(x_q, [1 size(x_n, 2)]) - x_n).^2, 2));
           
                E_term = E_p - E_n;
                E = sum(max(E_term, 0));
                
                losses(end + 1) = E;
                
                %% triplet loss backpropagation
                dEdx_q = sum(2 .* (repmat(x_q, [1 size(x_p, 2)]) - x_p), 2);
                dEdx_p = -2 .* (repmat(x_q, [1 size(x_p, 2)]) - x_p);
                dEdx_n = 2 .* (repmat(x_q, [1 size(x_n, 2)]) - x_n);
                
                dEdx_q(find(E_term < 0), :) = 0;
                dEdx_p(find(E_term < 0), :) = 0;
                dEdx_n(find(E_term < 0), :) = 0;

                dEdx = [dEdx_q dEdx_p dEdx_n];
               
                if opts.useGPU
                    dEdx = gpuArray(dEdx);
                end
                allRes = [allRes ; vl_simplenn(net, ims, dEdx, res, ...
                            'mode', 'normal', ...
                            'skipForward', true, ...
                            'backPropDepth', opts.backPropDepth, ...
                            'conserveMemory', true, 'CuDNN', true)];
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
            if opts.useGPU
                gpu = gpuDevice();
            end
            updateNet(net, opts, lr, thisBatchSize, allRes);
            clear allRes;
            if opts.useGPU
                wait(gpu);
            end
            plotLoss(losses, figbatchloss);
            if size(losses, 1) > 0
                relja_display('Batch: %f, loss: %f.', iBatch, losses(end));
            end 
        end
        %% update lr
        if mod(iEpoch, opts.lrDownFreq) == 0
            newlr = lr / opts.lrDownFactor;
            relja_display('Changing Learning rate from: %f to %f.', lr, newlr);
            lr = newlr;
            clear newlr;
        end
        
        %% test
        testNow = iEpoch==opts.nEpoch || rem(iEpoch, opts.epochTestFrequency) == 0;
        if testNow
            test(dbTrain, dbVal, net, opts, obj, auxData, iEpoch,...
                    trainID, valID, ID);
        end
        relja_display('Epoch: %f.', iEpoch);
        %save net.mat net
    end
end

function [obj, auxData] = initObj(dbTrain)
    obj= struct();
    obj.train= struct('loss', [], 'recall', [], 'rankloss', []);
    obj.val= struct('loss', [], 'recall', [], 'rankloss', []);
    
    auxData= {};
    auxData.epochStartTime= {};
    auxData.numTrain= dbTrain.numQueries;
end

function test(dbTrain, dbVal, net, opts, obj, auxData, iEpoch,...
                 trainID, valID, ID)
    
    [qFeatVal, dbFeatVal] = computeAllFeats(dbVal, net, opts, valID, true);
    [obj.val.recall(:, end+1), obj.val.rankloss(:, end+1)] = testNet(dbVal, net, opts, valID, qFeatVal, dbFeatVal);
    clear qFeatVal dbFeatVal;


    [qFeat, dbFeat]= computeAllFeats(dbTrain, net, opts, trainID, true);


    %[obj.train.recall(:, end+1), obj.train.rankloss(:, end+1) ...
    %    ]= testNet(dbTrain, net, opts, trainID, qFeat, dbFeat);

    % to save the results
    saveNet(net, obj, opts, auxData, ID, sprintf('epoch-end %d', iEpoch));

    if opts.doDraw, plotResults(obj, opts, auxData); end

end

function plotLoss(losses, figloss)
    %semilogy(losses);
    set(figloss, 'ydata', losses);
    drawnow
end

function [net] = updateNet(net, opts, lr, thisBatchSize, allRes)
    if size(allRes, 1) > 0
        for l= 1:numel(net.layers)
            for j= 1:numel(allRes(1, l).dzdw)
                if ismember(net.layers{l}.name, opts.fixLayers) 
                    continue; 
                end

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
end

function [ims, num] = loadImages(qID, posIDs, negIDs, dbTrain, net, opts)
    imageFns= [ [dbTrain.qPath, dbTrain.qImageFns{qID}]; ...
        strcat( dbTrain.dbPath, dbTrain.dbImageFns([posIDs; negIDs]) ) ];
    num = length(imageFns);
    
    ims_= vl_imreadjpeg(imageFns, 'numThreads', opts.numThreads);
    ims = cat(4, ims_{:});

    %mr=0;
    %mg=0;
    %mb=0;
    
    mr = net.meta.normalization.currentdataset.averageImage(:,:,1);
    mg = net.meta.normalization.currentdataset.averageImage(:,:,2);
    mb = net.meta.normalization.currentdataset.averageImage(:,:,3);
    
    ims(:,:,1,:)= ims(:,:,1,:) - median(mr(:));
    ims(:,:,2,:)= ims(:,:,2,:) - median(mg(:));
    ims(:,:,3,:)= ims(:,:,3,:) - median(mb(:));

    if opts.useGPU
        ims= gpuArray(ims);
    end
end


function [qFeat, dbFeat]= computeAllFeats(db, net, opts, ID, delFile)
    if nargin<5, delFile= true; end
    outPrefix= sprintf('%s%s_%s', opts.outPrefix, opts.sessionID, ID);

    qFeatFn= sprintf('%s_q.bin', outPrefix);
    tmpFn= sprintf('%s.tmp', qFeatFn);
    serialAllFeats(net, db.qPath, db.qImageFns, ...
        tmpFn, 'useGPU', opts.useGPU, 'batchSize', opts.computeBatchSize);
    movefile(tmpFn, qFeatFn);

    qFeat= fread( fopen(qFeatFn, 'rb'), inf, 'float32=>single');
    qFeat= reshape(qFeat, [], db.numQueries);
    if delFile, delete(qFeatFn); end

    dbFeatFn= sprintf('%s_db.bin', outPrefix);
    tmpFn= sprintf('%s.tmp', dbFeatFn);
    serialAllFeats(net, db.dbPath, db.dbImageFns, ...
        tmpFn, 'useGPU', opts.useGPU, 'batchSize', opts.computeBatchSize);
    movefile(tmpFn, dbFeatFn);

    dbFeat= fread( fopen(dbFeatFn, 'rb'), [size(qFeat,1), db.numImages], 'float32=>single');
    if delFile, delete(dbFeatFn); end
end


