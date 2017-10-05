function net= addLayersDag(net, opts, dbTrain, layerName)
    
    
    
    methodOpts= strsplit(opts.method, '_');
    [~, sz]= brejcha_netOutputDimDag(net, {layerName});
    D= sz(3);
    
    if ismember('preL2', methodOpts)
        % normalize feature-wise
        %output from specified layer is input to the new layer
        inputs = brejcha_getOutputNamesDag(net, layerName); 
        preL2 = dagnn.LRN();
        preL2.param = [2*D, 1e-12, 1, 0.5];
        net.addLayer('preL2', preL2, inputs, 'prel2normalized', {});
        methodOpts = removeOpt(methodOpts, 'preL2');
        doPreL2 = true;
    else
        doPreL2 = false;
    end
    
    
    
    if doPreL2
        pool_inputs = 'prel2normalized';
    else
        pool_inputs = brejcha_getOutputNamesDag(net, layerName); 
    end
    
    if ismember('max', methodOpts)
        method_used = 'totMaxPooled';
        methodOpts = removeOpt(methodOpts, 'max');
        %net.layers{end+1}= layerTotalMaxPool('max:core');
        totMaxPool = layerTotalMaxPool('max:core');
        net.addLayer('totMaxPool', totMaxPool, pool_inputs, ...
            method_used, {});
        
        
        
    elseif ismember('avg', methodOpts)
        method_used = 'totAvgPooled';
        methodOpts = removeOpt(methodOpts, 'avg');
        totAvgPool = layerTotalAvgPool('avg:core');
        net.addLayer('totAvgPool', totAvgPool, pool_inputs, ...
            method_used, {});
        
        
        
    elseif any( ismember( {'vlad', 'vladv2'}, methodOpts) )
        method_used = 'vlad_core';
        if doPreL2
            L2str= '_preL2';
        else
            L2str= '';
        end
        
        whichDesc= sprintf('%s_%s%s', opts.netID, opts.layerName, L2str);
        
        k= 64;
        paths= localPaths();
        trainDescFn= sprintf('%s%s_%s_traindescs.mat', paths.initData, dbTrain.name, whichDesc);
        clstFn= sprintf('%s%s_%s_k%03d_clst.mat', paths.initData, dbTrain.name, whichDesc, k);
        
        if doPreL2
            clustLayerName = 'preL2';
        else
            clustLayerName = layerName;
        end
        clsts= getClustersDag(net, opts, clstFn, k, dbTrain, trainDescFn, clustLayerName);
        
        load( trainDescFn, 'trainDescs');
        load( clstFn, 'clsts');
        net.meta.sessionID= sprintf('%s_%s', net.meta.sessionID, dbTrain.name);
        
        
        
        % --- VLAD layer
        
        if ismember('vladv2', methodOpts)
            methodOpts= removeOpt(methodOpts, 'vladv2');
            
            % set alpha for sparsity
            [~, dsSq]= yael_nn(clsts, trainDescs, 2); clear trainDescs;
            alpha= -log(0.01)/mean( dsSq(2,:)-dsSq(1,:) ); clear dsSq;
            
            vladv2= layerVLADv2('vlad:core');
            vladv2.constructor({alpha*2*clsts, -alpha*sum(clsts.^2,1), -clsts});
            net.addLayer('vladv2', vladv2, pool_inputs, method_used, ...
                {});
            %net.layers{end}= net.layers{end}.constructor({alpha*2*clsts, -alpha*sum(clsts.^2,1), -clsts});
            
        elseif ismember('vlad', methodOpts)
            % see comments on vladv2 vs vlad in the README_more.md
            
            methodOpts= removeOpt(methodOpts, 'vlad');
            
            % set alpha for sparsity
            clstsAssign= relja_l2normalize_col(clsts);
            dots= sort(clstsAssign'*trainDescs, 1, 'descend'); clear trainDescs;
            alpha= -log(0.01)/mean( dots(1,:) - dots(2,:) ); clear dots;
            
            D = size(clstsAssign, 1);
            K = size(clstsAssign, 2);
            vlad= layerVLADDag('D', D, 'K', K) %.constructor({alpha*clstsAssign, clsts});
            net.addLayer('vlad', vlad, pool_inputs, method_used, {'vlad_w1', 'vlad_w2'});
            w1idx = net.getParamIndex('vlad_w1');
            w2idx = net.getParamIndex('vlad_w2');
            net.params(w1idx).value = reshape(alpha*clstsAssign, [1,1,D,K]);
            net.params(w2idx).value = reshape(-clsts, [1,1,D,K]);
            
        else
            error('Unsupported method "%s"', opts.method);
        end
        
        if ismember('intra', methodOpts)
            % --- intra-normalization
            method_used = 'vlad_intranormed'
            intraL2 = dagnn.LRN();
            intraL2.param = [2*D, 1e-12, 1, 0.5];
            net.addLayer('vlad_intranorm', intraL2, 'vlad_core', method_used, {});
            methodOpts= removeOpt(methodOpts, 'intra');
        end
        
    else
        error('Unsupported method "%s"', opts.method);
    end
    
    
    
    % --- final normalization
    finalL2= layerWholeL2Normalize('postL2');
    net.addLayer('postL2', finalL2, method_used, 'postl2normalized', {});
    
    
    % --- check if all options are used
    if ~isempty(methodOpts)
        error('Unsupported options (method=%s): %s', opts.method, strjoin(methodOpts, ', '));
    end
    
    net.meta.sessionID= sprintf('%s_%s', net.meta.sessionID, opts.method);
    net.meta.epoch= 0;
    
end



function opts= removeOpt(opts, optName)
    opts(ismember(opts, optName))= [];
end
