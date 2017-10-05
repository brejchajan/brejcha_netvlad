classdef layerVLADDag < dagnn.Layer
    
    properties
        D
        K
        %this is now in params
        %weights = params{1}
            %momentum
            %learningRate
            %weightDecay
            %K
            %D
            %vladDim
            %precious= false
    end
    
    methods
        
        function obj= layerVLADDag(varargin)
            obj.load(varargin{:});
        end
        
        function params = initParams(obj)
            params{1} = reshape(randn(obj.D, obj.K), [1,1,D,K]);
            params{2} = reshape(-randn(obj.D, obj.K), [1,1,D,K]);                         
        end
        
        
        function outputs= forward(l, x, weights)
            x = x{1};
            batchSize= size(x, 4);
            
            % --- assign
            
            assgn= vl_nnsoftmax( vl_nnconv(x, weights{1}, []) );
            
            % --- aggregate
            
            if isa(x, 'gpuArray')
                y= zeros([1, l.K, l.D, batchSize], 'single', 'gpuArray');
            else
                y= zeros([1, l.K, l.D, batchSize], 'single');
            end
            
            for iK= 1:l.K
                % --- sum over descriptors: assignment_iK * (descs - offset_iK)
                y(:,iK,:,:)= ...
                    sum(sum( ...,
                        repmat( assgn(:,:,iK,:), [1,1,l.D,1] ) .* ...
                        vl_nnconv(x, [], weights{2}(1,1,:,iK)), ...
                        1), 2);
            end
            
            outputs = {y};
            % --- normalizations (intra-normalization, L2 normalization)
            % performed outside as separate layers
            
        end
        
        function [dzdx, dzdw]= backward(l, x, weights, dzdy)
            x = x{1};
            dzdy = dzdy{1};
            batchSize= size(x, 4);
            H= size(x, 1);
            W= size(x, 2);
            % assert(l.D==size(x, 3));
            
            % TODO: stupid to run forward again? remember results?
            
            % --- assign
            
            p= vl_nnconv(x, weights{1}, []);
            assgn= vl_nnsoftmax(p);
            
            % --- dz/da (soft assignment)
            
            dzda= assgn; % just for the shape/class
            
            for iK= 1:l.K
                dzda(:,:,iK,:)= sum( ...
                        bsxfun(@times, ...
                            dzdy(:,iK,:,:), ...
                            vl_nnconv(x, [], weights{2}(1,1,:,iK))), ...
                        3);
            end
            
            % --- dz/dp (product of descriptors and clusters)
            
            dzdp= vl_nnsoftmax(p, dzda); clear dzda p;
            
            % --- dz/dw1 (assignment clusters) and dz/dx (via assignment)
            
            [dzdx, dzdw{1}]= vl_nnconv(x, weights{1}, [], dzdp); clear dzdp;
            
            % --- dz/dx (via aggregation)
            % --- and add to current dz/dx to get the full thing
            
            dzdy= reshape(dzdy, [l.K, l.D, batchSize]);
            
            assgn_= reshape(assgn, [H*W, l.K, batchSize]);
            for iB= 1:batchSize
                dzdx(:,:,:,iB)= dzdx(:,:,:,iB) + reshape( ...
                    assgn_(:,:,iB) * dzdy(:,:,iB), ...
                    [H, W, l.D]);
            end
            clear assgn_;
            
            % --- dz/dw2 (offset)
            
            dzdw{2}= reshape( sum( ...
                dzdy .* ...
                repmat( ...
                    reshape( sum(sum(assgn,1),2), [l.K, 1, batchSize] ), ...
                    [1, l.D, 1] ), ...
                3 )', [1, 1, l.D, l.K] );
            dzdx = {dzdx};
            
        end
        
        function objStruct= saveobj(obj)
            objStruct= relja_saveobj(obj);
        end
        
    end
    
    methods (Static)
        
        function l= loadobj(objStruct)
            l= layerVLAD();
            l= relja_loadobj(l, objStruct);
        end
    %}
    end
    
end
