function [avgimg] = averageImage(dbTrain, batchsize)
    if (nargin < 2)
        batchsize = 100;
    end
    
    imageFns= [ strcat( dbTrain.qPath, dbTrain.qImageFns(:)); ...
        strcat( dbTrain.dbPath, dbTrain.dbImageFns(:) ) ];
    
    avgimg = vl_imreadjpeg(imageFns(1)); avgimg = avgimg{1};
    avgimg = zeros(size(avgimg));
    numImgs = size(imageFns, 1);
    if numImgs < 2
        return
    end
    
    for i = 1 : batchsize : numImgs - batchsize
        ims_ = vl_imreadjpeg(imageFns(i:i+batchsize)); 
        % fix non-colour images
        for iIm= 1:batchsize
            if size(ims_{iIm},3)==1
                ims_{iIm}= cat(3,ims_{iIm},ims_{iIm},ims_{iIm});
            end
        end         
        ims = cat(4, ims_{:});
        avgimg = avgimg * ((i-batchsize)/i) + sum(ims, 4) / i;
    end
end
