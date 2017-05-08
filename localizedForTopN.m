function [fraction, Ns]= localizedForTopN(searcher, nQueries, isPos, nTop)
    toTest= 1:nQueries;
    
    
    recalls= zeros(length(toTest), 1);
    printRecalls= zeros(length(toTest),1);
    
    evalProg= tic;
    
    for iTestSample= 1:length(toTest)
        
        relja_progress(iTestSample, ...
                       length(toTest), ...
                       sprintf('%.4f', mean(printRecalls(1:(iTestSample-1)))), evalProg);
        
        iTest= toTest(iTestSample);
        
        ids= searcher(iTest, nTop);
        numReturned= length(ids);
        assert(numReturned<=nTop); % if your searcher returns fewer, it's your fault
        candidates = find(isPos(iTest, ids) > 0);
        candidate = -1;
        if (size(candidates >= 1))
            candidate = candidates(1);
        end
        recalls(iTestSample)= candidate;
    end
    Ns = [1 : nTop];
    fraction = zeros(nTop, 1);
    for i = 1 : nTop
        fraction(i) = size(find(recalls == i), 1);
    end
    fraction = cumsum(fraction);
    fraction = fraction ./ nQueries;
    
end
