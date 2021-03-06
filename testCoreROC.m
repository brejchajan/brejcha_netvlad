function [fraction, Ns]= testCoreROC(db, qFeat, dbFeat, varargin)
    opts= struct(...
        'nTestSample', inf, ...
        'recallNs', [1:5, 10:5:100], ...
        'printN', 10 ...
        );
    opts= vl_argparse(opts, varargin);
    
    searcherRAW_= @(iQuery, nTop) rawNnSearch(qFeat(:,iQuery), dbFeat, nTop);
    if ismethod(db, 'nnSearchPostprocess')
        searcherRAW= @(iQuery, nTop) db.nnSearchPostprocess(searcherRAW_, iQuery, nTop);
    else
        searcherRAW= searcherRAW_;
    end
    [fraction, Ns]= localizedForTopN( searcherRAW, db.numQueries,  @(iQuery, iDb) db.isPosQ(iQuery, iDb), max(opts.recallNs));
    
end
