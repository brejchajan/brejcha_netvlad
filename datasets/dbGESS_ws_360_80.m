classdef dbGESS_ws_360_80 < dbGESS
    
    methods
    
        function db = dbGESS_ws_360_80(whichSet)
            % whichSet is one of: train, val, test
            
            db = db@dbGESS(whichSet);
            
            assert( ismember(whichSet, {'train', 'val', 'test'}) );
            
            db.name= sprintf('GESS_ws_360_80_%s', whichSet);
            
            paths= localPaths();
            dbRoot= paths.dsetRootGESS_ws_360_80;
            db.dbPath= [dbRoot, 'db/']
            db.qPath= [dbRoot, 'query/']
            
            db.dbLoad();
        end
        
    end
    
end