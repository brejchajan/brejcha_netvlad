classdef dbGP3K_old_segments < dbBase
    
    methods
    
        function db = dbGP3K_old_segments(whichSet)
            % whichSet is one of: train, val, test
            
            assert( ismember(whichSet, {'train', 'val', 'test'}) );
            
            db.name= sprintf('GP3K_old_segments_%s', whichSet);
            
            paths= localPaths();
            dbRoot= paths.dsetRootGP3K_old_segments;
            db.dbPath= [dbRoot, 'db/']
            db.qPath= [dbRoot, 'query/']
            
            db.dbLoad();
        end
        
        function posIDs= nontrivialPosQ(db, iQuery)
            [posIDs, dSq]= db.cp.getPosIDs(db.utmQ(:,iQuery));
            posIDs= posIDs(dSq>=0 & dSq<=db.nonTrivPosDistSqThr );
        end
        
        function posIDs= nontrivialPosDb(db, iDb)
            [posIDs, dSq]= db.cp.getPosDbIDs(iDb);
            posIDs= posIDs(dSq>=0 & dSq<=db.nonTrivPosDistSqThr );
        end
        
    end
    
end