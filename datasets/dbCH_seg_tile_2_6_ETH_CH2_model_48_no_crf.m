classdef dbCH_seg_tile_2_6_ETH_CH2_model_48_no_crf < dbBase
    
    methods
    
        function db = dbCH_seg_tile_2_6_ETH_CH2_model_48_no_crf()
            db.name= sprintf('CH_seg_tile_5_6_subarea_46.6714_7.78997_46.5228_8.14857_ETH_CH2_model_48_no_crf_test');
            paths= localPaths();
            dbRoot= paths.dsetRootCH_seg;
            db.dbPath= dbRoot
            db.qPath= dbRoot
            
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