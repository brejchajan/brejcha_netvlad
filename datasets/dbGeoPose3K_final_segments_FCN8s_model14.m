classdef dbGeoPose3K_final_segments_FCN8s_model14 < dbGeoPose3K_final_segments
        
    methods
    
        function db = dbGeoPose3K_final_segments_FCN8s_model14(set)
            db = db@dbGeoPose3K_final_segments(set);
            assert( ismember(set, {'train', 'val', 'test'}) );
            
            db.name = sprintf('geoPose3K_final_segments_pano_small_FCN8s_model14_%s', set);
            paths= localPaths();
            dbRoot= paths.dsetRootGP3KPano;
            db.dbPath= [dbRoot 'db/'];
            db.qPath= [dbRoot 'query_FCN8s/segments/color/'];
            
            db.dbLoad();
        end
        
    end
    
end
