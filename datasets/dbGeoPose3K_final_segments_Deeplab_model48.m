classdef dbGeoPose3K_final_segments_Deeplab_model48 < dbGeoPose3K_final_segments
    
    methods
        
        function db = dbGeoPose3K_final_segments_Deeplab_model48(set)
            db = db@dbGeoPose3K_final_segments(set);
            assert( ismember(set, {'train', 'val', 'test'}) );
            
            db.name = sprintf('geoPose3K_final_segments_pano_small_Deeplab_model48_%s', set);
            paths= localPaths();
            dbRoot= paths.dsetRootGP3KPano;
            db.dbPath= [dbRoot 'db/'];
            db.qPath= [dbRoot 'query_Deeplab_model48/segments/color/'];
            
            db.dbLoad();
        end
    end
end
