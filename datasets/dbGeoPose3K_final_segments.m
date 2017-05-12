classdef dbGeoPose3K_final_segments < dbBase
    
    properties
        dbCamParams
        qCamParams
    end
    
    methods
    
        function db = dbGeoPose3K_final_segments(set)
            
            assert( ismember(set, {'train', 'val', 'test'}) );
            
            %db.name= sprintf('geoPose3K_final_segments_pano_small_oldgp3kquery_%s', set);
            db.name = sprintf('geoPose3K_final_segments_pano_small_FCN8s_model14_%s', set);
            %db.name = sprintf('geoPose3K_final_segments_pano_small_Deeplab_model48_%s', set);
            paths= localPaths();
            dbRoot= paths.dsetRootGP3KPano;
            db.dbPath= [dbRoot 'db/'];
            db.qPath= [dbRoot 'query_FCN8s/segments/color'];
            %db.qPath= [dbRoot 'query_Deeplab_model48/segments/color'];
            
            db.dbLoad();
        end
        
        function showNontrivialPosQ(db, iQuery)
            %search for non-trivial positives
            posIDs = db.nontrivialPosQ(iQuery);
            
            if (size(posIDs, 1) < 1)
                warning('The database does not contain positive image for this query.');
            else
                % plot query
                pos_size = size(posIDs, 1);
                figure
                h_q = subplot(2, pos_size, 1);
                q_img_name = fullfile(db.qPath, db.qImageFns{iQuery})
                query_img = imread(q_img_name);
                imshow(query_img);
                % plot db
                h_db = [];
                for i = 1 : size(posIDs, 1)
                    h = subplot(2, pos_size, pos_size + i);
                    h_db = [h_db h];
                    db_img = imread(fullfile(db.dbPath, db.dbImageFns{posIDs(i)}));
                    imshow(db_img);
                end
                linkaxes([h_db h_q]);
            end
        end
        
        function angle = inZeroTwoPi(db, phi)
            angle = phi;
            while (angle < 0)
                angle = angle + 2*pi;
            end
            while (angle > 2*pi)
                angle = angle - 2*pi;
            end
        end
        
        function posIDs= nontrivialPosQ(db, iQuery)
            [posIDs, dSq]= db.cp.getPosIDs(db.utmQ(:,iQuery));
            posIDs= posIDs(dSq>=0 & dSq<=db.nonTrivPosDistSqThr );
            q_yaw = db.inZeroTwoPi(db.qCamParams(iQuery, 1));
            q_fov = db.qCamParams(iQuery, 4);
            pos_db_yaws = db.dbCamParams(posIDs, 1);
            yaw_delta = abs(pos_db_yaws - q_yaw);
            posIDs = posIDs(yaw_delta < (q_fov/2));
        end
        
        function posIDs= nontrivialPosDb(db, iDb)
            [posIDs, dSq]= db.cp.getPosDbIDs(iDb);
            posIDs= posIDs(dSq>=0 & dSq<=db.nonTrivPosDistSqThr);
        end
        
    end
    
end
