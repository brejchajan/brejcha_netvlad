function [ db_test_s ] = createDatasetDatabaseStruct( data_path, query_dir_name, dataset_name )
%db creates database struct from bare directory
%   Looks through the images and their coordinates defined in list files
%   and creates NetVLAD database structure.
%   -   data_path must contain subdirectories db and query
%   -   both db and query subdirs needs to contain train.txt, test.txt and
%   val.txt files, which define train, test, and validation sets. 
%   Furthermore, the db and query subdirs needs to
%   contain location.txt file, which defines location in GPS coordinates of
%   each image.
    %load('db_utm');
    
    %go through data_db_path, iterate over directories and process
    %datasetInfo.csv
    
    data_db_path = [data_path 'db/'];
    db_dir = dir(data_db_path);  
    
    db_names = {};
    db_lat = [];
    db_lon = [];
    for i = 1 : size(db_dir, 1)
        if (~ strcmp(db_dir(i).name, '.') &&  ~ strcmp(db_dir(i).name, '..') && strcmp(db_dir(i).name, 'CH_tile_5_6'))
            dsetInfo = [data_db_path db_dir(i).name '/datasetInfoClean_subarea_46.6714_7.78997_46.5228_8.14857.csv']
            [id, place_name, lat, lon, elev, y, p, r, fov] = textread(dsetInfo, '%s %s %f %f %f %f %f %f %f', 'delimiter', ', ');
            %id to image name
            numimgs = size(id, 1);
            prefix = repmat({['db/' db_dir(i).name '/']}, numimgs, 1);
            suffix = repmat({'_segments.jpg'}, numimgs, 1);
            names = strcat(id, suffix);
            names = strcat(prefix, names);
            db_names = [db_names ; names];
            db_lat = [db_lat ; lat];
            db_lon = [db_lon ; lon];
            break
        end
    end
    foo = [];
    
    %get the reference utmzone for the whole dataset
    [x, y, utmzone, utmhemi] = wgs2utm(db_lat(1), db_lon(1));
    
    [db_utm_x, db_utm_y] = wgs2utm(db_lat, db_lon, utmzone, utmhemi);
    db_utm = [db_utm_x, db_utm_y]';
    clear db_utm_x db_utm_y;
     
    q_loc_file = [data_path query_dir_name '/location.txt']
    [q_names, q_lat, q_lon] = textread(q_loc_file, '%s %f %f');
    %prepend query dir name to query image names.
    numqueries = size(q_names, 1);
    q_prefix = repmat({[query_dir_name '/']}, numqueries, 1);
    q_names = strcat(q_prefix, q_names);
    [q_utm_x, q_utm_y] = wgs2utm(q_lat, q_lon, utmzone, utmhemi);
    q_utm = [q_utm_x, q_utm_y]';
    clear q_utm_x q_utm_y;

    posDistThr = 20;
    nonTrivPosDist = 400;
    db_test_s = buildDbStruct(db_names, q_names, db_utm, q_utm, posDistThr, nonTrivPosDist);
    
    paths = localPaths();
    dbStruct = db_test_s;
    save([paths.dsetSpecDir '/' dataset_name '_test'], 'dbStruct', '-v7.3');

end

function [db] = buildDbStruct(db_names, q_names, db_utm, q_utm, posDistThr, nonTrivPosDistSqThr)
    db = struct;
    db.dbImageFns = db_names;
    db.qImageFns = q_names;
    db.utmDb = db_utm;
    db.utmQ = q_utm;
    db.posDistThr = posDistThr;
    db.nonTrivPosDistSqThr = nonTrivPosDistSqThr;
end