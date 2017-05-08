function [ db_train_s, db_test_s, db_val_s ] = createDatasetStruct( data_path, dataset_name )
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
    db_train = textread([data_path '/db/train.txt'], '%s');
    
    [db_names, db_lat, db_lon] = textread([data_path '/db/location.txt'], '%s %f %f');
    db_utm = zeros(size(db_names, 1), 2);
    for i = 1 : size(db_names, 1)
        db_utm(i,:) = gpsToUTM([db_lat(i) db_lon(i)]);
    end
    
    db_test = textread([data_path '/db/test.txt'], '%s');
    db_val = textread([data_path '/db/val.txt'], '%s');
    
    [q_names, q_lat, q_lon] = textread([data_path '/query/location.txt'], '%s %f %f');
    q_utm = zeros(size(db_names, 1), 2);
    for i = 1 : size(q_names, 1)
        q_utm(i,:) = gpsToUTM([q_lat(i) q_lon(i)]);
    end
    q_train = textread([data_path '/query/train.txt'], '%s');
    q_test = textread([data_path '/query/test.txt'], '%s');
    q_val = textread([data_path '/query/val.txt'], '%s');

    posDistThr = 20;
    nonTrivPosDist = 400;
    db_train_s = buildDbStruct(db_names, q_names, db_train, q_train, db_utm, q_utm, posDistThr, nonTrivPosDist);
    db_test_s = buildDbStruct(db_names, q_names, db_test, q_test, db_utm, q_utm, posDistThr, nonTrivPosDist);
    db_val_s = buildDbStruct(db_names, q_names, db_val, q_val, db_utm, q_utm, posDistThr, nonTrivPosDist);
    
    paths = localPaths();
    dbStruct = db_train_s;
    save([paths.dsetSpecDir '/' dataset_name '_train'], 'dbStruct');
    dbStruct = db_test_s;
    save([paths.dsetSpecDir '/' dataset_name '_test'], 'dbStruct');
    dbStruct = db_val_s;
    save([paths.dsetSpecDir '/' dataset_name '_val'], 'dbStruct');    
end

function [db_utm] = getSetUtm(namesList, setList, utm)
    db_utm = zeros(size(setList, 1), 2);
    for i = 1 : size(setList, 1)
        idx = find(strcmp(namesList, setList{i})); idx = idx(1);
        db_utm(i, :) = utm(idx, :);
    end
    db_utm = db_utm';
end

function [db] = buildDbStruct(db_names, q_names, db_list, q_list, db_utm, q_utm, posDistThr, nonTrivPosDistSqThr)
    db = struct;
    db.dbImageFns = db_list;
    db.qImageFns = q_list;
    db.utmDb = getSetUtm(db_names, db_list, db_utm);
    db.utmQ = getSetUtm(q_names, q_list, q_utm);
    db.posDistThr = posDistThr;
    db.nonTrivPosDistSqThr = nonTrivPosDistSqThr;
end

