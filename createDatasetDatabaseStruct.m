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
        if (~ strcmp(db_dir(i).name, '.') &&  ~ strcmp(db_dir(i).name, '..'))
            if (strcmp(db_dir(i).name, 'datasetInfoClean.csv'))
                dsetInfo = [data_db_path db_dir(i).name]
            else
                dsetInfo = [data_db_path db_dir(i).name '/datasetInfoClean.csv']
            end
            [id, place_name, lat, lon, elev, y, p, r, fov] = textread(dsetInfo, '%s %s %f %f %f %f %f %f %f', 'delimiter', ', ');
            %id to image name
            numimgs = size(id, 1);
            dirname = [db_dir(i).name '/'];
            if (strcmp(dirname, 'datasetInfoClean.csv/'))
                dirname = '';
            end
            prefix = repmat({['db/']}, numimgs, 1);
            suffix = repmat({'_segments.jpg'}, numimgs, 1);
            names = strcat(id, suffix);
            %names = strcat(prefix, names);
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
     
    q_loc_file = [data_path query_dir_name '/datasetInfoClean.csv']
    [q_id, q_names, q_lat, q_lon, q_elev, q_y, q_p, q_r, q_fov] = textread(q_loc_file, '%s %s %f %f %f %f %f %f %f', 'delimiter', ', ');
    %append suffix to query names to form complete file names.
    q_suffix = repmat({'.jpg'}, size(q_names, 1), 1);
    q_names = strcat(q_names, q_suffix);
    %prepend query dir name to query image names.
    numqueries = size(q_names, 1);
    q_prefix = repmat({[query_dir_name '/']}, numqueries, 1);
    %q_names = strcat(q_prefix, q_names);
    [q_utm_x, q_utm_y] = wgs2utm(q_lat, q_lon, utmzone, utmhemi);
    q_utm = [q_utm_x, q_utm_y]';
    clear q_utm_x q_utm_y;
    
    %% load query sets
    checkSets(data_path, query_dir_name);
    q_set_path = [data_path '/' query_dir_name];
    [q_train, q_test, q_val] = loadSets(q_set_path, q_names);
    
    %% load db sets
    [db_train, db_test, db_val] = loadSets([data_path '/db'], db_names);
    
    %% create the structures
    posDistThr = 20;
    nonTrivPosDist = 400;
   
    paths = localPaths();
   
    if (size(q_train, 1) > 0)
        dbStruct = buildDbStruct(db_names, q_names, db_train, q_train, db_utm, q_utm, posDistThr, nonTrivPosDist);
        save([paths.dsetSpecDir '/' dataset_name '_train'], 'dbStruct', '-v7.3');
    end
    if (size(q_test, 1) > 0)
        dbStruct = buildDbStruct(db_names, q_names, db_test, q_test, db_utm, q_utm, posDistThr, nonTrivPosDist);
        save([paths.dsetSpecDir '/' dataset_name '_test'], 'dbStruct', '-v7.3');
    end
    if (size(q_val, 1) > 0)
        dbStruct = buildDbStruct(db_names, q_names, db_val, q_val, db_utm, q_utm, posDistThr, nonTrivPosDist);
        save([paths.dsetSpecDir '/' dataset_name '_val'], 'dbStruct', '-v7.3');    
    end
end

function checkSets(data_path, query_dir_name)
    checkSet(data_path, query_dir_name, 'train')
    checkSet(data_path, query_dir_name, 'test')
    checkSet(data_path, query_dir_name, 'val')
end

function checkSet(data_path, query_dir_name, set)
    assert(ismember(set, {'train', 'val', 'test'}));
    
    q_set_path = [data_path '/' query_dir_name];
    db_set_path = [data_path '/db'];
    if (xor(exist([q_set_path '/' set '.txt'], 'file'), exist([db_set_path '/' set '.txt'])))
        error([set ' set must be specified in both query and db!']);
    end
end
% Loads train, test and validation sets from files if the train.txt, 
% test.txt and val.txt files exist. If not, only the test set containing
% all specified names is created.
%
% path  the path containing train.txt, test.txt and val.txt files listing
% the filenames for particular set.
% names complete list of names within the dataset (db or query).
function [train, test, val] = loadSets(path, names)
    train_list = [path '/train.txt'];
    test_list = [path '/test.txt'];
    val_list = [path '/val.txt'];
    
    if (exist(train_list, 'file') && exist(test_list, 'file') && exist(val_list, 'file'))
        train = textread(train_list, '%s');
        test = textread(test_list, '%s');
        val = textread(val_list, '%s');
    else
        train = [];
        test = names;
        val = [];
    end
end

function [db_utm] = getSetUtm(namesList, setList, utm)
    db_utm = zeros(size(setList, 1), 2);
    for i = 1 : size(setList, 1)
        idx = find(strcmp(namesList, setList{i})); 
        idx = idx(1);
        db_utm(i, :) = utm(idx, :);
    end
    db_utm = db_utm';
end

function [db] = buildDbStruct(db_names, q_names, db_list, q_list, db_utm, q_utm, posDistThr, nonTrivPosDistSqThr)
    db = struct;
    db.dbImageFns = db_list;
    db.qImageFns = q_list;
    db.utmDb = getSetUtm(db_names, db_list, db_utm');
    db.utmQ = getSetUtm(q_names, q_list, q_utm');
    db.posDistThr = posDistThr;
    db.nonTrivPosDistSqThr = nonTrivPosDistSqThr;
end