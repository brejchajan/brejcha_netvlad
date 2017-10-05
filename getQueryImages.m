
%dbTrain = dbGeoPose3K_final_segments('train');
imFns = {};
for i = 1 : dbTrain.numQueries
    imFns = vertcat(imFns, dbTrain.dbImageFns(dbTrain.nontrivialPosQ(i)));
end

fid = fopen('geoPose3K_final_segments_positive_train.txt', 'w');
formatSpec = '%s\n';
for i = 1 : size(imFns, 1)
    fprintf(fid, formatSpec, imFns{i});
end
    