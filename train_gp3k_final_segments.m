
dbTrain = dbGeoPose3K_final_segments_FCN8s_model14('train'); 
dbVal = dbGeoPose3K_final_segments_FCN8s_model14('val');

%dbTrain = dbGeoPose3K_final_segments_Deeplab_model48('train'); 
%dbVal = dbGeoPose3K_final_segments_Deeplab_model48('val');


trainWeakly(dbTrain, dbVal, ...
    'netID', 'vd16', 'layerName', 'conv5_3', ...
    'method', 'vlad_preL2_intra', 'backPropToLayer', 'conv5_3', ...
    'margin', 0.1, ...
    'batchSize', 10, 'learningRate', 0.001, 'lrDownFreq', 3, 'momentum', 0.9, 'weightDecay', 0.1, 'compFeatsFrequency', 5, ...
    'nNegChoice', 30, 'nNegCap', 10, 'nNegCache', 10, ...
    'nEpoch', 10, ...
    'epochTestFrequency', 1, 'test0', true, ...
    'nTestSample', inf, 'nTestRankSample', 40, ...
    'saveFrequency', 15, 'doDraw', true, ...
    'useGPU', true, 'numThreads', 12, ...
    'computeBatchSize', 1, ...
    'info', 'GeoPose3K final segments FCN8s model 14, database 7m above ground.');
