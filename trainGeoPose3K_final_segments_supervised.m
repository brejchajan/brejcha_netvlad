
dbTrain = dbGeoPose3K_final_segments_FCN8s_model14('train'); 
dbVal = dbGeoPose3K_final_segments_FCN8s_model14('val');

%dbTrain = dbGeoPose3K_final_segments_Deeplab_model48('train'); 
%dbVal = dbGeoPose3K_final_segments_Deeplab_model48('val');

gpuDevice(1);
trainOrientationSupervised(dbTrain, dbVal, ...
    'netID', 'caffe', 'layerName', 'fc8', ...
    'method', 'vlad_preL2_intra', ...
    'margin', 0.1, ...
    'batchSize', 5, 'learningRate', 0.00005, 'lrDownFreq', 1, 'momentum', 0.9, 'weightDecay', 0.1, 'compFeatsFrequency', 20, ...
    'nNegChoice', 1, 'nPosChoice', 1, 'nNegCap', 10, 'nNegCache', 10, ...
    'nEpoch', 10, ...
    'epochTestFrequency', 1, 'test0', true, ...
    'nTestSample', inf, 'nTestRankSample', 40, ...
    'saveFrequency', 15, 'doDraw', true, ...
    'useGPU', false, 'numThreads', 12, ...
    'computeBatchSize', 10, ...
    'showTrainingImgs', false, ...
    'info', 'GeoPose3K final segments FCN8s model 14, database 7m above ground.');

%'sessionID', 'ac5f', 'startEpoch', 2, ...
%'backPropToLayer', 'fc6', ...