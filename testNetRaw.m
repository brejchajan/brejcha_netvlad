dbVal = dbGeoPose3K_final_segments_FCN8s_model14('val');

load net
paths = localPaths()
netID='caffe'

dbFeatFn= sprintf('%s%s_%s_db.bin', paths.outPrefix, netID, dbVal.name);
qFeatFn = sprintf('%s%s_%s_q.bin', paths.outPrefix, netID, dbVal.name);

serialAllFeats(net, dbVal.qPath, dbVal.qImageFns, qFeatFn, 'batchSize', 30, 'useGPU', false); % Tokyo 24/7 query images have different resolutions so batchSize is constrained to 1
serialAllFeats(net, dbVal.dbPath, dbVal.dbImageFns, dbFeatFn, 'batchSize', 30, 'useGPU', false); % adjust batchSize depending on your GPU / network size

[recall, ~, ~, opts]= testFromFn(dbVal, dbFeatFn, qFeatFn);
figure
plot(opts.recallNs, recall, 'ro-'); grid on; xlabel('N'); ylabel('Recall@N');
