%netID = 'vd16_pitts30k_conv5_3_vlad_preL2_intra_white'; %best netvlad
netID = '775c_ep000020_latest'; %segments
%netID = 'fb44_ep000010_latest' %best newly trained on small google earth (GESS)
%netID = 'd5ad_latest' %currently being trained newly on larger Google earth GESS_ws_380_80
%netID = 'imagenet-vgg-verydeep-16' %original pretrained network used as starting point for GESS*.

paths = localPaths();
%load( sprintf('%s%s.mat', paths.ourCNNs, netID), 'net' );
load( sprintf('%s%s.mat', '~/data/netvlad/output/trained/', netID), 'net');
%load( sprintf('%s%s.mat', paths.pretrainedCNNs, netID), 'net' );
net = relja_simplenn_tidy(net);


%% test on test set
dbTest = dbCH_seg_tile_2_6_ETH_CH2_model_48_no_crf()
dbFeatFn= sprintf('%s%s_%s_db.bin', paths.outPrefix, netID, dbTest.name);
qFeatFn = sprintf('%s%s_%s_q.bin', paths.outPrefix, netID, dbTest.name);

% Compute db/query image representations
%serialAllFeats(net, dbTest.dbPath, dbTest.dbImageFns, dbFeatFn, 'batchSize', 30); % adjust batchSize depending on your GPU / network size
%serialAllFeats(net, dbTest.qPath, dbTest.qImageFns, qFeatFn, 'batchSize', 30); % Tokyo 24/7 query images have different resolutions so batchSize is constrained to 1
        
% Measure recall@N
%[fraction, Ns opts]= testFromFnROC(dbTest, dbFeatFn, qFeatFn);
%figure
%plot(Ns, fraction, 'r-'); grid on; xlabel('N'); ylabel('Recall@N');
[recall, ~, ~, opts]= testFromFn(dbTest, dbFeatFn, qFeatFn);
figure
plot(opts.recallNs, recall, 'ro-'); grid on; xlabel('N'); ylabel('Recall@N');