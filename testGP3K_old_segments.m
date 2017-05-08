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

%{
old test
im_synth = vl_imreadjpeg({'google_maps/google_dom_sp_dom1.jpg'}); im_synth= im_synth{1}; % slightly convoluted because we need the full image path for `vl_imreadjpeg`, while `imread` is not appropriate - see `help computeRepresentation`
im_synth_neg = vl_imreadjpeg({'google_maps/dom_20y_20a_83t.jpg'}); im_synth_neg= im_synth_neg{1}; % slightly convoluted because we need the full image path for `vl_imreadjpeg`, while `imread` is not appropriate - see `help computeRepresentation`
im_orig = vl_imreadjpeg({'google_maps/sp_dom1.jpg'}); im_orig= im_orig{1}; % slightly convoluted because we need the full image path for `vl_imreadjpeg`, while `imread` is not appropriate - see `help computeRepresentation`


feats_synth = computeRepresentation(net, im_synth, 'useGPU', 'false'); % add `'useGPU', false` if you want to use the CPU
feats_synth_neg = computeRepresentation(net, im_synth_neg, 'useGPU', 'false'); % add `'useGPU', false` if you want to use the CPU
feats_orig = computeRepresentation(net, im_orig, 'useGPU', 'false'); % add `'useGPU', false` if you want to use the CPU

norm(feats_synth-feats_orig, 2)
norm(feats_synth_neg-feats_orig, 2)
norm(feats_synth_neg-feats_synth, 2)

[ids, dist] = yael_nn([feats_synth feats_synth_neg], feats_orig, 2)
%}

%% test on test set
%dbTest = dbGP3K_old_segments('test');
dbTest = dbGP3K_old_segments_tile_3_8('test')
dbFeatFn= sprintf('%s%s_%s_db.bin', paths.outPrefix, netID, dbTest.name);
qFeatFn = sprintf('%s%s_%s_q.bin', paths.outPrefix, netID, dbTest.name);

% Compute db/query image representations
serialAllFeats(net, dbTest.dbPath, dbTest.dbImageFns, dbFeatFn, 'batchSize', 30); % adjust batchSize depending on your GPU / network size
serialAllFeats(net, dbTest.qPath, dbTest.qImageFns, qFeatFn, 'batchSize', 30); % Tokyo 24/7 query images have different resolutions so batchSize is constrained to 1
        
% Measure recall@N
%[fraction, Ns opts]= testFromFnROC(dbTest, dbFeatFn, qFeatFn);
%figure
%plot(Ns, fraction, 'r-'); grid on; xlabel('N'); ylabel('Recall@N');
[recall, ~, ~, opts]= testFromFn(dbTest, dbFeatFn, qFeatFn);
figure
plot(opts.recallNs, recall, 'ro-'); grid on; xlabel('N'); ylabel('Recall@N');
