
idx_bbox = isInBBox(lat, lon, 46.6714, 7.78997, 46.5228, 8.14857);

fid = fopen('eth_ch2_CH_tile_5_6.csv', 'w');
for i = 1 : size(idx_bbox, 1)
    fprintf(fid, '%s %s %f %f %f %f %f %f %f\n', id{idx_bbox(i)}, place_name{idx_bbox(i)}, lat(idx_bbox(i)), lon(idx_bbox(i)), elev(idx_bbox(i)), y(idx_bbox(i)), p(idx_bbox(i)), r(idx_bbox(i)), fov(idx_bbox(i)));
end
fclose(fid)