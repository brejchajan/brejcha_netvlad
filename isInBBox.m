function [ idx ] = isInBBox( lat, lon, nw_lat, nw_lon, se_lat, se_lon )
%isInBBox tests whether given position is inside the bounding box. 
%lat lon are arrays of latitude and longitude positions to be tested,
%returns indices of elements inside the bounding box.

    idx = find((lat < nw_lat) & (lat > se_lat) & (lon > nw_lon) & (lon < se_lon));

end

