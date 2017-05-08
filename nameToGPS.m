function [ gps ] = nameToGPS(name, names, gpsdb)
%Jan Brejcha 2016
%nameToGps Function translating retrieved name to gps.
    [pathstr,filename,ext] = fileparts(name); 
    ind = find(ismember(names, filename));
    gps = gpsdb(ind, :);
end

