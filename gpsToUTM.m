function [ utm_coords ] = gpsToUTM( gps_coords )
%gpsToUTM converts the GPS coordinates [<latitude> <longitude>] to UTM.
%   gps_coords  contains coordinate in GPS: [<latitude> <longitude>].
%   utm_coords  output coordinate transformed into UTM coordinate system.
    
    z1 = utmzone(gps_coords);
    [ellipsoid, estr] = utmgeoid(z1);
    utmstruct = defaultm('utm');
    utmstruct.zone = '18T';
    utmstruct.geoid = ellipsoid;
    utmstruct = defaultm(utmstruct);
    [x,y] = mfwdtran(utmstruct, gps_coords(1), gps_coords(2));
    utm_coords = [x y];
end

