function visualizeTop(queryPath, positivePath, queryId, nn, count)
%UNTITLED Summary of this function goes here
%   queryId - id of the query
%   nn - nearest neighbour cell array
    numIm = (count + 2);
    i = queryId;
    subplot(3, 3, 1);
    imshow(queryPath);
    subplot(3, 3, 2);
    imshow(positivePath);
    for j = 3 : numIm
        subplot(3, 3, j);
        imshow(nn{i}{j-2});
    end
end

