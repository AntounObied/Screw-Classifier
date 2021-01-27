function features = get_features(filename)
% This function takes an image filename as an input argument, and returns
% the area and perimeter features of each screw in the image.

    % Add testing and training directories to access images inside
    addpath('Images_Training_for_CS631')
    addpath('Images_Testing_for_CS631')

    % Read image and convert it to grayscale
    im = imread(filename);
    im_gray = rgb2gray(im);

    % Get histogram of the grayscale image
    hst = imhist(im_gray, 256);

    % Find the cumulative sum of the histogram
    cumulative_hst = cumsum(hst);

    % Normalize the matrix of cumulative sum by dividing by its last and
    % largest value
    normalized = cumulative_hst(:) / cumulative_hst(end);

    % Find the first point on the histogram above a chosen percentage of
    % the image as the background, and get the threshold to create binary
    % image
    [ff, idx] = find(normalized >= 0.94, 1, 'First');
    threshold = ff + 1;

    % Create binary image of values above the threshold
    im_bin = im_gray >= threshold;

    % Create structural elements to use for image cleaning
    s = strel('disk', 10);
    s_erode = strel('disk', 3);

    % Clean noise by opening and closing, and separate touching screws by
    % eroding slightly
    im_bin = imopen(im_bin, s);
    im_bin = imclose(im_bin, s);
    im_bin = imerode(im_bin, s_erode);

    % Find isolated regions in the binary image
    [L, num_regions] = bwlabel(im_bin);

    % Initialize feature matrix
    features = [];

    % For every region found by bwlabel
    for index = 1 : num_regions

        % Create variable of labeled region
        region = L == index;

        % Find area and perimeter properties of the region
        properties = regionprops('table', region, 'Area', 'Perimeter', 'MajorAxisLength');

        % If the region is too small or too large, ignore it
        if properties.Area < 4000 || properties.Area > 60000
            continue
        end

        % Add properties of current screw to the full feature table
        features = [features; properties(1, :)];

    end
end