function Classify_Screws()
% This function trains a classifier by reading in training images, then
% classifies screws in the testing images.

    % Boolean variable to specify if a classifier is already trained
    is_already_trained = false;

    % Names of training and testing directories
    training_dir = 'Images_Training_for_CS631';
    testing_dir = 'Images_Testing_for_CS631';
    
    % Types of names of training and testing files
    name_mix          = 'im_mix*.jpg';
    
    % Boolean to specify that feature table needs to be initialized
    init_feature_table = true;
    
    % Possible classes available in the images
    classes = 'ABC';
    
    % If a classifier has not been trained yet, initiate training process
    if (is_already_trained == false)
        
        % For each class A, B, or C
        for class = 1 : length(classes)
            
            % Current letter to be trained
            this_letter = classes(class);
            
            % Get list of all files in training directory
            files = sprintf('%s%cim_type_%c*.jpg', training_dir, filesep(), this_letter);
            training_files = dir(files);

            % For every file in the directory
            for index = 1 : length(training_files)
                
                % Get current file name and print it
                filename = sprintf('%s%c%s', training_dir, filesep(), training_files(index).name);
                fprintf('%s\n', filename);

                % Get the features of all the screws in the image
                features = get_features(filename);
                
                % Get number of screws found for appending features to the
                % table
                n_new = size(features, 1);

                % If this is the first entry
                if (init_feature_table == true)
                    
                    % Set to false because subsequent entries will not be
                    % the first
                    init_feature_table = false;

                    % Set current features to total corrected features, and
                    % list them as the current class
                    collected_features = features;
                    class_list(1 : n_new) = class;
                else
                    % If not first entry, append new features to total
                    % features collected
                    collected_features(end + 1: end + n_new , :) = features;
                    class_list(end + 1 : end + n_new) = class;

                end
            end
        end

        % Fit multiclass model for support vector machine classifier
        classifier = fitcecoc(collected_features, class_list.');

        % Save the fitted classifier
        save classifier.mat classifier
    else
        % If the program has already been trained, just load classifier to
        % avoid unnecessary time-consuming training
        load classifier.mat classifier
    end
    
    
    % Testing portion begins here
    fprintf('\n\n\nTesting Begun \n\n\n')
    
    % Get list of files from testing directory
    testing_files = dir(strcat(testing_dir, filesep(), name_mix));
    
    % Print header for user convenience
    disp('Filename                                           Count A       Count B      Count C      Unknown')
    
    % For every file in the testing directory
    for index = 1 : length(testing_files)
        
        % Initialize counters for types A, B, and C, as well as any unknown
        % regions
        countA = 0;
        countB = 0;
        countC = 0;
        countUnknown = 0;
        
        % Get current filename
        filename = sprintf('%s%c%s', testing_dir, filesep(), testing_files(index).name);
        
        % Read current image and convert it to grayscale format
        im = imread(filename);
        im_gray = rgb2gray(im);
        
        % Get image size and number of channels to form output colored
        % image based on program classification
        rows = size(im, 1);
        cols = size(im, 2);
        channels = 3;
        output = ones(rows, cols, channels);

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
        
        % For every screw in the image
        for count = 1 : num_regions
            
            % Create region of the screw
            region = L == count;

            % Get the area and perimeter properties of the screw
            properties = regionprops('table', region, 'Area', 'Perimeter', 'MajorAxisLength');

            % If area is too small or too large, increment unknown count,
            % color the region magenta by subtracting all the green from
            % the output image, and avoid classifying this region
            if properties.Area < 4000 || properties.Area > 60000
                countUnknown = countUnknown + 1;
                output(:, :, 2) = output(:, :, 2) - (255 * region);
                continue
            end
    
            % Predict the class of this current screw using built
            % classifier
            current_class = predict(classifier, properties);
            
            % If screw is found to be a particular class, increment the
            % count of that type, and color in the region appropriately by
            % subtracting the two other colors from the output image
            if current_class == 1
                countA = countA + 1;
                output(:, :, 2) = output(:, :, 2) - (255 * region);
                output(:, :, 3) = output(:, :, 3) - (255 * region);
            elseif current_class == 2
                countB = countB + 1;
                output(:, :, 1) = output(:, :, 1) - (255 * region);
                output(:, :, 3) = output(:, :, 3) - (255 * region);
            else
                countC = countC + 1;
                output(:, :, 1) = output(:, :, 1) - (255 * region);
                output(:, :, 2) = output(:, :, 2) - (255 * region);
            end
            
        end
        
        % Create output title for figure and for writing out image
        output_title = strcat('Sorted_', testing_files(index).name);
        
        % Show sorted image
        figure
        imshow(output)
        title(output_title, 'Interpreter', 'none')
        axis on
        axis image
        
        % Print the count of each type, as well as the unknown
        fprintf('%s       %d            %d            %d            %d\n\n', filename, countA, countB, countC, countUnknown);
        
        imwrite(output, output_title);
    end
    
end