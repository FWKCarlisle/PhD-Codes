
fileno = 6448;
computerName = getenv('COMPUTERNAME');
%the following section will change directories depending on which PC you
%are using. You will have to set up the file structure yourself
switch computerName
    case 'DESKTOP-FKQN57G'
        loc = sprintf('C:\\Users\\nsf001\\OneDrive - The University of Nottingham(1)\\graphene domains\\', fileno);
        loc2 = "C:\\Users\\nsf001\\The University of Nottingham(1)\\Surface Structure - Duncan - I06_2024_10_LEEM_graphene\\";
    case 'LUIN81347'
        loc = sprintf('C:\\Users\\bramc44\\OneDrive - The University of Nottingham\\graphene domains\\', fileno);
        loc2 = 'C:\\Users\\bramc44\\The University of Nottingham\\Surface Structure - Duncan - I06_2024_10_LEEM_graphene\\';
end
loc = string(loc); 
%% image prepping function
function [normeddata, cleaned2] = image_prepping(loadpath,savepath, loc, fileno)
    % IMAGE_PREPPING - Prepares, normalizes, contours, and cleans images.
    %
    % Inputs:
    %   folder  - Full path to folder containing input .tif images
    %   loc     - Base location (string) for output folders
    %   fileno  - File number (used in output path naming)
    %
    % Outputs:
    %   normeddata - Normalized image stack
    %   contoured  - Contoured image stack (bright + sharp regions)
    %   contours   - Edge masks for each frame
    %   cleaned2   - Morphologically cleaned contoured images
    current_time = datetime('now', 'TimeZone', 'UTC+1');
    fprintf('Begins image processing %s\n',  datestr(current_time, 'yyyy-mm-dd HH:MM:SS'))
    normedFolder = fullfile(savepath, 'normed_data');
    cleanedFolder = fullfile(savepath, 'cleaned');
    if ~exist(normedFolder, 'dir')
        mkdir(normedFolder);
    end
    if ~exist(cleanedFolder, 'dir')
        mkdir(cleanedFolder);
    end


    % Load .tif images
    files = dir(fullfile(loadpath, '*.tif')); 
    numFiles = length(files);

    % Preallocate stack
    firstImage = imread(fullfile(loadpath, files(1).name));
    [rows, cols] = size(firstImage);
    imageStack = zeros(rows, cols, numFiles, 'like', firstImage);

    % Read all images
    for k = 1:numFiles
        img = imread(fullfile(loadpath, files(k).name));
        imageStack(:, :, k) = img;
    end

    % Normalization
    normeddata = zeros(size(imageStack));
    for i = 1:numFiles
        imagearray = imageStack(:, :, i);

        meanVal = mean(imagearray(:));
        stdVal = std(imagearray(:));

        withinRange = abs(imagearray - meanVal) <= 3 * stdVal;
        inliers = imagearray(withinRange);

        minVal = min(inliers);
        maxVal = max(inliers);

        normalizedInliers = (inliers - minVal) / (maxVal - minVal);
        normalizedArray = imagearray;
        normalizedArray(withinRange) = normalizedInliers;

        normeddata(:, :, i) = normalizedArray;
        filename = sprintf('normed_frame_%d.png', i);
        filePath = fullfile(normedFolder, filename);
        imwrite(normalizedArray, filePath)
       
        
    end

    % Contouring
    contoured = zeros(rows, cols, numFiles);
    contours  = zeros(rows, cols, numFiles);
    for t = 1:numFiles
        frame = normeddata(:,:,t);
        Iblur = imgaussfilt(frame, 2);

        [Gmag, ~] = imgradient(Iblur);

        bright_thresh = prctile(Iblur(:), 95);
        edge_thresh   = prctile(Gmag(:), 90);

        bright_mask = Iblur > bright_thresh;
        edge_mask   = Gmag > edge_thresh;

        final_mask = bright_mask & edge_mask;
        final_mask_filled = imfill(final_mask, 'holes');

        maskedframe = frame;
        maskedframe(~final_mask_filled) = 0;

        contoured(:,:,t) = maskedframe;
        contours(:,:,t)  = edge_mask;

        
    end

    % Second cleaning (morphological)
    cleaned2 = zeros(rows, cols, numFiles);
    se = strel('disk', 1);
    for t = 1:numFiles
        opened_img = imopen(contoured(:,:,t), se);
        processed_img = imclose(opened_img, se);
        cleaned2(:,:,t) = processed_img;
        % Optional save:
         filename = sprintf('cleaned_frame_%d.png', t);
         filePath = fullfile(cleanedFolder, filename);
         imwrite(processed_img, filePath);
    end
    current_time = datetime('now', 'TimeZone', 'UTC+1');
    fprintf('Completed image processing %s\n',  datestr(current_time, 'yyyy-mm-dd HH:MM:SS'))
end

%% just cleaning function
function cleaned2 = justcleaning(cropped_frames)
%Extracts bright & sharp regions and returns cleaned frames
%
% Inputs:
%   cropped_frames - 3D matrix (H x W x N) of normalized frames
%   loc        - (optional) string path to save results (e.g. 'C:\data\')
%
% Output:
%   cleaned2   - 3D matrix after masking + morphological cleaning
%
% Example:
%   cleaned2 = processFrames(normeddata, 'C:\output\');

    % Preallocate
    [h, w, n] = size(cropped_frames);
    contoured = zeros(h, w, n);

    % Parameters
    bright_percentile = 95;   
    edge_percentile   = 90;  
    for t = 1:n
        frame = cropped_frames(:,:,t);
        Iblur = imgaussfilt(frame, 2);

        [Gmag, ~] = imgradient(Iblur);

        bright_thresh = prctile(Iblur(:), bright_percentile);
        edge_thresh   = prctile(Gmag(:), edge_percentile);

        bright_mask = Iblur > bright_thresh;
        edge_mask   = Gmag > edge_thresh;

        final_mask = bright_mask & edge_mask;
        final_mask_filled = imfill(final_mask, 'holes');

        maskedframe = frame;
        maskedframe(~final_mask_filled) = 0;

        contoured(:,:,t) = maskedframe;

        end


    % Second cleaning (morphological processing)
    cleaned2 = zeros(h, w, n);
    se = strel('disk', 1);
    for t = 1:n
        frame = contoured(:,:,t);
        opened_img = imopen(frame, se);
        processed_img = imclose(opened_img, se);
        cleaned2(:,:,t) = processed_img;
    end

end

%% vector shift function

function allvectors = object_tracking_and_vectors(cleaned2, savepath1, minObjectSize, distanceThreshold, minFrames)
    % OBJECT_TRACKING_AND_VECTORS
    % Tracks objects across frames, finds motion vectors, and returns cumulative vectors.
    %
    % Inputs:
    %   cleaned2          - Binary image stack (rows x cols x frames) after morphological cleaning
    %   saveloc               - Output folder for saving variables (so you
    %   don't have to run this again if testing things)
    %   minObjectSize     - Minimum object area (pixels) (25 pixels has
    %   worked before)
    %   distanceThreshold - Max centroid distance for same-object tracking
    %   (normally set to two pixels)
    %   minFrames         - Minimum frames object must appear in to be
    %   included - normally just select two frame objects (eg = 2)
    %
    % Output:
    %   allvectors        - Cumulative average motion vectors across frames
    
    n = size(cleaned2, 3);
    
    % Step 1: Pick out objects in each frame
    current_time = datetime('now', 'TimeZone', 'UTC+1');
    fprintf('counting objects. %s\n',  datestr(current_time, 'yyyy-mm-dd HH:MM:SS'))
    trackedObjects = cell(n, 1);
    for k = 1:n
        frame = cleaned2(:,:,k);
        CC = bwconncomp(frame);
        stats = regionprops(CC, 'Area', 'Centroid', 'BoundingBox');
        validIdx = find([stats.Area] > minObjectSize);
        trackedObjects{k} = stats(validIdx);
    end
    
    % Step 2: Track objects across frames
    current_time = datetime('now', 'TimeZone', 'UTC+1');
    fprintf('tracking objects. %s\n',  datestr(current_time, 'yyyy-mm-dd HH:MM:SS'))
    nextObjectID = 1;
    trackedData = struct();

    % Initialize from first frame
    for i = 1:length(trackedObjects{1})
        trackedData(nextObjectID).ID = nextObjectID;
        trackedData(nextObjectID).frames = 1;
        trackedData(nextObjectID).centroids = trackedObjects{1}(i).Centroid;
        trackedData(nextObjectID).areas = trackedObjects{1}(i).Area;
        nextObjectID = nextObjectID + 1;
    end

    % Track over remaining frames
    for k = 2:n
        currentObjects = trackedObjects{k};
        for i = 1:length(currentObjects)
            currCentroid = currentObjects(i).Centroid;
            bestDist = inf;
            bestID = -1;
            for id = 1:length(trackedData)
                if trackedData(id).frames(end) == k-1
                    lastCentroid = trackedData(id).centroids(end, :);
                    dist = norm(currCentroid - lastCentroid);
                    if dist < bestDist && dist < distanceThreshold
                        bestDist = dist;
                        bestID = id;
                    end
                end
            end
            if bestID > 0
                trackedData(bestID).frames(end+1) = k;
                trackedData(bestID).centroids(end+1, :) = currCentroid;
                trackedData(bestID).areas(end+1) = currentObjects(i).Area;
            else
                trackedData(nextObjectID).ID = nextObjectID;
                trackedData(nextObjectID).frames = k;
                trackedData(nextObjectID).centroids = currCentroid;
                trackedData(nextObjectID).areas = currentObjects(i).Area;
                nextObjectID = nextObjectID + 1;
            end
        end
    end

    % Step 3: Filter for objects appearing in >= minFrames
    current_time = datetime('now', 'TimeZone', 'UTC+1');
    fprintf('organising %d frame objects. %s\n',minFrames,  datestr(current_time, 'yyyy-mm-dd HH:MM:SS'))
    filteredObjects = trackedData([cellfun(@length, {trackedData.frames})] >= minFrames);
    save(fullfile(savepath1, '2frameobjects.mat'), 'filteredObjects');

    % Step 4: Extract pairs of centroids in consecutive frames
    two_frame_objects = cell(1, n-1);
    for k = 1:n-1
        centroids_k = [];
        centroids_k_plus_1 = [];
        for i = 1:length(filteredObjects)
            obj = filteredObjects(i);
            if ismember(k, obj.frames) && ismember(k+1, obj.frames)
                idx1 = find(obj.frames == k);
                centroids_k = [centroids_k; obj.centroids(idx1, :)];
                idx2 = find(obj.frames == k+1);
                centroids_k_plus_1 = [centroids_k_plus_1; obj.centroids(idx2, :)];
            end
        end
        two_frame_objects{k} = {centroids_k, centroids_k_plus_1};
    end
    save(fullfile(savepath1, 'two_frame_objects.mat'), 'two_frame_objects');

    % Step 5: Motion vector analysis
    current_time = datetime('now', 'TimeZone', 'UTC+1');
    fprintf('calculating vectors. %s\n',  datestr(current_time, 'yyyy-mm-dd HH:MM:SS'))
    vectors = zeros(n, 2);
    for t = 1:n-1
        x1s = two_frame_objects{t}{1}(:,1);
        y1s = two_frame_objects{t}{1}(:,2);
        x2s = two_frame_objects{t}{2}(:,1);
        y2s = two_frame_objects{t}{2}(:,2);

        fixedPoints = [x1s, y1s];
        movingPoints = [x2s, y2s];
        shifts = fixedPoints - movingPoints;

        if ~isempty(shifts)
            angles = atan2(shifts(:, 2), shifts(:, 1));
            numBins = 36;
            windowSize = 9;
            agreementThreshold = 0.5;
            [counts, edges] = histcounts(angles, numBins);
            countsExtended = [counts, counts(1:windowSize-1)];
            windowSums = movsum(countsExtended, [0 windowSize-1]);
            [maxInWindow, maxWindowStart] = max(windowSums(1:numBins));

            if maxInWindow >= agreementThreshold * sum(counts)
                angleLower = edges(maxWindowStart);
                angleUpper = edges(mod(maxWindowStart + windowSize - 2, numBins) + 1);
                if angleUpper > angleLower
                    selectedIdx = angles >= angleLower & angles < angleUpper;
                else
                    selectedIdx = angles >= angleLower | angles < angleUpper;
                end
                qualifyingVectors = shifts(selectedIdx, :);
                avgVector = mean(qualifyingVectors, 1);
                vectors(t,:) = avgVector;
            else
                vectors(t,:) = [0, 0];
            end
        end
    end

    % Step 6: Cumulative vectors
    allvectors = cumsum(vectors, 1);
    current_time = datetime('now', 'TimeZone', 'UTC+1');
    fprintf('vector calculation complete complete. %s\n',  datestr(current_time, 'yyyy-mm-dd HH:MM:SS'))
end


%% calling the functions
%normalise and threshold the images
fileno = 6448;%your file number here
loadpath = loc+sprintf('IGOR Output\\adjusted%dgood',fileno); %where you saved your .tif files
savepath1 = loc+sprintf('slides\\matlab\\%d\\final_attempt_secondary',fileno); %where you want to save everything - folders will be created in this location
%these are commented out by default so you will need to look inside the
%functions to enable this
[normed,cleaned] = image_prepping(loadpath,savepath1, loc, fileno);
%object based vector shift
minObjectSize = 25; %minimum size in pixels of objects you want the code to detect
distanceThreshold = 2; %distance in pixels between two objects on consecutive slides for them to be considered the same object
minFrames = 2; %number of frames an object has to exist to be considered when calculating vectors
allvectors = object_tracking_and_vectors(cleaned, savepath1, minObjectSize, distanceThreshold, minFrames);
% obviously this function is designed to produce the frame to frame shift;
% you will have to edit it a bit to get the object properties out but it
% shouldn't be too hard