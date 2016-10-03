%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%VISUAL TRACKING
% ----------------------
% Background Subtraction
% ----------------
% Date: september 2015
% Authors: You !!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clear all
close all


%%%%% LOAD THE IMAGES
%=======================

% Give image directory and extension
imPath = 'Sequences/highway/input'; imExt = 'jpg';
gtPath = 'Sequences/highway/groundtruth'; gtExt = 'png';
% check if directory and files exist
if isdir(imPath) == 0
    error('USER ERROR : The image directory does not exist');
end

filearray = dir([imPath filesep '*.' imExt]); % get all files in the directory
gtarray = dir([gtPath filesep '*.' gtExt]);

NumImages = size(filearray,1); % get the number of images
if NumImages < 0
    error('No image in the directory');
end

disp('Loading image files from the video sequence, please be patient...');
% Get image parameters
imgname = [imPath filesep filearray(1).name]; % get image name
gtname = [gtPath filesep gtarray(1).name];

I = imread(imgname); % read the 1st image and pick its size
GT = imread(gtname);
VIDEO_WIDTH = size(I,2);
VIDEO_HEIGHT = size(I,1);

ImSeq = zeros(VIDEO_HEIGHT, VIDEO_WIDTH, NumImages);
GtSeq = zeros(VIDEO_HEIGHT, VIDEO_WIDTH, NumImages);
for i=1:NumImages
    imgname = [imPath filesep filearray(i).name]; % get image name
    ImSeq(:,:,i) = rgb2gray(imread(imgname)); % load image
    gtname = [gtPath filesep gtarray(i).name]; % get image name
    GtSeq(:,:,i) = im2bw(imread(gtname)); % load image
end
disp(' ... OK!');


%%BACKGROUND SUBTRACTION
%=======================

%% %%%%%%%%%%%%%%%%%%
%Frame differencing%
%%%%%%%%%%%%%%%%%%%%

alpha = 0.05;
B = ImSeq(:,:,1);
% method = 1; %With median = 1
method = 1; %With update = 2

% Describe here your background subtraction method
% for i = 471:NumImages
i = 1690;
j = 1;
for T = 0.1:0.1:1
    I = ImSeq(:,:,i);
    if(i > 1 && method == 1)
        B = median(ImSeq(:,:,i-470:i-1),3);
    end
    Diff = mat2gray(abs(I - B));
%     mask = Diff > graythresh(Diff);
    mask = Diff > T;
    
    %Get only foreground
    F = mask.*I;
    
    %Update
    B(mask) = alpha*I(mask)+(1-alpha)*B(mask);
    
    %Proprocessing for better detection
    I_track = im2bw(F);
%     I_track = imerode(I_track, strel('rectangle', [2 2]));
%     I_track = imdilate(I_track, strel('rectangle', [5 5]));
    
%     %Detect biggest area
%     s = regionprops(I_track, 'BoundingBox', 'Area');
%     area = cat(1, s.Area);
%     if(area)
%         [~,ind] = max(area);
%         bbox = s(ind).BoundingBox;
%     end
%    
    %Dipslay results
    subplot(121), imshow(GtSeq(:,:,i),[]), title('Ground truth');
    subplot(122),imshow(I_track,[]), title('Detected moving objects');
%     subplot(133),imshow(I,[]), title('Moving object with bounding box)');
%     if(area)
%         hold on;
%         rectangle('Position', bbox,'EdgeColor','r');
%         hold off;
%     end
    drawnow;
    false_p = sum(sum((I_track - GtSeq(:,:,i)) > 0));
    true_p = sum(sum((I_track == 1) & (GtSeq(:,:,i) == 1)));
    false_n = sum(sum((I_track - GtSeq(:,:,i)) < 0));
    precision(1,j) = true_p/(true_p + false_p);
    recall(1,j) = true_p/(true_p + false_n);
    F_score(1,j) = 2*(precision(1,j)*recall(1,j))/(precision(1,j)+recall(1,j));
    disp(j);
    j = j + 1;
    
end

figure;
plot(1-recall(1,1:9),precision(1,1:9));
title('Precision according to the recall while the threshold is increasing');
xlabel('1-recall')
ylabel('precision');

%% %%%%%%%%%%%%%%%%%%%%%%%%
%Running average Gaussian%
%%%%%%%%%%%%%%%%%%%%%%%%%%

%Initialize variables
alpha = 0.01;
mu = ImSeq(:,:,1);
sig = ones(size(I))*50;
% T = 2.5;

%Iterate over all images
% for i = 1:NumImages
i = 1690;
j = 1;
for T = 1:0.5:5
    
    %Update mean and variance
    I = ImSeq(:,:,i);
    mu = alpha*I + (1-alpha)*mu;
    d = abs(I - mu);
    sig = d.^2.*alpha + (1-alpha).*sig;
    
    %detect foreground
    mask = abs(I-mu) > T*sqrt(sig);
    F = I.*mask;
    
    %Proprocessing for better detection
    I_track = im2bw(F);
%     I_track = imopen(I_track, strel('rectangle', [2 2]));
%     I_track = imerode(I_track, strel('rectangle', [2 2]));
%     I_track = imdilate(I_track, strel('rectangle', [5 5]));
%     
%     %Detect biggest area
%     s = regionprops(I_track, 'BoundingBox', 'Area');
%     area = cat(1, s.Area);
%     if(area)
%         [~,ind] = max(area);
%         bbox = s(ind).BoundingBox;
%     end
    
    %Dipslay results
    subplot(121), imshow(GtSeq(:,:,i),[]), title('Ground truth');
    subplot(122),imshow(I_track,[]), title('Detected moving objects');
% %     subplot(133),imshow(I,[]), title('Moving object with bounding box)');
%     if(area)
%         hold on;
%         rectangle('Position', bbox,'EdgeColor','r');
%         hold off;
%     end
    drawnow;   
    
    false_p = sum(sum((I_track - GtSeq(:,:,i)) > 0));
    true_p = sum(sum((I_track == 1) & (GtSeq(:,:,i) == 1)));
    false_n = sum(sum((I_track - GtSeq(:,:,i)) < 0));
    precision(2,j) = true_p/(true_p + false_p);
    recall(2,j) = true_p/(true_p + false_n);
    F_score(2,j) = 2*(precision(2,j)*recall(2,j))/(precision(2,j)+recall(2,j));
    disp(j);
    
    j = j+1;
end

figure;
plot(1-recall(2,1:9),precision(2,1:9));
title('Precision according to the recall while the threshold is increasing');
xlabel('1-recall')
ylabel('precision');

%% %%%%%%%%%%%%%%%%%%%%%%%%%
% Eigen background
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%initialize variables
N = 470;
k = 10;
T = 25;

%Reshape images
for i = 1:NumImages
   x(:,i) = reshape(ImSeq(:,:,i), [], 1);
end

%Building the mean image
m = 1/N*sum(x(:,1:N),2);

%Compute mean-normalized image vectors
X = x - repmat(m,1,size(x,2));

%Performe the singular value decomposition of the matrix X
[U, S, V] = svd(X, 'econ');

%Keep the k principal components of U
Uk = U(:,1:k);

% for i = 1:NumImages
i = 1690;
j = 1;
for T = 10:5:50
    %Project an image
    y = x(:,i);
    p = Uk' * (y - m);
    y_hat = Uk * p + m;

    %Detect moving object
    mask = abs(y_hat - y) > T;
    F = reshape(y .* mask, VIDEO_HEIGHT, VIDEO_WIDTH);
    
    %Get the bounding box
    I_track = im2bw(F);
%     I_track = imerode(I_track, strel('rectangle', [2 2]));
%     I_track = imdilate(I_track, strel('rectangle', [5 5]));
%     s = regionprops(I_track, 'BoundingBox', 'Area');
%     area = cat(1, s.Area);
%     if(area)
%         [~,ind] = max(area);
%         bbox = s(ind).BoundingBox;
%     end
    
    %Dipslay results
    subplot(121), imshow(GtSeq(:,:,i),[]), title('Ground truth');
    subplot(122),imshow(I_track,[]), title('Detected moving objects');
%     subplot(133),imshow(I,[]), title('Moving object with bounding box)');
%     if(area)
%         hold on;
%         rectangle('Position', bbox,'EdgeColor','r');
%         hold off;
%     end
    drawnow;
    
    false_p = sum(sum((I_track - GtSeq(:,:,i)) > 0));
    true_p = sum(sum((I_track == 1) & (GtSeq(:,:,i) == 1)));
    false_n = sum(sum((I_track - GtSeq(:,:,i)) < 0));
    precision(3,j) = true_p/(true_p + false_p);
    recall(3,j) = true_p/(true_p + false_n);
    F_score(3,j) = 2*(precision(3,j)*recall(3,j))/(precision(3,j)+recall(3,j));
    disp(i);

    j = j+1;
end

figure;
plot(1-recall(3,1:9),precision(3,1:9));
title('Precision according to the recall while the threshold is increasing');
xlabel('1-recall')
ylabel('precision');