%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%VISUAL TRACKING
% ----------------------
% Background Subtraction
% ----------------
% Date: september 2015
% Authors: You !!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
clear all;
close all;


%%%%% LOAD THE IMAGES
%=======================

% Give image directory and extension
imPath = 'Sequences/car'; imExt = 'jpg';

% check if directory and files exist
if isdir(imPath) == 0
    error('USER ERROR : The image directory does not exist');
end

filearray = dir([imPath filesep '*.' imExt]); % get all files in the directory
NumImages = size(filearray,1); % get the number of images
if NumImages < 0
    error('No image in the directory');
end

disp('Loading image files from the video sequence, please be patient...');
% Get image parameters
imgname = [imPath filesep filearray(1).name]; % get image name
I = imread(imgname); % read the 1st image and pick its size
VIDEO_WIDTH = size(I,2);
VIDEO_HEIGHT = size(I,1);

ImSeq = zeros(VIDEO_HEIGHT, VIDEO_WIDTH, NumImages);
for i=1:NumImages
    imgname = [imPath filesep filearray(i).name]; % get image name
    ImSeq(:,:,i) = imread(imgname); % load image
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
for i = 1:NumImages
    I = ImSeq(:,:,i);
    if(i > 1 && method == 1)
        B = median(ImSeq(:,:,1:i-1),3);
    end
    Diff = mat2gray(abs(I - B));
    mask = Diff > graythresh(Diff);
    
    %Get only foreground
    F = mask.*I;
    
    %Update
    B(mask) = alpha*I(mask)+(1-alpha)*B(mask);
    
    %Post processing for better detection
    I_track = im2bw(F);
    I_track = imerode(I_track, strel('rectangle', [2 2]));
    I_track = imdilate(I_track, strel('rectangle', [5 5]));
    
    %Detect biggest area
    s = regionprops(I_track, 'BoundingBox', 'Area');
    area = cat(1, s.Area);
    if(area)
        [~,ind] = max(area);
        bbox = s(ind).BoundingBox;
    end
   
    %Dipslay results
    subplot(221), imshow(I,[]), title('Current frame');
    subplot(222), imshow(B,[]), title('Estimated background');
    subplot(223),imshow(I_track,[]), title('Detected moving objects');
    subplot(224),imshow(I,[]), title('Moving object with bounding box)');
    if(area)
        hold on;
        rectangle('Position', bbox,'EdgeColor','r');
        hold off;
    end
    drawnow;
end

%% %%%%%%%%%%%%%%%%%%%%%%%%
%Running average Gaussian%
%%%%%%%%%%%%%%%%%%%%%%%%%%

%Initialize variables
alpha = 0.01;
mu = ImSeq(:,:,1);
sig = ones(size(I))*50;
T = 2.5;

%Iterate over all images
for i = 1:NumImages
    
    %Update mean and variance
    I = ImSeq(:,:,i);
    mu = alpha*I + (1-alpha)*mu;
    d = abs(I - mu);
    sig = d.^2.*alpha + (1-alpha).*sig;
    
    %detect foreground
    mask = abs(I-mu) > T*sqrt(sig);
    F = I.*mask;
    
    %Post processing for better detection
    I_track = im2bw(F);
    I_track = imerode(I_track, strel('rectangle', [2 2]));
    I_track = imdilate(I_track, strel('rectangle', [5 5]));
    
    %Detect biggest area
    s = regionprops(I_track, 'BoundingBox', 'Area');
    area = cat(1, s.Area);
    if(area)
        [~,ind] = max(area);
        bbox = s(ind).BoundingBox;
    end
    
    %Show results
    subplot(221), imshow(I,[]), title('Current frame');
    subplot(222), imshow(F,[]), title('Estimated background');
    subplot(223),imshow(I_track,[]), title('Detected moving objects');
    subplot(224),imshow(I,[]), title('Moving object with bounding box)');
    if(area)
        hold on;
        rectangle('Position', bbox,'EdgeColor','r');
        hold off;
    end

    drawnow;   
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%
% Eigen background
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%initialize variables
N = 30;
k = 15;
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

for i = 1:NumImages
    %Project an image
    y = x(:,i);
    p = Uk' * (y - m);
    y_hat = Uk * p + m;

    %Detect moving object
    mask = abs(y_hat - y) > T;
    F = reshape(y .* mask, VIDEO_HEIGHT, VIDEO_WIDTH);
    
    %Get the bounding box
    I_track = im2bw(F);
    I_track = imerode(I_track, strel('rectangle', [2 2]));
    I_track = imdilate(I_track, strel('rectangle', [5 5]));
    s = regionprops(I_track, 'BoundingBox', 'Area');
    area = cat(1, s.Area);
    if(area)
        [~,ind] = max(area);
        bbox = s(ind).BoundingBox;
    end
    
    %Show result
subplot(221), imshow(ImSeq(:,:,i),[]), title('Current frame');
    subplot(222), imshow(F,[]), title('Estimated background');
    subplot(223),imshow(I_track,[]), title('Detected moving objects');
    subplot(224),imshow(ImSeq(:,:,i),[]), title('Moving object with bounding box)');
    if(area)
        hold on;
        rectangle('Position', bbox,'EdgeColor','r');
        hold off;
    end

    drawnow;
end



