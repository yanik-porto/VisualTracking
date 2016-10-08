function z = compute_meanshift_vector(imPatch, prev_center, w)

% z = compute_meanshift_vector(imPatch, y, w)
%
% imPatch = Patch in the total image
% prev_center = prev_center in the patch
% w = weights

sz = floor(size(imPatch)/2);
% if(mod(sz(1),2)==0)
%     sz(


%get array of coordinates
[x, y] = meshgrid(1:size(imPatch,2),1:size(imPatch,1));
x = x - sz(2) + prev_center(1);
y = y - sz(1) + prev_center(2);
% [x, y] = meshgrid(-sz(2):sz(2),-sz(1):sz(1));

%Compute new center of mass
z = [sum(sum(x.*w)) sum(sum(y.*w))]./sum(sum(w));

%Add to previous center
z = round(z);% + prev_center);