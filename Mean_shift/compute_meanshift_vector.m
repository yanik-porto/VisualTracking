function z = compute_meanshift_vector(imPatch, prev_center, w)

% z = compute_meanshift_vector(imPatch, y, w)
%
% imPatch = Patch in the total image
% prev_center = prev_center in the patch
% w = weights

%get array of coordinates
[x, y] = meshgrid(1:size(imPatch,2),1:size(imPatch,1));

%Compute new center of mass
z = [sum(sum(x.*w)) sum(sum(y.*w))]./sum(sum(w));

%Add to previous center
z = round(z + prev_center);