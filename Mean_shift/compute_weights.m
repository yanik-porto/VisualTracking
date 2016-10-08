function w = compute_weights(imPatch, q, p, m)

% w = compute_weights(imPatch, q, p, m)
%
% imPatch = Patch in the total image
% q = color distribution of the target
% p = color distribution of the current patch
% m = number of bins in the color distribution
%
% w = weights 

%Compute the ratio vector btw both color distributions
ratio = sqrt(q./p);

%Find weights over all the patch, for every pixels
for i = 1:size(imPatch,1)
    for j = 1:size(imPatch,2)
        %Check in which bin is the current pixel value
        val = imPatch(i,j);
        bin = ceil(val*m/255);
        if(bin==0)
            bin = 1;
        end
        %Compute the weight
        w(i,j) = ratio(bin);
    end
end


