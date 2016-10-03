function k = compute_bhattacharyya_coefficient(p,q)
%
% k = compute_bhattacharyya_coefficient(p,q)
%
% p,q = color distribution of each candidate
% k = bhattacharyya coefficient

%Compute the coefficient
k = sum(sqrt(p.*q));