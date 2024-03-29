function [cross_idx] = GetIdx_of_xFold_Cross( idx, xFold )
% idx 为某类数据的位置索引值
% xFold 交叉次数
% 将idx随机分为xFold组，即通过cross_idx可以将原始数据分为xFold组。
if isempty(xFold)
    xFold = 2; %default
end
size_each_part = fix( size(idx,1)/xFold );
randidx = randperm( size(idx,1) )';
cross_idx = [];
for ii = 1:xFold
    tmp = (ii-1)*size_each_part;
    tmp1 = randidx( (tmp+1):(tmp+size_each_part),1);
    cross_idx = cat( 2, cross_idx,idx(tmp1,1) ); 
end
