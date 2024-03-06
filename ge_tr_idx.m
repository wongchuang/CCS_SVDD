% 输入：
%    idx — 输入x矩阵的行索引向量
%    sizeX_ratio — 需要获取的比例
% 输出：
%     为行索引向量两个子集向量:
%    tr_idx — 按比例获取的索引
%    te_idx — 剩余的索引
%

function [tr_idx te_idx] = ge_tr_idx(idx,sizeX_ratio);
get_size = sizeX_ratio*size(idx,1);
get_size = uint32(get_size);
randidx = randperm(size(idx,1))';
tr_idx = idx(randidx(1:get_size,1),:);
te_idx = idx(randidx((get_size+1):size(idx,1),1),:);
