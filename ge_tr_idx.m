% ���룺
%    idx �� ����x���������������
%    sizeX_ratio �� ��Ҫ��ȡ�ı���
% �����
%     Ϊ���������������Ӽ�����:
%    tr_idx �� ��������ȡ������
%    te_idx �� ʣ�������
%

function [tr_idx te_idx] = ge_tr_idx(idx,sizeX_ratio);
get_size = sizeX_ratio*size(idx,1);
get_size = uint32(get_size);
randidx = randperm(size(idx,1))';
tr_idx = idx(randidx(1:get_size,1),:);
te_idx = idx(randidx((get_size+1):size(idx,1),1),:);
